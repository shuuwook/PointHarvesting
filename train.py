import torch
import torch.nn as nn
import torch.optim as optim

from data.datasets import get_datasets, init_np_seed
from model.gan_network import Generator, Discriminator, Identity
from model.gradient_penalty import GradientPenalty

from utils.parallel import DataParallelModel, DataParallelCriterion, Reduce
from utils.progress import ProgressiveSampling
from utils.validation import validate_sample

from arguments import Arguments

import os
import time
import visdom
import numpy as np

class PointHarvesting():
    def __init__(self, args):
        self.args = args

        # ------------------------------------------------Dataset---------------------------------------------- #
        # initialize datasets and loaders
        tr_dataset, te_dataset = get_datasets(args)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=tr_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True, worker_init_fn=init_np_seed)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False, worker_init_fn=init_np_seed)

        print("Training Dataset : {} prepared.".format(len(tr_dataset)))
        print("Test Dataset : {} prepared.".format(len(te_dataset)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        if torch.cuda.device_count() > 1:
            self.I = DataParallelModel(Identity()).to(args.device)
            self.G = DataParallelModel(Generator(feature=args.G_FEAT, noise_scale=args.noise_scale)).to(args.device)
            self.D = DataParallelCriterion(Discriminator(feature=args.D_FEAT)).to(args.device)
        else:
            self.I = Identity().to(args.device)
            self.G = Generator(feature=args.G_FEAT, noise_scale=args.noise_scale).to(args.device)
            self.D = Discriminator(feature=args.D_FEAT).to(args.device)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, multi_gpu=True if args.gpu == -1 else False)
        self.Progress = ProgressiveSampling(500, args.step_num, args.start_samples, args.end_samples)
        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        self.vis = visdom.Visdom(port=args.visdom_port)
        assert self.vis.check_connection(), "Can not have with specified {:d} visdom port.".format(args.visdom_port)
        print("Visdom connected.")
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, ngpus_per_node=None, save_ckpt=None, load_ckpt=None, fpd_ckpt=None):
        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())
            
            print("Checkpoint loaded.")

        for epoch in range(epoch_log, self.args.epochs):
            self.Progress.progress_update(epoch)
            sample_num = self.train_loader.dataset.tr_sample_size = self.Progress.get_sample_num()
            for _iter, data in enumerate(self.train_loader):
                # Start Time
                start_time = time.time()
                point, _ = data['train_points'], data['test_points']
                
                real_feat = point.to(self.args.device) # (B,N,3)

                if torch.cuda.device_count() > 1:
                    real_feat = self.I(real_feat)

                # -------------------- Discriminator -------------------- #
                for _ in range(self.args.D_iter):
                    self.optimizerD.zero_grad()
                    
                    with torch.no_grad():
                        z = torch.randn(point.size(0), 1, 64).to(self.args.device)
                        fake_feat = self.G(z, sample_num)

                    D_real = self.D(real_feat)
                    if torch.cuda.device_count() > 1:
                        D_real = Reduce.apply(*D_real) / len(D_real)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_feat)
                    if torch.cuda.device_count() > 1:
                        D_fake = Reduce.apply(*D_fake) / len(D_fake)
                    D_fakem = D_fake.mean()
                    
                    # WGAN loss
                    gp_loss = self.GP(self.D, real_feat, fake_feat)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss

                    (d_loss_gp).backward()

                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())      
                
                # ---------------------- Generator ---------------------- #
                self.optimizerG.zero_grad()
                
                z = torch.randn(point.size(0), 1, 64).to(self.args.device)
                fake_feat = self.G(z, sample_num)

                G_fake = self.D(fake_feat)
                if torch.cuda.device_count() > 1:
                    G_fake = Reduce.apply(*G_fake) / len(G_fake)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem

                (g_loss).backward()

                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())        
                
                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ Points ] ", "{:d}".format(sample_num),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss),
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))


                if _iter % 10 == 0:
                    if torch.cuda.device_count() > 1:
                        generated_point = fake_feat[0][-1,:,:3]
                    else:
                        generated_point = fake_feat[-1,:,:3]

                    plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)

                    self.vis.line(X=plot_X, Y=plot_Y, win=1,
                                  opts={'title': 'PointHarvesting Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})

                    self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], win=2,
                                     opts={'title': "Generated Pointcloud", 'markersize': 1, 'webgl': True})
            
            # ---------------- Evaluation --------------- #
            if (epoch+1) % 10 == 0:
                result = validate_sample(self.test_loader, self.G, self.args, max_samples=None, save_dir=None)
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'result': result,
                }, save_ckpt+self.args.cates[0]+'_'+str(epoch+1)+'.pt')

            

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = 'cuda'
    try:
        print("{} used for learning.".format(os.environ['CUDA_VISIBLE_DEVICES']))
    except:
        print("Setting specific devices is recommended. (default : CUDA_VISIBLE_DEVICES=0)")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        print("{} used for learning.".format(os.environ['CUDA_VISIBLE_DEVICES']))

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None

    model = PointHarvesting(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT)
