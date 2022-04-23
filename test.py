from data.datasets import get_datasets, synsetid_to_cate
from arguments import Arguments
from pprint import pprint
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from model.gan_network import Generator
import os
import torch
import numpy as np
import torch.nn as nn

def get_test_loader(args):
    _, te_dataset = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def evaluate_gen(model, args):
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    for data in loader:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        
        z = torch.randn(B, 1, 64).to(args.device)
        out_pc = model(z, N)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(args, load_ckpt=None):
    # -------------------------------------------------Module---------------------------------------------- #
    model = Generator(feature=args.G_FEAT, noise_scale=0.).to(args.device)
    print("Network prepared.")
    # ----------------------------------------------------------------------------------------------------- #

    print("Resume Path:%s" % load_ckpt)
    checkpoint = torch.load(load_ckpt)
    model.load_state_dict(checkpoint['G_state_dict'])
    model.eval()

    with torch.no_grad():
        evaluate_gen(model, args)


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

    main(args, load_ckpt=LOAD_CHECKPOINT)