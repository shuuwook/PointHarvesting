import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for Point Generation via Stochastic Sampling.')

        # Distributed arguments
        self._parser.add_argument('--world_size', default=1, type=int, help='Number of distributed nodes.')
        self._parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str, help='url used to set up distributed training')
        self._parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
        self._parser.add_argument('--distributed', action='store_true', 
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')
        self._parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
        self._parser.add_argument('--gpu', default=None, type=int, help='GPU id to use. None means using all available GPUs.')

        # Dataset arguments
        self._parser.add_argument('--dataset_type', type=str, default="shapenet15k", help="Dataset types.", choices=['shapenet15k', 'modelnet40_15k', 'modelnet10_15k'])
        self._parser.add_argument('--dataset_dir', type=str, default='data/ShapeNetCore.v2.PC15k', help='Dataset file path.')
        self._parser.add_argument('--cates', type=str, nargs='+', default=["airplane"], help="Categories to be trained (useful only if 'shapenet' is selected)")

        self._parser.add_argument('--normalize_per_shape', action='store_true', help='Whether to perform normalization per shape.')
        self._parser.add_argument('--normalize_std_per_axis', action='store_true', help='Whether to perform normalization per axis.')
        self._parser.add_argument("--tr_max_sample_points", type=int, default=2048, help='Max number of sampled points (train)')
        self._parser.add_argument("--te_max_sample_points", type=int, default=2048, help='Max number of sampled points (test)')
        self._parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading threads')

        self._parser.add_argument('--batch_size', type=int, default=150, help='Integer value for batch size.')

        self._parser.add_argument('--start_samples', type=int, default=512, help='Integer value for number of points when it starts.')
        self._parser.add_argument('--end_samples', type=int, default=2048, help='Integer value for number of points when it ends.')
        self._parser.add_argument('--step_num', type=int, default=8, help='Integer value for steps of progress.')

        # Training arguments
        self._parser.add_argument('--epochs', type=int, default=2000, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--ckpt_path', type=str, default='./model/checkpoints/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='ph_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--ckpt_load', type=str, default='ph_ckpt_plane_810.pt', help='Checkpoint name to load. (default:None)')
        self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')

        # Evaluation arguments
        self._parser.add_argument('--JSD', default=True, dest='JSD', action='store_false')
        self._parser.add_argument('--ALL', default=True, dest='ALL', action='store_false')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[64, 512, 512, 512, 512, 512,  512], nargs='+', help='Features for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3,  32,  128, 256, 512, 1024, 2048], nargs='+', help='Features for discriminator.')
        self._parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale for sampling.')

    def parser(self):
        return self._parser