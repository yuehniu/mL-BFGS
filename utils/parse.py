# util function for parsing running commands
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument( '--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                         help='name of dataset' )
    parser.add_argument( '--data', metavar='DIR', help='path to dataset' )
    parser.add_argument( '--model', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'deit'] )
    parser.add_argument( '--phases', type=str,
                        help='Specify epoch order of data resize and learning rate schedule: [{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]')
    # parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument( '--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM', 'SLIM', 'SLIMBLOCK', 'KFAC'],
                         help='optimization method' )
    parser.add_argument( '-j', '--workers', default=4, type=int, metavar='N',
                         help='number of data loading workers (default: 4)' )
    parser.add_argument( '--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)' )
    parser.add_argument( '--lr', default=0.1, type=float, help='initial learning rate' )
    parser.add_argument( '--decay-period', default=10, type=int, help='lr rate decay period' )
    parser.add_argument( '--lr-decay', default=0.5, type=float, help='lr decay scale' )
    parser.add_argument( '--decay-steps', nargs='+', type=int, default=[25, 35, 40, 45, 50],
                         help='epoch intervals to decay lr (default: 25,35,40,45,50)' )
    parser.add_argument( '--momentum', default=0.9, type=float, metavar='M', help='momentum' )
    parser.add_argument( '--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)' )
    parser.add_argument( '--max-epoch', default=100, type=int, help='max epochs' )
    parser.add_argument( '--init-bn0', action='store_true', help='Intialize running batch norm mean to 0' )
    parser.add_argument( '--print-freq', '-p', default=20, type=int,
                         metavar='N', help='log/print every this many steps (default: 5)' )
    parser.add_argument( '--no-bn-wd', action='store_true', help='Remove batch norm from weight decay' )
    parser.add_argument( '--resume', default='', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)' )
    parser.add_argument( '-e', '--evaluate', dest='evaluate', action='store_true',
                         help='evaluate model on validation set' )
    parser.add_argument( '--fp16', action='store_true', help='Run model fp16 mode. Default True' )
    parser.add_argument( '--loss-scale', type=float, default=1024,
                         help='Loss scaling, positive power of 2 values can improve fp16 convergence.' )
    parser.add_argument( '--distributed', action='store_true', help='Run distributed training. Default True' )
    parser.add_argument( '--dist-url', default='env://', type=str,
                         help='url used to set up distributed training' )
    parser.add_argument( '--dist-backend', default='nccl', type=str, help='distributed backend' )
    parser.add_argument( '--local_rank', default=0, type=int,
                         help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.' )
    parser.add_argument( '--logdir', default='', type=str, help='where logs go' )

    parser.add_argument( '--grad-clip', default=0.05, type=float, help='gradient clipping' )

    # LBFGS hyper parameters
    parser.add_argument( '--mm-p', default=0.9, type=float, help='stat decay for parameters' )
    parser.add_argument( '--mm-g', default=0.9, type=float, help='stat decay for gradients' )
    parser.add_argument( '--mmm', default=0.8, type=float, help='double momentum' )
    parser.add_argument( '--update-freq', default=200, type=int, help='update frequency for Hessian approximation' )
    parser.add_argument( '--hist-sz', default=20, type=int, help='hisotry size for LBFGS-related vectors' )
    parser.add_argument( '--lbfgs-damping', default=0.2, type=float, help='LBFGS damping factor' )

    # KFAC hyper parameters
    parser.add_argument( '--kfac-update-freq', type=int, default=100,
                         help='iters between kfac inv ops (0 disables kfac) (default: 100)' )
    parser.add_argument( '--kfac-cov-update-freq', type=int, default=10,
                         help='iters between kfac cov ops (default: 10)' )
    parser.add_argument( '--kfac-update-freq-alpha', type=float, default=10,
                         help='KFAC update freq multiplier (default: 10)' )
    parser.add_argument( '--kfac-update-freq-decay', nargs='+', type=int, default=None,
                         help='KFAC update freq decay schedule (default None)' )
    parser.add_argument( '--use-inv-kfac', action='store_true', default=False,
                         help='Use inverse KFAC update instead of eigen (default False)' )
    parser.add_argument( '--stat-decay', type=float, default=0.95,
                         help='Alpha value for covariance accumulation (default: 0.95)' )
    parser.add_argument( '--damping', type=float, default=0.001,
                         help='KFAC damping factor (defaultL 0.001)' )
    parser.add_argument( '--damping-alpha', type=float, default=0.5,
                         help='KFAC damping decay factor (default: 0.5)' )
    parser.add_argument( '--damping-decay', nargs='+', type=int, default=None,
                         help='KFAC damping decay schedule (default None)' )
    parser.add_argument( '--kl-clip', type=float, default=0.001,
                         help='KL clip (default: 0.001)' )
    parser.add_argument( '--skip-layers', nargs='+', type=str, default=[],
                         help='Layer types to ignore registering with KFAC (default: [])' )
    parser.add_argument( '--coallocate-layer-factors', action='store_true', default=True,
                         help='Compute A and G for a single layer on the same worker. ' )
    parser.add_argument( '--kfac-comm-method', type=str, default='comm-opt',
                         help='KFAC communication optimization strategy. One of comm-opt, '
                             'mem-opt, or hybrid_opt. (default: comm-opt)' )
    parser.add_argument( '--kfac-grad-worker-fraction', type=float, default=0.25,
                         help='Fraction of workers to compute the gradients '
                             'when using HYBRID_OPT (default: 0.25)' )
    return parser