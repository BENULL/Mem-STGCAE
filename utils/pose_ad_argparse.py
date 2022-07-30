import os
import time
import argparse


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)


def init_sub_args(args):
    if args.debug:
        args.epochs = 5

    args.pose_path = {'train': os.path.join(args.data_dir, 'pose', 'training/tracked_person/'),
                      'test':  os.path.join(args.data_dir, 'pose', 'testing/tracked_person/')}

    args.ckpt_dir = create_exp_dirs(args.exp_dir)
    res_args = args_rm_prefix(args, 'res_')
    return args, res_args


def init_parser(default_data_dir='data/ShanghaiTech', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser("Pose_AD_Experiment")
    # General Args
    parser.add_argument('--debug', action='store_true',
                        help='Debug experiment script with minimal epochs. (default: False)')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV',
                        help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999,
                        help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1],
                        help='Verbosity [1/0] (default: 1)')

    parser.add_argument('--hr', action='store_true',
                        help='Use human related dataset. (default: False)')
    parser.add_argument('--headless', action='store_true',
                        help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DROPOUT',
                        help='Dropout training Parameter (default: 0.3)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR',
                        help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                        help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=8, metavar='W',
                        help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH',
                        help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=12, metavar='SGLEN',
                        help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=6, metavar='SGST',
                        help='Stride for training segment sliding window (default: 8)')

    parser.add_argument('--in_channels', type=int, default=3,
                        help='channels of model input (3=include confidence) (default: 3)')
    parser.add_argument('--alpha', '-a', type=float, default=1e-3, metavar='G',
                        help='Alpha value for weighting L2 regularization (default: 1e-3)')
    parser.add_argument('--gamma', '-g', type=float, default=0.6, metavar='G',
                        help='Gamma values for weighting loss (default: 0.6)')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', metavar='OPT',
                        help="Optimizer (default: 'adam')")
    parser.add_argument('--sched', '-s', type=str, default='tri', metavar='SCH',
                        help="Optimization LR scheduler (default: 'tri')")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='Optimizer Learning Rate Parameter (default: 1e-4)')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5, metavar='WD',
                        help='Optimizer Weight Decay Parameter (default: 1e-5)')
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.99, metavar='LD',
                        help='Optimizer Learning Rate Decay Parameter (default: 0.99)')
    parser.add_argument('--test_every', type=int, default=20, metavar='T',
                        help='How many epochs between test evaluations (default: 20)')
    parser.add_argument('--epochs', '-e', type=int, default=10, metavar='E',
                        help='Number of epochs per cycle. (default: 10)')
    parser.add_argument('--batch_size', '-b', type=int, default=512, metavar='B',
                        help='Batch sizes. (default: 512)')

    # Scoring
    parser.add_argument('--sigma', type=int, default=20,
                        help='sigma for guassian filter (default: 20)')
    parser.add_argument('--save_results', type=int, default=1, metavar='SR', choices=[0, 1],
                        help='Save results to npz (default: 1)')   #
    parser.add_argument('--res_batch_size', '-res_b', type=int, default=256,  metavar='B',
                        help='Batch size for scoring. (default: 256)')

    # Visualization
    parser.add_argument('--vis_output', action='store_true', default=False,
                        help='Visualization model output. (default: False)')

    return parser


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            ae_key = key[len(prefix):]
            wp_args_dict[ae_key] = value

    return wp_args


def create_exp_dirs(experiment_dir):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, time_str)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints/')
    dirs = [checkpoints_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return checkpoints_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)
