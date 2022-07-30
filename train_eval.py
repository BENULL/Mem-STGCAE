import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.optim_utils.optim_init import init_optimizer, init_scheduler
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_ad_argparse import init_parser, init_sub_args
from utils.scoring_utils import score_dataset
from utils.train_utils import Trainer, csv_log_dump
from models.mem_stgcae import MemSTGCAE

def main():
    parser = init_parser()
    args = parser.parse_args()
    log_dict = collections.defaultdict(int)

    torch.autograd.set_detect_anomaly(True)

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args, res_args = init_sub_args(args)
    print(args)

    dataset, loader = get_dataset_and_loader(args)

    model = MemSTGCAE(args)

    loss = nn.MSELoss(reduction='none')
    # loss = nn.SmoothL1Loss(reduction='none')

    optimizer_f = init_optimizer(args.optimizer, lr=args.lr)
    scheduler_f = init_scheduler(args.sched, lr=args.lr, epochs=args.epochs)
    trainer = Trainer(args, model, loss, loader['train'], loader['test'], optimizer_f=optimizer_f,
                      scheduler_f=scheduler_f)

    print(f"train dataset lens {len(loader['train'].dataset)}")
    print(f"test dataset lens {len(loader['test'].dataset)}")
    fn, log_dict['loss'] = trainer.train(args=args)
    print(f"model loss {log_dict['loss']}")

    output_arr, rec_loss_arr = trainer.test(args.epochs, ret_output=True, args=args)

    max_clip = 5 if args.debug else None

    auc, shift, sigma = score_dataset(rec_loss_arr, loader['test'].dataset.metadata, loader['test'].dataset.person_keys, max_clip=max_clip, args=args)

    # Logging and recording results
    print("{} Done with {} AuC for {} samples and {} trans".format(args.ckpt_dir.split('/')[-3], auc, len(loader['test'].dataset), args.num_transform))
    log_dict['auc'] = 100 * auc

    csv_log_dump(args, log_dict)
    res_npz_path = save_result_npz(args, output_arr, rec_loss_arr, auc)



def get_dataset_and_loader(args):

    dataset_args = {'debug': args.debug, 'headless': args.headless,
                    'seg_len': args.seg_len,
                    'return_indices': True, 'return_metadata': True, 'hr': args.hr}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': False}

    dataset, loader = dict(), dict()
    for split in ['train', 'test']:
        dataset_args['seg_stride'] = args.seg_stride if split is 'train' else 1  # No strides for test set
        dataset_args['train_seg_conf_th'] = args.train_seg_conf_th if split is 'train' else 0.0
        dataset[split] = PoseSegDataset(args.pose_path[split], **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    return dataset, loader


def save_result_npz(args, output_arr, rec_loss_arr, auc):
    debug_str = '_debug' if args.debug else ''
    res_fn = f'res_{int(auc*10000)}_{debug_str}.npz'
    res_path = os.path.join(args.ckpt_dir, res_fn)
    np.savez(res_path, output_arr=output_arr, args=args, rec_loss_arr=rec_loss_arr)
    return res_path


if __name__ == '__main__':
    main()

