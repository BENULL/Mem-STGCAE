import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import normalize_pose, re_normalize_pose
from utils.draw_utils import draw_mask_skeleton



class Trainer:
    def __init__(self, args, model, loss, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        self.loss = loss
        self.motion_loss = nn.SmoothL1Loss()
        self.writer = SummaryWriter(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 f'log/vis/{args.ckpt_dir.split("/")[-3]}'))

        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr, )

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.lr, self.args.lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1,
                'state_dict': self.model.state_dict(),
                optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + '_checkpoint.pth.tar'
        if num_epochs is None:  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = self.args.epochs
        else:
            start_epoch = 0

        self.model.train()
        self.model = self.model.to(args.device)
        for epoch in range(start_epoch, num_epochs):
            loss_list = []
            local_loss_list = []
            motion_loss_list = []
            reg_loss_list = []
            perceptual_loss_list = []
            ep_start_time = time.time()
            print("Started epoch {}".format(epoch))
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:, :args.in_channels, :, :]
                local_data, pose_xy_perceptual, xy_global, bounding_box_wh = normalize_pose(data)

                x = local_data[:, :, :args.seg_len // 2, :]
                local_out, perceptual_out = self.model(x)

                local_loss = self.loss(local_out, local_data)  # [N, C, T, V]
                local_loss = torch.mean(local_loss)

                perceptual_loss = self.loss(perceptual_out, data)  # [N, C, T, V]
                perceptual_loss = torch.mean(perceptual_loss)

                # motion loss
                N, C, T, V = data.size()
                data_prev_data = torch.cat((torch.zeros(N, C, 1, V).to(data.device), data[:, :, :-1, :]), dim=2)
                motion_y = torch.abs(data - data_prev_data)
                motion_y_hat = torch.abs(perceptual_out - data_prev_data)
                motion_loss = self.motion_loss(motion_y, motion_y_hat)

                reg_loss = calc_reg_loss(self.model)
                motion_loss = motion_loss / (motion_loss / local_loss).detach()
                perceptual_loss = perceptual_loss / (perceptual_loss / local_loss).detach()
                loss = 4 * local_loss + 1 * perceptual_loss + 1 * motion_loss + 1e-3 * args.alpha * reg_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())
                perceptual_loss_list.append(perceptual_loss.item())
                motion_loss_list.append(motion_loss.item())
                local_loss_list.append(local_loss.item())
                reg_loss_list.append(reg_loss.item())

            print("Epoch {0} done, loss: {1:.7f}, took: {2:.3f}sec".format(epoch, np.mean(loss_list),
                                                                           time.time() - ep_start_time))

            new_lr = self.adjust_lr(epoch)
            print('lr: {0:.3e}'.format(new_lr))
            self.writer.add_scalar('training total loss', np.mean(loss_list), epoch)
            self.writer.add_scalar('training reg loss', np.mean(reg_loss_list), epoch)
            self.writer.add_scalar('training predict loss', np.mean(local_loss_list), epoch)
            self.writer.add_scalar('training motion loss', np.mean(motion_loss_list), epoch)
            self.writer.add_scalar('training perceptual loss', np.mean(perceptual_loss_list), epoch)
            self.writer.add_scalar('lr', new_lr, epoch)

            self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)

            if (epoch+1) % args.test_every == 0:
                self._test(epoch, self.test_loader, ret_output=False, log=False, args=args)
        return checkpoint_filename, np.mean(loss_list)

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state

    def test(self, cur_epoch, ret_output=False, log=True, args=None):
        return self._test(cur_epoch, self.test_loader, ret_output=ret_output, log=log, args=args)

    def _test(self, cur_epoch, test_loader, ret_output=True, log=True, args=None):
        print("Testing")
        self.model.eval()
        output_arr = []
        ret_reco_loss_arr = []
        reco_loss_arr = []
        for itern, data_arr in enumerate(test_loader):
            with torch.no_grad():
                data = data_arr[0].to(args.device)
                data = data[:, :args.in_channels, :, :]
                local_data, pose_xy_perceptual, xy_global, bounding_box_wh = normalize_pose(data)

                x = local_data[:, :, :args.seg_len // 2, :]

                local_out, perceptual_out = self.model(x)
                origin_pose, _ = re_normalize_pose(local_out, perceptual_out, xy_global, bounding_box_wh)

                if ret_output:
                    output_sfmax = origin_pose
                    output_arr.append(output_sfmax.detach().cpu().numpy())
                    del output_sfmax

                for origin, predict in zip(data, perceptual_out):
                    loss = self.loss(origin, predict)
                    ret_loss = torch.mean(loss, (0, 2))
                    ret_reco_loss_arr.append(ret_loss.cpu().numpy())

                reco_loss = self.loss(data, perceptual_out)
                reco_loss = torch.mean(reco_loss)
                reco_loss_arr.append(reco_loss.item())

                if args.vis_output:
                    draw_mask_skeleton(data.cpu().numpy(), perceptual_out.cpu().numpy(), data_arr[2],
                                       args.ckpt_dir.split('/')[2])

        test_loss = np.mean(reco_loss_arr)

        print("--> Test set loss {:.7f}".format(test_loss))
        self.model.train()
        if ret_output:
            return output_arr, ret_reco_loss_arr


def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
        # new_lr = scheduler.get_last_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def csv_log_dump(args, log_dict):
    """
    Create CSV log line, with the following format:
    Date, Time, Seed, seg_stride, seg_len,
    optimizer, dropout, batch_size, epochs, lr, lr_decay, wd,
    loss, alpha (=L2 reg coef), gamma,
    :return:
    """
    try:
        date_time = args.ckpt_dir.split('/')[-3]  # 'Aug20_0839'
        date_str, time_str = date_time.split('_')[:2]
    except:
        date_time = 'parse_fail'
        date_str, time_str = '??', '??'
    param_arr = [date_str, time_str, args.seed, args.seg_stride, args.seg_len,
                 args.optimizer, args.sched, args.dropout, args.batch_size,
                 args.epochs, args.lr, args.lr_decay, args.weight_decay, log_dict['loss'], log_dict['auc'],
                 args.alpha, args.gamma]

    res_str = '_{}'.format(int(10 * log_dict['auc']))
    log_template = len(param_arr) * '{}, ' + '\n'
    log_str = log_template.format(*param_arr)
    debug_str = '_debug' if args.debug else ''
    csv_path = os.path.join(args.ckpt_dir, '{}{}{}_log_dump.csv'.format(date_time, debug_str, res_str))
    with open(csv_path, 'w') as csv_file:
        csv_file.write(log_str)

    experiment_csv_log = os.path.join(args.exp_dir, 'experiment_csv_log.csv')
    with open(experiment_csv_log, "a+") as file:
        file.write(log_str)
