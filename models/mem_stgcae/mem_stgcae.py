#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn
from models.mem_stgcae.two_branch_decoder import TwoBranchEncoderRNN
from models.stgcn import STGCN
from models.memory_module import MemModule


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.in_channels = args.in_channels
        self.headless = args.headless
        self.seg_len = args.seg_len//2
        self.gcn = STGCN(in_channels=self.in_channels, headless=self.headless, seg_len=self.seg_len)
        self.mlp_input_size = self.in_channels * (14 if self.headless else 18)
        self.ae_hidden_size = self.mlp_input_size * self.seg_len 
        self.encoder = TwoBranchEncoderRNN(input_size=self.mlp_input_size,
                                           hidden_size=self.ae_hidden_size,
                                           num_layers=2)

        self.mlp_in = nn.Sequential(
            nn.Linear(self.mlp_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.mlp_input_size),
        )

        self.rec_mlp_out = nn.Sequential(
            nn.Linear(self.ae_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.mlp_input_size),
        )
        self.pre_mlp_out = nn.Sequential(
            nn.Linear(self.ae_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.mlp_input_size),
        )
        self.perceptual_linear = nn.Sequential(
            nn.Linear(self.mlp_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.mlp_input_size),
        )

        self.mem_bank1 = MemModule(mem_dim=6000, fea_dim=18)
        self.mem_bank3 = MemModule(mem_dim=6000, fea_dim=18)

    def forward(self, x):
        N, C, T, V = x.size()
        gcn_out1, gcn_out2, gcn_out3 = self.gcn(x)
        gcn_out1 = gcn_out1.reshape(-1, 18, 6, 3).contiguous()
        # gcn_out2 = gcn_out2.reshape(-1, 18, 6, 3).contiguous()
        gcn_out3 = gcn_out3.reshape(-1, 18, 1, 3).contiguous()

        gcn_out1, _ = self.mem_bank1(gcn_out1)
        gcn_out1 = gcn_out1.reshape(N, -1).contiguous()
        # gcn_out2, _ = self.mem_bank2(gcn_out2)
        # gcn_out2 = gcn_out2.reshape(N, -1).contiguous()
        gcn_out3, _ = self.mem_bank3(gcn_out3)
        gcn_out3 = gcn_out3.reshape(N, -1).contiguous()
        ae_in = torch.stack((gcn_out3, gcn_out1), dim=0)

        rec_out, pre_out = self.encoder(ae_in)
        rec_out = self.rec_mlp_out(rec_out)
        pre_out = self.pre_mlp_out(pre_out)
        rec_out = torch.flip(rec_out, dims=[1])
        local_out = torch.cat((rec_out, pre_out), dim=1)
        perceptual_out = self.perceptual_linear(local_out)

        local_out = local_out.reshape(N, C, -1, V)
        perceptual_out = perceptual_out.reshape(N, C, -1, V)

        return local_out, perceptual_out


if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(**{'in_channels':3,'headless':False,'seg_len':12})
    model = Model(args)
    data = torch.ones((2, 3, 6, 18))
    out = model(data)
    print(out[0].shape)
