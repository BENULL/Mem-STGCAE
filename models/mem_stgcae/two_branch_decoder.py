#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午3:44
"""
import torch
import torch.nn as nn
import numpy as np


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, output_size, num_layers, batch_first=True,
                          dropout=0.2, bidirectional=bidirectional)
        # initialize weights
        nn.init.xavier_uniform_(self.gru.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.gru.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x, h_0):
        # forward propagate lstm
        out, _ = self.gru(x, h_0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class TwoBranchEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):
        super(TwoBranchEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rec_decoder = DecoderRNN(hidden_size, hidden_size, num_layers, bidirectional)
        self.pre_decoder = DecoderRNN(hidden_size, hidden_size, num_layers, bidirectional)

    def forward(self, x):
        _, N, K = x.size()
        T = K // 3 // 18
        encoder_h_n = x
        rec_decoder_h_0 = encoder_h_n.clone()
        pre_decoder_h_0 = encoder_h_n.clone()

        rec_input = torch.zeros((N, T, self.hidden_size)).to(x.device)
        pre_input = rec_input.clone()

        rec_out = self.rec_decoder(rec_input, rec_decoder_h_0)
        pre_out = self.pre_decoder(pre_input, pre_decoder_h_0)
        return rec_out, pre_out


if __name__ == '__main__':
    # N, T, V*C
    data = torch.randn((256, 6, 54))
    data = data.to(0)
    model = TwoBranchEncoderRNN(input_size=54, hidden_size=20)
    out = model(data)
    print(out.shape)
