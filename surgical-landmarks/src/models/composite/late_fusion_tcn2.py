#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LF_MH_MS_TCN2(nn.Module):
    def __init__(self, fused_tcn2, left_hand_tcn2, right_hand_tcn2):
        super(LF_MH_MS_TCN2, self).__init__()
        self.left_hand_tcn2 = left_hand_tcn2
        self.right_hand_tcn2 = right_hand_tcn2
        self.fused_tcn2 = fused_tcn2
    
    def _fuse(self, left_usage, right_usage, features):
        # print(features.shape)
        # print(left_usage.shape)
        # print(right_usage.shape)
        return torch.concat((features, left_usage, right_usage), dim=1)
    
    def forward(self, x, mask):
        left_output = self.left_hand_tcn2(x, mask)
        right_right = self.right_hand_tcn2(x, mask)
        fused = self._fuse(left_output[-1], right_right[-1], x)
        output = self.fused_tcn2(fused, mask)
        return output, left_output, right_right



class LF_LSTM_MH_MS_TCN2(nn.Module):
    
    def __init__(self, fused_tcn2, left_hand_tcn2, right_hand_tcn2, lstml, lstmr):
        super(LF_LSTM_MH_MS_TCN2, self).__init__()
        self.left_hand_tcn2 = left_hand_tcn2
        self.right_hand_tcn2 = right_hand_tcn2
        self.fused_tcn2 = fused_tcn2
        self.lstm_left_hand = lstml
        self.lstm_right_hand = lstmr
    
    def _fuse(self, left_usage, right_usage, features, mask):
        # print(features.shape)
        # print(left_usage.shape)
        # print(right_usage.shape)
        left_usage_out = self.lstm_left_hand(left_usage, mask).squeeze(0)
        right_usage_out = self.lstm_right_hand(right_usage, mask).squeeze(0)
        # print(left_usage_out.shape)
        # print(right_usage_out.shape)
        # print(features.shape)        
        return torch.concat((features, left_usage_out, right_usage_out), dim=1)
    
    def forward(self, x, mask):
        # print(x.shape)
        left_output = self.left_hand_tcn2(x, mask)
        # print(left_output.shape)
        right_right = self.right_hand_tcn2(x, mask)
        fused = self._fuse(left_output[-1], right_right[-1], x, mask)
        output = self.fused_tcn2(fused, mask)
        return output, left_output, right_right