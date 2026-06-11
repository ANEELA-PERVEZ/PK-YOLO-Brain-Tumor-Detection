# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from utils.misc import is_pow2n


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )
    
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, upsample_ratio, sbn=False, width=640): # width ko 640 karein (aapka max channel)
        super().__init__()
        self.width = width
        assert is_pow2n(upsample_ratio)
        n = round(math.log2(upsample_ratio))
        channels = [self.width // 2 ** i for i in range(n + 1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = None
        for i, d in enumerate(self.dec):
            # 1. Sahi resolution aur channel wala tensor dhoondna
            current_expected_cin = d.upsample.in_channels
            
            matched_tensor = None
            for t in to_dec:
                if t is not None and t.shape[1] == current_expected_cin:
                    matched_tensor = t
                    break
            
            # 2. Agar x pehle se hai toh usme add karein, warna naya tensor lein
            if matched_tensor is not None:
                if x is None:
                    x = matched_tensor
                else:
                    # Agar spatial size (H, W) alag hai toh resize karke add karein
                    if x.shape[2:] != matched_tensor.shape[2:]:
                        x = torch.nn.functional.interpolate(x, size=matched_tensor.shape[2:], mode='nearest')
                    x = x + matched_tensor
            
            # 3. Agar abhi bhi x khali hai toh loop skip karein
            if x is None:
                continue
                
            # 4. Upsample aur Conv block chalayein
            x = self.dec[i](x)
            
        return self.proj(x)
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
