# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model
import torch
from torch import nn
import sys
from  HG.HGBlock import HGStem,HGBlock
from  HG.block import DWConv


class YourConvNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlist=nn.ModuleList(
            [HGStem(3, 32, 64),
            HGBlock(64, 64, 128, 3, n=6),

            DWConv(128, 128, 3, 2, 1, False),
            HGBlock(128, 128, 512, 3, n=6),
            HGBlock(512, 128, 512, 3, lightconv=False,shortcut=True,n=6),


            DWConv(512, 512, 3, 2, 1, False),
            HGBlock(512, 256, 1024, 5,lightconv=True,shortcut=False,n=6),
            HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
            HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
            HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
            HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),



            DWConv(1024, 1024, 3, 2, 1, False),
            HGBlock(1024, 512, 2048, 5, lightconv=True, shortcut=False, n=6),
             HGBlock(2048, 512, 2048, 5, lightconv=True, shortcut=True, n=6)
             ]
        )

    
    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return [128,512,1024,2048]

    def forward(self, x: torch.Tensor, hierarchical=False):
        if hierarchical:
            ls = []
            for index,modules in enumerate( self.mlist):
                x = modules(x)
                if index in [1,4,10,13]:
                    ls.append(x)
            return ls
        else:
            for modules in self.mlist:
                x = modules(x)
        return x


@register_model
def HGNetv2(pretrained=False, **kwargs):
    return YourConvNet(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('HGNetv2')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
