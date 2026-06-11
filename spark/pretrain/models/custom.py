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
#f#rom  HG.HGBlock import HGStem,HGBlock
#from  HG.block import DWConv
#from v9back.common import *


class YourConvNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlist=nn.ModuleList(
            [Silence(),
             Bbackbone(),
             ]
        )
        self.d0=  Down0(64)
        self.d1 = Down1(128)
        self.d2 = Down2(256)
        self.d3 = Down3(512)
        self.d4 = Down4(1024)
        self.alld = [self.d0,self.d1,self.d2,self.d3,self.d4]
        self.cblinear1 = CBLinear(64,[64])
        self.cblinear3 = CBLinear(128, [64, 128])
        self.cblinear5 = CBLinear(256, [64, 128, 256])
        self.cblinear7 = CBLinear(512, [64, 128, 256, 512])
        self.cblinear9 = CBLinear(1024, [64, 128, 256, 512, 1024])
        self.allcblinear = [self.cblinear1,self.cblinear3,self.cblinear5,self.cblinear7,self.cblinear9]
        # # conv down 1
        self.conv1 = Conv(3, 64, 3, 2 )
        self.cbfuse1 = CBFuse([0, 0, 0, 0, 0])

        ## conv down 2
        self.conv2= Conv(64, 128, 3, 2)
        self.cbfuse2 = CBFuse([1, 1, 1, 1])
        self.rep2 = RepNCSPELAN4(128, 256, 128, 64, 2)
        ##   avg-conv down fuse 1
        self.adown3 = ADown(256, 256)
        self.cbfuse3 = CBFuse([2, 2, 2])
        self.rep3 = RepNCSPELAN4(256, 512, 256, 128, 2)

        ##   avg-conv down fuse 2
        self.adown4 = ADown(512, 512)
        self.cbfuse4 = CBFuse([3,3])
        self.rep4 = RepNCSPELAN4(512, 1024, 512, 256, 2)

        ##   avg-conv down fuse 3
        self.adown5 = ADown(1024, 1024)
        self.cbfuse5 = CBFuse([4])
        self.rep5 = RepNCSPELAN4(1024, 1024, 512, 256, 2)

    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return [  256,512,1024,1024]

    def forward(self, x: torch.Tensor, hierarchical=False):
        if hierarchical:
            origin = x.clone()
            ls = []
            tmp = []
            bx = None
            for index,modules in enumerate( self.mlist):
                x = modules(x)
                if index ==1:
                    bx = x
            for i in  range(5):
                tmp.append(self.allcblinear[i](self.alld[i](bx)))

            fuse1 = self.cbfuse1([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],self.conv1(origin)])
            fuse2 = self.cbfuse2([tmp[1],tmp[2],tmp[3],tmp[4],self.conv2(fuse1)])
            fuse2 = self.rep2(fuse2)

            fuse3= self.cbfuse3([ tmp[2], tmp[3], tmp[4], self.adown3(fuse2)])
            fuse3 = self.rep3(fuse3)

            fuse4 = self.cbfuse4([tmp[3], tmp[4], self.adown4(fuse3)])
            fuse4 = self.rep4(fuse4)

            fuse5 = self.cbfuse5([tmp[4], self.adown5(fuse4)])
            fuse5 = self.rep5(fuse5)

            ls.append(fuse2)
            ls.append(fuse3)
            ls.append(fuse4)
            ls.append(fuse5)
            return ls
        else:
            for modules in self.mlist:
                x = modules(x)
        return x


@register_model
def V9back(pretrained=False, **kwargs):
    return YourConvNet(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('V9back')
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
