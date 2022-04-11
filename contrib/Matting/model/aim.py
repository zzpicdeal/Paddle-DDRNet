import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

def get_masked_local_from_global(global_sigmoid, local_sigmoid):
	values, index = paddle.max(global_sigmoid,1)
	index = index[:,None,:,:].float()
	### index <===> [0, 1, 2]
	### bg_mask <===> [1, 0, 0]
	bg_mask = index.clone()
	bg_mask[bg_mask==2]=1
	bg_mask = 1- bg_mask
	### trimap_mask <===> [0, 1, 0]
	trimap_mask = index.clone()
	trimap_mask[trimap_mask==2]=0
	### fg_mask <===> [0, 0, 1]
	fg_mask = index.clone()
	fg_mask[fg_mask==1]=0
	fg_mask[fg_mask==2]=1
	fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
	return fusion_sigmoid

def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2D(in_channels,out_channels,3,padding=1),
        nn.BatchNorm(out_channels),
        nn.ReLU(),
        nn.Upsample(scale_factor=up_sample, mode='bilinear'))

class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PSPModule(nn.Layer):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.LayerList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2D(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2D(features, features, kernel_size=1, bias_attr=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(paddle.cat(priors, 1))
        return self.relu(bottle)

class AimNet(nn.Layer):
    def __init__(self,backbone=resnet34_mp):
        super(AimNet).__init__()
        self.resnet = backbone
        ##########################
        ### Encoder part - RESNET
        ##########################
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            )
        self.mp0 = self.resnet.maxpool1
        self.encoder1 = nn.Sequential(
            self.resnet.layer1
            )
        self.mp1 = self.resnet.maxpool2
        self.encoder2 = self.resnet.layer2
        self.mp2 = self.resnet.maxpool3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5
        ##########################
        ### Decoder part - GLOBAL
        ##########################
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)
        self.decoder4_g = nn.Sequential(
            nn.Conv2D(1024,512,3,padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,512,3,padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder4_g_se = SELayer(256)
        self.decoder3_g = nn.Sequential(
            nn.Conv2D(512,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Conv2D(256,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Conv2D(256,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder3_g_se = SELayer(128)
        self.decoder2_g = nn.Sequential(
            nn.Conv2D(256,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder2_g_se = SELayer(64)
        self.decoder1_g = nn.Sequential(
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder1_g_se = SELayer(64)
        self.decoder0_g = nn.Sequential(
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder0_g_spatial = nn.Conv2D(2,1,7,padding=3)
        self.decoder0_g_se = SELayer(64)
        self.decoder_final_g = nn.Conv2D(64,3,3,padding=1)
        ##########################
        ### Decoder part - LOCAL
        ##########################
        self.bridge_block = nn.Sequential(
            nn.Conv2D(512,512,3,dilation=2, padding=2),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,512,3,dilation=2, padding=2),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,512,3,dilation=2, padding=2),
            nn.BatchNorm(512),
            nn.ReLU())
        self.decoder4_l = nn.Sequential(
            nn.Conv2D(1024,512,3,padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,512,3,padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Conv2D(512,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU())
        self.decoder3_l = nn.Sequential(
            nn.Conv2D(512,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Conv2D(256,256,3,padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Conv2D(256,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU())
        self.decoder2_l = nn.Sequential(
            nn.Conv2D(256,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128,128,3,padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU())
        self.decoder1_l = nn.Sequential(
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU())
        self.decoder0_l = nn.Sequential(
            nn.Conv2D(128,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv2D(64,64,3,padding=1),
            nn.BatchNorm(64),
            nn.ReLU())

        self.decoder_final_l = nn.Conv2D(64,1,3,padding=1)

        
    def forward(self, input):

        #####################################
        ### Encoder part - MODIFIED RESNET
        #####################################
        e0 = self.encoder0(input)
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, id2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, id3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, id4 = self.mp4(e3)
        e4 = self.encoder4(e4p)
        #####################################
        ### Decoder part - GLOBAL: Semantic
        #####################################
        psp = self.psp_module(e4)
        d4_g = self.decoder4_g(paddle.cat((psp,e4),1))
        d4_g = self.decoder4_g_se(d4_g)
        d3_g = self.decoder3_g(paddle.cat((self.psp4(psp),d4_g),1))
        d3_g = self.decoder3_g_se(d3_g)
        d2_g = self.decoder2_g(paddle.cat((self.psp3(psp),d3_g),1))
        d2_g = self.decoder2_g_se(d2_g)
        d1_g = self.decoder1_g(paddle.cat((self.psp2(psp),d2_g),1))
        d1_g = self.decoder1_g_se(d1_g)
        d0_g = self.decoder0_g(paddle.cat((self.psp1(psp),d1_g),1))
        d0_g_avg = paddle.mean(d0_g, dim=1,keepdim=True)
        d0_g_max, _ = paddle.max(d0_g, dim=1,keepdim=True)
        d0_g_cat = paddle.cat([d0_g_avg, d0_g_max], dim=1)
        d0_g_spatial = self.decoder0_g_spatial(d0_g_cat)
        d0_g_spatial_sigmoid = F.sigmoid(d0_g_spatial)
        d0_g = self.decoder0_g_se(d0_g)
        d0_g = self.decoder_final_g(d0_g)
        global_sigmoid = F.sigmoid(d0_g)
        #####################################
        ### Decoder part - LOCAL: Matting
        #####################################
        bb = self.bridge_block(e4)
        d4_l = self.decoder4_l(paddle.cat((bb, e4),1))
        d3_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        d3_l = self.decoder3_l(paddle.cat((d3_l, e3),1))
        d2_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)
        d2_l = self.decoder2_l(paddle.cat((d2_l, e2),1))
        d1_l  = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)
        d1_l = self.decoder1_l(paddle.cat((d1_l, e1),1))
        d0_l  = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        d0_l  = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        d0_l = self.decoder0_l(paddle.cat((d0_l, e0),1))
        d0_l = d0_l+d0_l*d0_g_spatial_sigmoid
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)
        ##########################
        ### Fusion net - G/L
        ##########################
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid
@manager.MODELS.add_component
def AimNet_imagenet(backbone,pretrained=None):
    model = AimNet(backbone)
    if pretrained:
        pretrained_state = paddle.load(pretrained)
        model.set_state_dict(pretrained_state )
    return model