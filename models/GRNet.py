"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
Modified version (MRCNet) by Hao Wang
Modified version (GRNet) by Zi Wang
"""


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class AttentionCorrelation(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionCorrelation, self).__init__()
        self.query_transform = nn.Conv2d(feature_dim, feature_dim, kernel_size=1,stride=2)
        self.key_transform = nn.Conv2d(feature_dim, feature_dim, kernel_size=1,stride=2)
        self.value_transform = nn.Conv2d(feature_dim, feature_dim, kernel_size=1,stride=2)

    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: Tensor of shape (B, C, H, W)
            depth_features: Tensor of shape (B, C, H, W)
        Returns:
            attention_scores: Tensor of shape (B, H*W, H*W)
            output_features: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = rgb_features.shape

        # Transform to Query, Key, and Value
        Q = self.query_transform(rgb_features).reshape(B, C, -1)  # (B, C, 1/4H*W)
        K = self.key_transform(depth_features).reshape(B, C, -1)  # (B, C, 1/4H*W)
        V = self.value_transform(depth_features).reshape(B, C, -1)  # (B, C, 1/4H*W)

        # Compute attention scores
        attention_scores = torch.bmm(Q.permute(0, 2, 1), K)  # (B, 1/4H*W, 1/4H*W)
        attention_weights = F.softmax(attention_scores / (C ** 0.5), dim=-1)  # Softmax normalization

        # Compute output features
        output_features = torch.bmm(attention_weights, V.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H // 2, W // 2)

        return output_features


class AttentionFeatureProcessor(nn.Module):
    def __init__(self, input_dim, fc_hidden_dim, output_dim):
        super(AttentionFeatureProcessor, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.fc2_quaternion = nn.Linear(fc_hidden_dim, 4)  # Quaternion
        self.fc2_translation = nn.Linear(fc_hidden_dim, 3)  # Translation

    def forward(self, attention_outputs):
        """
        Args:
            attention_outputs: List of attention features from different resolutions
                               [(B, C, H1, W1), (B, C, H2, W2), ...]
        Returns:
            quaternion: (B, 4)
            translation: (B, 3)
        """

        flattened_features = [
            feature.reshape(feature.size(0), -1)  # (B, C*H*W)
            for feature in attention_outputs
        ]
        concatenated_features = torch.cat(flattened_features, dim=1)  # (B, total_features)

        # Pass through fully connected layers
        hidden = torch.relu(self.fc1(concatenated_features))
        quaternion = self.fc2_quaternion(hidden)
        translation = self.fc2_translation(hidden)

        return quaternion, translation


class IFNet(nn.Module):
    """
    Based on the MRCNet. fuse all scales of features among each branch, and features from two branches(LiDAR & Cam) interact correspondingly.
    New structures different from MRCNet:
    1.Full FPN fusion;
    2.Cross modal attention;
    3.Mixed Loss(Define in utils.py, named mixed_Q_loss;

    """

    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(IFNet, self).__init__()
        self.toplayer = nn.Conv2d(512, 64, 1, 1, 0)

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth3 = nn.Conv2d(64, 64, 3, 1, 1)

        self.latlayer1 = nn.Conv2d(256, 64, 1, 1, 0)
        self.latlayer2 = nn.Conv2d( 128, 64, 1, 1, 0)
        self.latlayer3 = nn.Conv2d( 64, 64, 1, 1, 0)

        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2

        # original resnet
        self.pretrained_encoder = False
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
            add_list = [1024, 512, 256, 64]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]
            add_list = [256, 128, 64, 64]

        if self.attention:
            block = SEBottleneck
        else:
            if self.res_num == 50:
                block = Bottleneck
            elif self.res_num == 18:
                block = BasicBlock


        # rgb_image
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)

        # lidar_image
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.attcorr6 = AttentionCorrelation(512) #传入之后得到B*512*8*16的特征图
        self.attcorr5 = AttentionCorrelation(64)  #传入之后得到B*64*8*16的特征图
        self.attcorr4 = AttentionCorrelation(64)  #传入之后得到B*64*16*32的特征图
        self.attcorr3 = AttentionCorrelation(64)  #传入之后得到B*64*32*64的特征图
        self.attcorr2 = AttentionCorrelation(64)  #传入之后得到B*64*64*128的特征图





        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = 512
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        # self.conv9_0 = myconv(od, 128, kernel_size=3, stride=1)
        # self.conv9_1 = myconv(128, 64, kernel_size=3, stride=1)
        # self.conv9_2 = myconv(64, 32, kernel_size=3, stride=1)
        # self.conv7_0 = myconv(od, 128, kernel_size=3, stride=1)
        # self.conv7_1 = myconv(128, 64, kernel_size=3, stride=1)
        # self.conv7_2 = myconv(64, 16, kernel_size=3, stride=1)
        # self.conv8_0 = myconv(od, 128, kernel_size=3, stride=1)
        # self.conv8_1 = myconv(128, 64, kernel_size=3, stride=1)
        # self.conv8_2 = myconv(64, 8, kernel_size=3, stride=1)
        # self.conv10_0 = myconv(od, 128, kernel_size=3, stride=1)
        # self.conv10_1 = myconv(128, 64, kernel_size=3, stride=1)
        # self.conv10_2 = myconv(64, 4, kernel_size=3, stride=1)


        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= (image_size[0] // downsample)+1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= (image_size[1] // downsample)+1
        # self.fc1 = nn.Linear(10368 , 512)
        # self.fc2 = nn.Linear(4*10368 , 512)
        # self.fc3 = nn.Linear(16*10368 , 512)
        # self.fc4 = nn.Linear(64*10368 , 512)

        self.fc1_trasl = nn.Linear(123120, 256)
        self.fc1_rot = nn.Linear(123120, 256)
        self.fc2_trasl = nn.Linear(256, 32)
        self.fc2_rot = nn.Linear(256, 32)
        # self.fc3_trasl = nn.Linear(128, 64)
        # self.fc3_rot = nn.Linear(128, 64)
        # self.fc4_trasl = nn.Linear(64, 32)
        # self.fc4_rot = nn.Linear(64, 32)
        self.fc5_trasl = nn.Linear(32, 3)
        self.fc5_rot = nn.Linear(32, 4)
        self.fc115_trasl = nn.Linear(8192, 128)
        self.fc114_trasl = nn.Linear(32768, 64)
        self.fc113_trasl = nn.Linear(131072, 32)
        self.fc112_trasl = nn.Linear(524288, 16)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x, size=(H,W), mode='bilinear') + y


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
        xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).cuda()
        mask = F.grid_sample(mask, vgrid)


        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask


    def forward(self, rgb, lidar):
        #encoder
        if self.pretrained_encoder:
            # rgb_image
            features1 = self.net_encoder(rgb)
            c12 = features1[0]  # 2
            c13 = features1[2]  # 4
            c14 = features1[3]  # 8
            c15 = features1[4]  # 16
            c16 = features1[5]  # 32
            # lidar_image
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c22 = self.elu_lidar(x2)  # 2
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
            c24 = self.layer2_lidar(c23)  # 8
            c25 = self.layer3_lidar(c24)  # 16
            c26 = self.layer4_lidar(c25)  # 32

        else: # here c starts with 1 means cam feat and 2 means lidar feat;
            x1 = self.conv1_rgb(rgb)
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c12 = self.leakyRELU_rgb(x1)  # 2
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c12 = self.elu_rgb(x1)  # 2
                c22 = self.elu_lidar(x2)  # 2
            c13 = self.layer1_rgb(c12)  # 4
            c23 = self.layer1_lidar(c22)  # B*64*128*256
            c14 = self.layer2_rgb(c13)  # 8
            c24 = self.layer2_lidar(c23)  # B*128*64*128
            c15 = self.layer3_rgb(c14)  # 16
            c25 = self.layer3_lidar(c24)  # B*256*32*64
            c16 = self.layer4_rgb(c15)  # 32
            c26 = self.layer4_lidar(c25)  # #B*512*16*32

        # Full FPN fusion process

        # Lat-layer process for lidar feat, and go through up-sample & fuse process
        p5 = self.toplayer(c26) #B*64*16*32
        p4 = self._upsample_add(p5, self.latlayer1(c25)) #B*64*32*64
        p3 = self._upsample_add(p4, self.latlayer2(c24)) #B*64*64*128
        p2 = self._upsample_add(p3, self.latlayer3(c23)) #B*64*128*256

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Lat-layer process for cam feat, and go through up-sample & fuse process
        p15 = self.toplayer(c16)
        p14 = self._upsample_add(p15, self.latlayer1(c15))
        p13 = self._upsample_add(p14, self.latlayer2(c14))
        p12 = self._upsample_add(p13, self.latlayer3(c13))

        p14 = self.smooth1(p14)
        p13 = self.smooth2(p13)
        p12 = self.smooth3(p12)

        # Cross-Modal attention module

        corr5 = self.attcorr5(p15,p5) #B*64*8*16
        x115 = self.leakyRELU(corr5)
        x115 = x115.reshape(x115.shape[0], -1)
        x115 = self.dropout(x115)
        x115 = self.leakyRELU(x115)
        x115 = self.leakyRELU(self.fc115_trasl(x115))  #B*128 this is one of the outputs

        corr4 = self.attcorr4(p14, p4)
        x114 = self.leakyRELU(corr4)
        x114 = x114.reshape(x114.shape[0], -1)
        x114 = self.dropout(x114)
        x114 = self.leakyRELU(x114)
        x114 = self.leakyRELU(self.fc114_trasl(x114))  #B*64 this is one of the outputs

        corr3 = self.attcorr3(p13, p3)#b*64*32*64
        x113 = self.leakyRELU(corr3)
        x113 = x113.reshape(x113.shape[0], -1)
        x113 = self.dropout(x113)
        x113 = self.leakyRELU(x113)
        x113 = self.leakyRELU(self.fc113_trasl(x113))  #B*32 this is one of the outputs

        corr2 = self.attcorr2(p12, p2)
        x112 = self.leakyRELU(corr2)
        x112 = x112.reshape(x112.shape[0], -1)
        x112 = self.dropout(x112)
        x112 = self.leakyRELU(x112)
        x112 = self.leakyRELU(self.fc112_trasl(x112))  #B*16 this is one of the outputs

        corr6 = self.attcorr6(c16, c26) #B*512*8*16
        x116 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1) #B，512+128，H,W
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)  #B*960*8*16


        x = x.reshape(x.shape[0], -1) #B*122880
        x = self.dropout(x)
        x = self.leakyRELU(x) # this is one of the outputs


        # fuse outputs from different scales

        x = torch.cat((x112, x), 1)
        x = torch.cat((x113, x), 1)
        x = torch.cat((x114, x), 1)
        x = torch.cat((x115, x), 1)

        # Regression for t & R

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.leakyRELU(self.fc2_trasl(transl))
        rot = self.leakyRELU(self.fc2_rot(rot))


        transl = self.fc5_trasl(transl)
        rot = self.fc5_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out

