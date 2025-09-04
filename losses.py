# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Hao Wang
# based on https://github.com/LvXudong-HIT/LCCNet/losses.py
# Modified Author: Zi Wang
# based on https://github.com/AlexWang0214/MRCNet/blob/master/losses.py
import torch
from torch import nn as nn
import numpy as np
from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix,mixed_Q_loss
import torch.nn.functional as F


class Multi_dim_Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(Multi_dim_Loss, self).__init__()
        self.rescale_trans = rescale_trans #此处为1
        self.rescale_rot = rescale_rot #此处为1
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}


    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err,cam_calib):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()

        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_rot * loss_rot + self.rescale_trans * loss_transl

         
        mix_q_loss = torch.tensor([0.0]).to(transl_err.device)
        bs=0 #初始化

        mix_q_loss,bs = mixed_Q_loss(point_clouds, target_transl, target_rot, transl_err, rot_err, cam_calib, mix_q_loss, bs)
        #The function for mixed loss can be found here, click mixed_Q_loss and check~

        if(bs!=0):
            total_loss =  (1 - self.weight_point_cloud) * pose_loss + self.weight_point_cloud * (mix_q_loss/bs)

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        return self.loss
