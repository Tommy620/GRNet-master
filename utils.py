# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py
# Modified Author: Hao Wang
# based on https://github.com/LvXudong-HIT/LCCNet/utils.py
# Modified Author: Zi Wang
# based on https://github.com/AlexWang0214/MRCNet/blob/master/utils.py



import math  

import mathutils   
import numpy as np  
import torch.nn as nn
import torch  
import torch.nn.functional as F  
from matplotlib import cm  
from torch.utils.data.dataloader import default_collate  


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix() 
        R.resize_4x4() 
        T = mathutils.Matrix.Translation(T) 
        RT = T*R 
    else:  
        RT=R.copy() 
    if inverse:
        RT.invert_safe()  
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)  

    if PC.shape[0] == 4:  
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())  
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC  


def rotate_points_torch(PC, R, T=None, inverse=True): 
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None): 
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None): 
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T): 

    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries): 
    point_clouds = [] 
    imgs = []
    reflectances = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'} 
    for input in queries: 
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds 
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns 


def quaternion_from_matrix(matrix): 
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def quatmultiply(q, r): 
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q): 
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t): 
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix): 
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T): 
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0): 
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
     
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

     
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
     
     
     
     
     
    return blended_img


def point_cloud_washing(target_transl, target_rot, point_clouds,cam_calib):
    for i in range(len(point_clouds)):  
        X1 = cam_calib[i].numpy() 

        X1=np.insert(X1, 3, np.array([0, 0, 0]), axis=0) 
        X1=np.insert(X1, 3, np.array([0, 0, 0,1]), axis=1) 
        X1=torch.Tensor(X1)
        X1 = X1.cuda() 

        R = mathutils.Quaternion(target_rot[i]).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(target_transl[i])
        RT = T * R
        RTcuda=torch.tensor(RT).cuda() 
        depth_img=torch.mm(X1,RTcuda) 
        depth_img=torch.mm(depth_img,point_clouds[i].cuda())  
        depth_img = depth_img.cpu().detach().numpy()
        depth_img=np.delete(depth_img, np.where((depth_img[0]/depth_img[2])<0)[0], axis=1)
        depth_img=np.delete(depth_img, np.where((depth_img[0]/depth_img[2])>1241)[0], axis=1) 
        depth_img=np.delete(depth_img, np.where((depth_img[1]/depth_img[2])>376)[0], axis=1) 
        depth_img=np.delete(depth_img, np.where((depth_img[1]/depth_img[2])<0)[0], axis=1) 
        depth_img=torch.from_numpy(depth_img).cuda()
        depth_img=torch.mm(X1.inverse(),depth_img)
        depth_img=torch.mm(RTcuda.inverse(),depth_img) 

        point_clouds[i]=depth_img 

    return point_clouds


def mixed_Q_loss(point_clouds, target_transl, target_rot, transl_err, rot_err, cam_calib, mix_q_loss, bs):
    for i in range(len(point_clouds)):   
        X = cam_calib[i].numpy()
        X1 = np.insert(X, 3, np.array([0, 0, 0]), axis=0)
        X1 = np.insert(X1, 3, np.array([0, 0, 0, 1]), axis=1)
        X1 = torch.Tensor(X1)
        X1 = X1.cuda()   

        point_cloud_gt = point_clouds[i].to(transl_err.device)
        R_target = quat2mat(target_rot[i])
        T_target = tvector2mat(target_transl[i])
        RT_target = torch.mm(T_target, R_target)   
        RTCA = torch.mm(X1, RT_target)
        RTCB = torch.mm(RTCA, point_cloud_gt)   
        temptensor = RTCB[2:3]
        RTCC = RTCB / temptensor   
        RTCC.F = RTCC[0]   
        RTCC.S = RTCC[1]   

        #optic-loss-factor
        Quanzhong1 = RTCC.F - cam_calib[i][0][2]
        Quanzhong2 = RTCC.S - cam_calib[i][1][2]   
        Quanzhong = torch.cat(([Quanzhong1, Quanzhong2]), dim=0)   
        Quanzhong = torch.reshape(Quanzhong, (2, -1))   
        Quanzhong = Quanzhong.norm(dim=0)   
        Quanzhong = torch.abs(Quanzhong)
        total_distance = Quanzhong.sum()
        Quanzhong = Quanzhong / total_distance  
         
        #depth-loss-factor
        Quanzhong_dis = temptensor.squeeze(0)   
        Quanzhong_dis = torch.abs(Quanzhong_dis)   
        Quanzhong_dis = 1 / (Quanzhong_dis + 0.0000001)   
        Quanzhong_dis = F.normalize(Quanzhong_dis.float(), p=1, dim=0, eps=5)

        #mixed-loss weighting pixels
        Real_QZ = (Quanzhong + Quanzhong_dis) * 0.5   

        R_predicted = quat2mat(rot_err[i])
        T_predicted = tvector2mat(transl_err[i])
        RT_predicted = torch.mm(T_predicted, R_predicted)
        RTCA1 = torch.mm(X1, RT_predicted)
        RTCB1 = torch.mm(RTCA1, point_cloud_gt)
        temptensor1 = RTCB1[2:3]
        RTCC1 = RTCB1 / temptensor1   
        RTCC1.F = RTCC1[0]
        RTCC1.S = RTCC1[1]
        RTCC.F = torch.where(RTCC.F < 1280, RTCC.F, RTCC1.F)   
        RTCC.F = torch.where(RTCC.F > 0, RTCC.F, RTCC1.F)
        RTCC1.F = torch.where(RTCC1.F > 0, RTCC1.F, RTCC.F)   
        RTCC1.F = torch.where(RTCC1.F < 1280, RTCC1.F, RTCC.F)
        RTCC.S = torch.where(RTCC.S < 720, RTCC.S, RTCC1.S)   
        RTCC.S = torch.where(RTCC.S > 0, RTCC.S, RTCC1.S)
        RTCC1.S = torch.where(RTCC1.S > 0, RTCC1.S, RTCC.S)   
        RTCC1.S = torch.where(RTCC1.S < 720, RTCC1.S, RTCC.S)
        RTCCF = torch.cat((RTCC.F, RTCC.S), dim=0)
        RTCC1F = torch.cat((RTCC1.F, RTCC1.S), dim=0)
        RTCCF = torch.reshape(RTCCF, (2, -1))   
        RTCC1F = torch.reshape(RTCC1F, (2, -1))   
        error = (RTCCF - RTCC1F).norm(dim=0)   
        error = torch.mul(error, Real_QZ)   
        mix_q_loss += error.mean()   
        bs = bs + 1   
    return mix_q_loss, bs
