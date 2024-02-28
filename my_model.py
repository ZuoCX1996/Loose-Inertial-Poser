import numpy as np
import torch

from Aplus.models import *
from my_trainner import *
from articulate.math import rotation_matrix_to_r6d, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
from articulate.math.general import normalize_tensor
from config import joint_set
import torch.nn.functional as F

def linear_interpolation_batch(vector1, vector2, target_length):
    if vector1.size(-1) != vector2.size(-1):
        raise ValueError("向量维度不匹配")

    # 计算插值步长
    interpolation_steps = target_length - 1
    step_size = (vector2 - vector1) / interpolation_steps

    # 初始化插值结果
    interpolated_data = [vector1]

    # 执行线性插值
    for i in range(1, interpolation_steps):
        interpolated_point = vector1 + i * step_size
        interpolated_data.append(interpolated_point)

    interpolated_data.append(vector2)

    # 将结果转换为批量形式
    interpolated_data = torch.stack(interpolated_data, dim=1)

    return interpolated_data

def rotation_matrix_to_axis_angle_torch(R):
    # 使用 torch 的 acos 函数计算旋转角度（批量）
    diag = torch.cat([R[:, 0, [0]], R[:, 1, [1]], R[:, 2, [2]]], dim=-1)
    cos_theta = 0.5 * (diag.sum(-1) - 1)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

    # 计算旋转轴（批量）
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)

    # 归一化旋转轴向量（批量）
    axis = F.normalize(axis, p=2, dim=-1)
    return axis * theta.unsqueeze(-1)

class DiscrepancyNomalizer(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.mmd = MMDLoss()
        self.BN = nn.BatchNorm1d(num_features=n_feature)
    def forward(self, x1, x2):
        dis = x2 - x1
        mean = torch.mean(dis, dim=0)
        dis = self.BN(dis)
        norm_noise = torch.randn(size=dis.shape).float().to(dis.device)
        return self.mmd(dis, norm_noise) + torch.pow(mean, 2).mean()
    # def forward(self, x1, x2):
    #     dis = x2 - x1
    #     dis = self.BN(dis)
    #     norm_noise = torch.randn(size=dis.shape).float().to(dis.device)
    #     return self.mmd(dis, norm_noise)

    def semo_recon(self, x1, eta=1, mask=None):
        # x1 shape: [batch, seq_len, dim]
        if mask is not None:
            std = torch.sqrt(self.BN.running_var.data).float().squeeze(0).to(x1.device)
            mask = mask.to(x1.device)
            std = std*mask
            return x1 + std
        norm_noise = torch.randn(size=x1.shape).float().to(x1.device)
        mean = self.BN.running_mean.data.float().squeeze(0).to(x1.device)
        std = torch.sqrt(self.BN.running_var.data).float().squeeze(0).to(x1.device)
        recon_semo = norm_noise * std *0.5 + mean

        # TCS
        bias_shift_1 = torch.randn(size=x1[:, 0, :].shape).float().to(x1.device)
        bias_shift_2 = torch.randn(size=x1[:, 0, :].shape).float().to(x1.device)
        bias_shift = linear_interpolation_batch(bias_shift_1, bias_shift_2, target_length=x1.shape[1])

        recon_semo = recon_semo + bias_shift * std * eta
        return x1 + recon_semo

    def get_std(self):
        return torch.sqrt(self.BN.running_var.data.float().squeeze(0))

    def get_mean(self):
        return self.BN.running_mean.data.float().squeeze(0)

class BiPoser(BaseModel):
    def __init__(self, net_s1, net_s2, export_mode=False):
        super(BiPoser, self).__init__()

        self.net_s1 = net_s1
        self.net_s2 = net_s2
        self.export_mode = export_mode

    def forward_train(self, x, *args):

        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2 = args
            out_1, h_s1, c_s1 = self.net_s1(x, h_s1, c_s1)
            out_2, h_s2, c_s2 = self.net_s2(torch.cat([x, out_1], dim=-1), h_s2, c_s2)
        else:
            out_1 = self.net_s1(x)
            out_2 = self.net_s2(torch.cat([x, out_1], dim=-1))

        if len(args) > 0:
            return out_1, out_2, h_s1, c_s1, h_s2, c_s2
        return out_1, out_2

    def joint_out(self, x):
        out_1 = self.net_s1(x)
        return out_1

    def pose_out(self, x, *args):
        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2 = args
            out_1, h_s1, c_s1 = self.net_s1(x, h_s1, c_s1)
            out_2, h_s2, c_s2 = self.net_s2(torch.cat([x, out_1], dim=-1), h_s2, c_s2)
        else:
            out_1 = self.net_s1(x)
            out_2 = self.net_s2(torch.cat([x, out_1], dim=-1))

        if len(args) > 0:
            return out_2, h_s1, c_s1, h_s2, c_s2
        return out_2

    def forward(self, acc_cat_rot, *args):
        x = acc_cat_rot
        if self.export_mode:
            return self.pose_out(x, *args)
        return self.forward_train(x, *args)
class Poser(BaseModel):
    def __init__(self, net, type='r6d', use_elbow_angle=False):
        super(Poser, self).__init__()

        self.data_dim = 6
        self.net = net
        self.type = type
        self.use_elbow_angle = use_elbow_angle
        if type == 'axis_angle':
            self.data_dim = 3

    # r6d
    @torch.no_grad()
    def forward(self, x, *args):

        # 48 = [4x3+4x9]

        # # 输入为旋转矩阵时 将r6d变换融合进模型
        # if self.use_elbow_angle:
        #     acc, rot, elbow_angle = x[:, :12], x[:, 12:48], x[:, 48:]
        #     rot_r6d = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 24)
        #     x = torch.cat([acc, rot_r6d, elbow_angle], dim=-1)
        # else:
        #     acc, rot = x[:, :12], x[:, 12:48]
        #     rot_r6d = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 24)
        #     x = torch.cat([acc, rot_r6d], dim=-1)

        x = x.unsqueeze(0)

        if len(args) > 0:
            output1, h_1, c_1, h_2, c_2 = self.net(x, *args)
        else:
            output1 = self.net(x)

        if self.type == 'axis_angle':
            pose = output1.reshape(-1, 6)
            pose = r6d_to_rotation_matrix(pose)
            # print(pose)
            output1 = rotation_matrix_to_axis_angle_torch(pose).unsqueeze(0)
            # print(output1)
            # print(axis_angle_to_rotation_matrix(output1))

        pose = output1.view(-1, len(joint_set.index_pose), self.data_dim)

        # pose_output = torch.zeros(len(pose), 24, 3).to(device)

        pose_output = pose[:, 0:1, :].repeat(1, 24, 1) * 0

        pose_output[:, joint_set.index_pose, :] += pose

        if len(args) > 0:
            return pose_output, h_1, c_1, h_2, c_2
        else:
            return pose_output

class SemoAE(BaseModel):
    def __init__(self, feat_dim, encode_dim):
        super(SemoAE, self).__init__()
        # act_fun = 'leakyrelu'
        act_fun = 'tanh'

        self.encoder = EasyDNN(n_input=feat_dim, n_hiddens=[128, 64], n_output=encode_dim, act_func=act_fun)
        self.decoder = EasyDNN(n_input=encode_dim, n_hiddens=[64, 128], n_output=feat_dim, act_func=act_fun)
        self.dis_normalizer = DiscrepancyNomalizer(n_feature=encode_dim)

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x, norm=False):
        x = self.decoder(x)
        if norm:
            acc, rot = x[:, :, :36][:, :, :-24], x[:, :, -24:]
            rot = self._r6d_norm(rot)
            x = torch.cat([acc, rot], dim=-1)
        return x

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def _r6d_norm(self, x, rot_num=4):
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], rot_num, 6)
        result = []
        for i in range(rot_num):
            column0 = normalize_tensor(x[:, :, i, 0:3])
            column1 = normalize_tensor(x[:, :, i, 3:6] - (column0 * x[:, :, i, 3:6]).sum(dim=-1, keepdim=True) * column0)
            result.append(torch.cat([column0, column1], dim=-1))
        x = torch.cat(result, dim=-1)
        return x

    @torch.no_grad()
    def secondary_motion_gen(self, x, eta=1, acc_gen=True, mask=None):
        if acc_gen:
            x = x.detach()
            x = self.encoder(x)
            x = self.dis_normalizer.semo_recon(x1=x, eta=eta)
            x = self.decoder(x)
            acc, rot = x[:, :, :36][:, :, :-24], x[:, :, -24:]
            rot = self._r6d_norm(rot)

        else:
            acc = x[:, :, :36][:, :, :-24]
            x = x.detach()
            x = self.encoder(x)
            x = self.dis_normalizer.semo_recon(x1=x, eta=eta, mask=mask)
            x = self.decoder(x)
            rot = x[:, :, -24:]
            rot = self._r6d_norm(rot)

        x = torch.cat([acc, rot], dim=-1)
        return x
