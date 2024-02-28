import numpy as np
import torch

from Aplus.data import *
from Aplus.data.process import add_gaussian_noise
from config import paths, joint_set
import os
from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d, axis_angle_to_quaternion, euler_angle_to_rotation_matrix, rotation_matrix_to_euler_angle, quaternion_to_rotation_matrix
from tqdm import tqdm
from Aplus.tools.annotations import timing
import quaternion
from clothes_imu_syn import *

def amass_read_seg(path, min_len=256):
    data = torch.load(path)
    selected_data = []
    seg_info = []
    for slice in data:
        # print(slice.shape)
        if len(slice) < min_len:
            continue
        else:
            selected_data.append(slice)
            seg_info.append(len(slice))
    data = torch.cat(selected_data, dim=0)
    # print(acc_t.shape)
    # index_info = seg_info_2_index_info(seg_info)
    # print(index_info)
    # find_seg_index(index_info, 2200)
    return data, seg_info

def seg_info_2_index_info(seg_info):
    index_info = [0]
    for v in seg_info:
        index_info.append(index_info[-1] + v)
    return index_info

def find_seg_index(index_info, data_index, n_seg=0):
    # index_info = np.array(index_info)
    # mask = np.array(index_info.__le__(data_index), dtype='int')
    # seq_index = int(mask.sum()) - 1

    seq_index = -1
    # n_seg = 0
    if n_seg != 0:
        seq_index = n_seg - 1

    # 0 1 2 3 4
    # 0 2 4 6 8
    # print(n_seg)
    # print(_index_info)
    # print(len(index_info))
    for v in index_info[n_seg:]:
        if v <= data_index:
            seq_index += 1
        else:
            break

    # index_info = np.array(index_info)
    # mask = np.array(index_info.__le__(data_index), dtype='int')
    # seq_index = int(mask.sum()) - 1
    # print(seq_index)
    inner_index = data_index - index_info[seq_index]
    return seq_index, inner_index

def bulid_rot(theta, rotation_axis):
    w = np.cos(theta * np.pi / 360)
    s = np.sin(theta * np.pi / 360)
    x = s * rotation_axis[0]
    y = s * rotation_axis[1]
    z = s * rotation_axis[2]

    q = quaternion.from_float_array([w, x, y, z])
    q = torch.Tensor([q.w, q.x, q.y, q.z]).float()
    rot = quaternion_to_rotation_matrix(q)

    return rot
def amass_read(path):
    data = torch.load(path)
    data = torch.cat(data, dim=0)
    # print(acc_t.shape)
    return data

class AmassData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step]
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len],
                self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d') -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """
        all_joint_num = len(joint_set.index_pose)

        rot = amass_read(os.path.join(folder_path, 'vrot.pt'))
        acc = amass_read(os.path.join(folder_path, 'vacc.pt'))
        # acc = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))
        pose = amass_read(os.path.join(folder_path, 'pose.pt'))

        # pose转为r6d
        rot_dim = 3
        if pose_type == 'r6d':
            # 数据分2段处内存占用
            data_len = len(pose)
            # pose = pose.view(data_len * 24, 3)
            len_pose_1 = data_len // 2
            len_pose_2 = data_len - len_pose_1

            pose_seg_1 = pose[:len_pose_1].view(len_pose_1 * 24, 3)
            pose_seg_2 = pose[len_pose_1:].view(len_pose_2 * 24, 3)

            pose_1 = axis_angle_to_rotation_matrix(pose_seg_1)
            pose_1 = rotation_matrix_to_r6d(pose_1).reshape(len_pose_1, 24, 6)

            pose_2 = axis_angle_to_rotation_matrix(pose_seg_2)
            pose_2 = rotation_matrix_to_r6d(pose_2).reshape(len_pose_2, 24, 6)

            pose = torch.cat([pose_1, pose_2], dim=0)

            rot_dim = 6

        # 限制范围 防止异常值干扰
        acc = torch.clamp(acc, min=-60, max=60)
        # acc normalization 非必须 可以只保留除30
        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30

        # 转换为相对根节点的旋转 非必须 可以不做
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)

        # 关节点空间坐标
        joint = amass_read(os.path.join(folder_path, 'joint.pt'))
        # 归一化
        joint = joint - joint[:, :1, :]
        # 转换到root节点下 非必须 可以不做
        joint = joint.bmm(rot[:, -1])

        # rot转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 4, 6)

        x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt
        pose_upper_body = pose[:, joint_set.index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        # 加上手部2个节点 扣除根节点
        joint_upper_body = joint[:, joint_set.index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)

        return {'x_s1': x_s1,
                # 'x_s2': x_s2,
                # 'pose_internal': pose_internal,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}


class DipData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step]
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len], self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d') -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        all_joint_num = len(joint_set.index_pose)

        rot = torch.load(os.path.join(folder_path, 'vrot.pt'))
        acc = torch.load(os.path.join(folder_path, 'vacc.pt'))

        # joint = torch.load(os.path.join(folder_path, 'joint.pt')).reshape(-1, 24, 3)
        pose = torch.load(os.path.join(folder_path, 'pose.pt'))

        # pose转为r6d

        if pose_type == 'r6d':
            data_len = len(pose)
            pose = pose.view(data_len * 24, 3)
            pose = axis_angle_to_rotation_matrix(pose)
            pose = rotation_matrix_to_r6d(pose).reshape(data_len, 24, 6)
            rot_dim = 6
        else:
            rot_dim = 3
            pose = pose.reshape(-1, 24, 3)

        # 限制范围 防止异常值干扰
        acc = torch.clamp(acc, min=-60, max=60)
        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30
        # 不进行相对加速度处理 从SMPL坐标系转到root坐标系
        # acc = acc.bmm(rot[:, -1]) / 30

        # 转换为相对根节点的旋转
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)

        joint = torch.load(os.path.join(folder_path, 'joint.pt')).reshape(-1, 24, 3)
        # 归一化
        joint = joint - joint[:, :1, :]
        # 转到root坐标系
        joint = joint.bmm(rot[:, -1])

        # rot转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 4, 6)


        x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt

        # x_s2 = torch.cat((x_s1.flatten(1), pose_internal.flatten(1)), dim=1)  # s2的输入
        # pose_external = pose[:, joint_set.external_joint].reshape(len(pose), external_joint_num * 6)  # s2输出的gt

        pose_upper_body = pose[:, joint_set.index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        joint_upper_body = joint[:, joint_set.index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)
        return {'x_s1': x_s1,
                # 'x_s2': x_s2,
                # 'pose_internal': pose_internal,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}

class TailorNetSynData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step]
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len], self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d') -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        all_joint_num = len(joint_set.index_pose)

        rot = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
        acc = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))

        # joint = torch.load(os.path.join(folder_path, 'joint.pt')).reshape(-1, 24, 3)
        pose = amass_read(os.path.join(folder_path, 'pose.pt'))

        # pose转为r6d

        if pose_type == 'r6d':
            # 数据分2段处内存占用
            data_len = len(pose)
            # pose = pose.view(data_len * 24, 3)
            len_pose_1 = data_len // 2
            len_pose_2 = data_len - len_pose_1

            pose_seg_1 = pose[:len_pose_1].view(len_pose_1 * 24, 3)
            pose_seg_2 = pose[len_pose_1:].view(len_pose_2 * 24, 3)

            pose_1 = axis_angle_to_rotation_matrix(pose_seg_1)
            pose_1 = rotation_matrix_to_r6d(pose_1).reshape(len_pose_1, 24, 6)

            pose_2 = axis_angle_to_rotation_matrix(pose_seg_2)
            pose_2 = rotation_matrix_to_r6d(pose_2).reshape(len_pose_2, 24, 6)

            pose = torch.cat([pose_1, pose_2], dim=0)

            rot_dim = 6
        else:
            rot_dim = 3
            pose = pose.reshape(-1, 24, 3)

        # 限制范围 防止异常值干扰
        acc = torch.clamp(acc, min=-60, max=60)
        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30
        # 不进行相对加速度处理 从SMPL坐标系转到root坐标系
        # acc = acc.bmm(rot[:, -1]) / 30

        # 转换为相对根节点的旋转
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)

        joint = amass_read(os.path.join(folder_path, 'joint.pt')).reshape(-1, 24, 3)
        # 归一化
        joint = joint - joint[:, :1, :]
        # 转到root坐标系
        joint = joint.bmm(rot[:, -1])

        # rot转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 4, 6)

        x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt

        # x_s2 = torch.cat((x_s1.flatten(1), pose_internal.flatten(1)), dim=1)  # s2的输入
        # pose_external = pose[:, joint_set.external_joint].reshape(len(pose), external_joint_num * 6)  # s2输出的gt

        pose_upper_body = pose[:, joint_set.index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        joint_upper_body = joint[:, joint_set.index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)
        return {'x_s1': x_s1,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}
class LipData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=2):
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step]
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len], self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path='E:\DATA\LIP-IMU', use_elbow_angle=False, pose_type='r6d', acc_scale=1, type='all', back_fix=False) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        all_joint_num = len(joint_set.index_pose)

        rot, acc, joint, pose = [], [], [], []

        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if type != 'all':
                    if dir_name.find(type) >= 0:
                        dir_path = os.path.join(root, dir_name)
                        print(f'loading {dir_name}')
                        rot.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                        acc.append(torch.load(os.path.join(dir_path, 'acc.pt')))
                        joint.append(torch.load(os.path.join(dir_path, 'joint.pt')).reshape(-1, 24, 3))
                        pose.append(torch.load(os.path.join(dir_path, 'pose.pt')))
                else:
                    dir_path = os.path.join(root, dir_name)
                    print(f'loading {dir_name}')
                    rot.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                    acc.append(torch.load(os.path.join(dir_path, 'acc.pt')))
                    joint.append(torch.load(os.path.join(dir_path, 'joint.pt')).reshape(-1, 24, 3))
                    pose.append(torch.load(os.path.join(dir_path, 'pose.pt')))

        rot = torch.cat(rot, dim=0)
        acc = torch.cat(acc, dim=0)
        joint = torch.cat(joint, dim=0)
        pose = torch.cat(pose, dim=0)

        if back_fix:
            rot_left, rot_right, rot_back, rot_root = rot[:, [0]], rot[:, [1]], rot[:, [2]], rot[:, [3]]
            rot_back = bulid_rot(theta=180, rotation_axis=[0, 1, 0]).matmul(rot_back)
            rot = torch.cat([rot_left, rot_right, rot_back, rot_root], dim=1)

        print(rot.shape)
        # print(rot.shape)
        print(acc.shape)
        print(joint.shape)
        print(pose.shape)

        # pose转为r6d

        if pose_type == 'r6d':
            data_len = len(pose)
            pose = pose.view(data_len * 24, 3)
            pose = axis_angle_to_rotation_matrix(pose)
            pose = rotation_matrix_to_r6d(pose).reshape(data_len, 24, 6)
            rot_dim = 6
        else:
            rot_dim = 3
            pose = pose.reshape(-1, 24, 3)

        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30
        # 转换为相对根节点的旋转
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)
        # 归一化
        joint = joint - joint[:, :1, :]
        # 转到root坐标系
        joint = joint.bmm(rot[:, -1])
        # 转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 9)).reshape(-1, 4, 6)

        x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt

        # x_s2 = torch.cat((x_s1.flatten(1), pose_internal.flatten(1)), dim=1)  # s2的输入
        # pose_external = pose[:, joint_set.external_joint].reshape(len(pose), external_joint_num * 6)  # s2输出的gt

        pose_upper_body = pose[:, joint_set.index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        joint_upper_body = joint[:, joint_set.index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)
        return {'x_s1': x_s1,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}

class SynPairedIMUData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, shuffle=True):
        self.x = x
        self.y = y
        self.data_len = len(x)
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        return self.x[self.indexer[i]], self.y[self.indexer[i]]

    @staticmethod
    @timing
    def load_data(folder_path: str, shuffle=False, normalization=True, clothes_imu_calibration=False) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        rot_bone = amass_read(os.path.join(folder_path, 'vrot.pt'))
        rot_imu = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
        acc_mesh = amass_read(os.path.join(folder_path, 'vacc.pt'))
        acc_imu = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))

        tpose_clothes_v = obj_load_vertices(path='./T-Pose_garment.obj')
        tpose_rot, _ = imu_syn(tpose_clothes_v)
        device2bone = tpose_rot.transpose(-2,-1)

        if clothes_imu_calibration:
            rot_imu =rot_imu.matmul(device2bone)
        # print(device2bone)



        # rot_bone = torch.load(os.path.join(folder_path, 'vrot.pt'))
        # rot_imu = torch.load(os.path.join(folder_path, 'rot.pt'))
        # acc_mesh = torch.load(os.path.join(folder_path, 'vacc.pt'))
        # acc_imu = torch.load(os.path.join(folder_path, 'acc.pt'))

        data_len = len(rot_bone)

        # 防止异常值
        acc_mesh = torch.clamp(acc_mesh, min=-60, max=60)
        acc_imu = torch.clamp(acc_imu, min=-60, max=60)

        if normalization:
            # 转换为相对根节点加速度
            acc_mesh = torch.cat((acc_mesh[:, :3] - acc_mesh[:, 3:], acc_mesh[:, 3:]), dim=1).bmm(rot_bone[:, -1]) / 30
            acc_imu = torch.cat((acc_imu[:, :3] - acc_imu[:, 3:], acc_imu[:, 3:]), dim=1).bmm(rot_imu[:, -1]) / 30

        acc_mesh = acc_mesh.reshape(data_len, -1)
        acc_imu = acc_imu.reshape(data_len, -1)

        if normalization:
            # 转换为相对根节点的旋转
            rot_bone = torch.cat((rot_bone[:, 3:].transpose(2, 3).matmul(rot_bone[:, :3]), rot_bone[:, 3:]), dim=1)
            rot_imu = torch.cat((rot_imu[:, 3:].transpose(2, 3).matmul(rot_imu[:, :3]), rot_imu[:, 3:]), dim=1)

        # rot转为r6d
        rot_bone = rot_bone.view(data_len * 4, 3, 3)
        rot_imu = rot_imu.view(data_len * 4, 3, 3)

        rot_bone = rotation_matrix_to_r6d(rot_bone).reshape(data_len, -1)
        rot_imu = rotation_matrix_to_r6d(rot_imu).reshape(data_len, -1)

        data_mesh = torch.cat([acc_mesh, rot_bone], dim=-1)
        data_garment = torch.cat([acc_imu, rot_imu], dim=-1)

        if shuffle:
            new_idx = random_index(data_len=len(data_mesh), seed=42)
            data_mesh = data_mesh[new_idx]
            data_garment = data_garment[new_idx]

        return {'data_mesh': data_mesh,
                'data_garment': data_garment}
