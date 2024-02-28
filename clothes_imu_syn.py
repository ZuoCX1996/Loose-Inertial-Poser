import torch
from articulate.math.angular import normalize_tensor
import trimesh

class garment_imu_set:
    # 起点/终点
    imu_axis = {'left' :{'axis_1':[916, 4795],  'axis_2':[4796, 4828], 'order':'zx'},
                'right':{'axis_1':[8504, 2109], 'axis_2':[8529, 8369], 'order':'zx'},
                'back' :{'axis_1':[7916, 6021], 'axis_2':[6022, 1226], 'order':'xy'},
                'root' :{'axis_1':[7972, 6000], 'axis_2':[1220, 5999], 'order':'xy'}}

def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    Turn 6D vectors into rotation matrices. (torch, batch)

    **Warning:** The two 3D vectors of any 6D vector must be linearly independent.

    :param r6d: 6D vector tensor that can reshape to [batch_size, 6].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    r6d = r6d.reshape(-1, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r

def two_axis_vec_2_rotation_matrix(axis_1:torch.Tensor, axis_2:torch.Tensor, order='xy'):
    """

    :param axis_1: [batch , 3]
    :param axis_2: [batch , 3]
    :param order: 'xy'|'yz'|'zx'
    :return:
    """
    column0 = normalize_tensor(axis_1, dim=-1)

    column1 = normalize_tensor(axis_2 - (column0 * axis_2).sum(dim=1, keepdim=True) * column0, dim=-1)
    # print(column1)
    column2 = column0.cross(column1, dim=-1)
    # column2 = normalize_tensor(column2, dim=-1)

    if order=='xy':
        r = torch.stack((column0, column1, column2), dim=-1)
    elif order=='yz':
        r = torch.stack((column2, column0, column1), dim=-1)
    elif order=='zx':
        r = torch.stack((column1, column2, column0), dim=-1)
    else:
        print(f"unexpected order: {order}, try 'xy'|'yz'|'zx'")

    r[torch.isnan(r)] = 0
    return r


def imu_syn(garment_vertices):
    imu_rot = []
    imu_position = []
    for key, value in garment_imu_set.imu_axis.items():
        # 通过顶点坐标计算坐标轴朝向
        axis_1 = torch.tensor(garment_vertices[value['axis_1'][1]] - garment_vertices[value['axis_1'][0]]).float()
        axis_2 = torch.tensor(garment_vertices[value['axis_2'][1]] - garment_vertices[value['axis_2'][0]]).float()

        # print(axis_1, axis_2)

        rot = two_axis_vec_2_rotation_matrix(axis_1=axis_1.unsqueeze(0), axis_2=axis_2.unsqueeze(0),
                                             order=value['order'])
        imu_rot.append(rot)

        position = (garment_vertices[value['axis_1'][0]] + garment_vertices[value['axis_1'][1]] + \
                    garment_vertices[value['axis_2'][0]] + garment_vertices[value['axis_2'][1]]) / 4
        position = torch.tensor(position).unsqueeze(0)
        imu_position.append(position)
    return torch.cat(imu_rot, dim=0).unsqueeze(0), torch.cat(imu_position, dim=0).unsqueeze(0)

# posed_garment = trimesh.load('./T-Pose_garment.obj', process=False)
# imu_axis = garment_imu_set.imu_axis
# vertices = posed_garment.vertices
# rots, pos = imu_syn(vertices)
# print(rots)
# print(pos)

def obj_load_vertices(path='./T-Pose_garment.obj'):
    from pywavefront import Wavefront
    import numpy as np
    obj = Wavefront(path)
    return np.array(obj.vertices)




