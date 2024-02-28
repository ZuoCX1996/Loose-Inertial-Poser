r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""


import articulate as art
import torch
import os
import pickle
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob
from articulate.math.angular import RotationRepresentation
from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator
# from Aplus.utils.data_meter import DataMeter

def process_amass(smooth_n=4):

    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])

    # ours
    vi_mask = torch.tensor([1961, 5424, 3059, 3021])
    ji_mask = torch.tensor([18, 19, 12, 0])

    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            # print(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    print(shape)
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        print(p.shape, shape[i].shape, tran[b:b + l].shape)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    # print('Saving')
    # os.makedirs(paths.amass_dir, exist_ok=True)
    # torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    # torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    # torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    # torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    # torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    # torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    # print('Synthetic AMASS dataset is saved at', paths.amass_dir)


def process_tailorNet_syn(smooth_n=4):

    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])

    # ours
    vi_mask = torch.tensor([1961, 5424, 3059, 3021])
    ji_mask = torch.tensor([18, 19, 12, 0])

    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)


def tailorNet_data_process(syn_data_path='E:\DATA\processed_AMASS\clothes_imu_syn'):
    def _syn_acc(v, smooth_n=4):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc
    oris = []
    positions = []
    accs = []
    for i in tqdm(range(9124)):
        ori = torch.load(os.path.join(syn_data_path, f'syn_imu_rot_{i}.pt'))
        pos = torch.load(os.path.join(syn_data_path, f'syn_imu_pos_{i}.pt'))
        # print(ori.shape)
        # print(pos.shape)
        oris.append(ori.float())
        positions.append(pos.float())
        accs.append(_syn_acc(pos).float())
    torch.save(oris, os.path.join(paths.amass_dir, 'syn_rot_on_garment.pt'))
    torch.save(positions, os.path.join(paths.amass_dir, 'syn_pos_on_garment.pt'))
    torch.save(accs, os.path.join(paths.amass_dir, 'syn_acc_on_garment.pt'))

# def syn_garment_imu_analysis():
#     per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=RotationRepresentation.ROTATION_MATRIX)
#     rot_amass = torch.load('E:\DATA\processed_AMASS/vrot.pt')
#     rot_garment_syn = torch.load('E:\DATA\processed_AMASS\syn_rot_on_garment.pt')
#     avg_meter = DataMeter()
#
#     for i in range(len(rot_amass)):
#         rot_1 = rot_amass[i]
#         rot_2 = rot_garment_syn[i]
#         err = per_joint_rot_err_evaluator(p=rot_1, t=rot_2, joint_num=4)
#         avg_meter.update(value=err, n_sample=len(err))
#         print(i, err)
#     print(avg_meter.get_avg())

def get_tpose_joint_position():
    body_model = art.ParametricModel(paths.smpl_file)

    shape = torch.tensor([-0.53062872, -0.03227116,  1.76599983,  1.0983942, 0.69472912, -0.15725264, 0.52576423,
                          -0.13045748,  0.24933542, -1.15570345])
    tran = torch.tensor([[0, 0, 0]])

    pose = torch.eye(3).squeeze(0).repeat(24, 1, 1)
    pose = pose.reshape(-1, 24, 3, 3)

    grot, joint, vert = body_model.forward_kinematics(pose, shape, tran, calc_mesh=True)

    return joint


if __name__ == '__main__':
    process_amass()
    # process_dipimu()
    # process_totalcapture()
    # tailorNet_data_process()
    # syn_garment_imu_analysis()
