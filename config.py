import torch

imu_num = 4

amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']

class joint_set:
    joint_name_list = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
                       "spine3", "l_toe", "r_toe", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
                       "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_palm", "r_palm"]
    # 设置要预测的关节
    # index_pose = torch.tensor([3, 6, 9, 12, 16, 17, 18, 19])
    index_pose = torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19])
    index_joint = torch.tensor([3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21])


    index_l_elbow = 20
    index_r_elbow = 21

class garment_imu_set:
    # 起点/终点
    imu_axis = {'left' :{'axis_1':[916, 4795],  'axis_2':[4796, 4828], 'order':'zx'},
                'right':{'axis_1':[8504, 2109], 'axis_2':[8529, 8369], 'order':'zx'},
                'back' :{'axis_1':[7916, 6021], 'axis_2':[6022, 1226], 'order':'xy'},
                'root' :{'axis_1':[7972, 6000], 'axis_2':[1220, 5999], 'order':'xy'}}

class paths:

    raw_amass_dir = 'E:/DATA/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'E:\DATA\processed_AMASS_upper'         # output path for the synthetic AMASS dataset

    raw_dipimu_dir = 'data/dataset_raw/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'E:\DATA\processed_DIP'      # output path for the preprocessed DIP-IMU dataset

    lip_dir = './LIP_Dataset'

    # smpl_file = 'models/SMPL_male.pkl'              # official SMPL model path
    smpl_file = 'E:\DATA\smpl\smpl/SMPL_MALE.pkl'  # official SMPL model path

    vis_dir = './data/vis' # output path for visualization

acc_scale = 30
vel_scale = 3
