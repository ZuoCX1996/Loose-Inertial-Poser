import config
from my_model import *
from my_data import *
from articulate.math.angular import rotation_matrix_to_r6d, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
import time
from socket import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare model

model_s1 = EasyLSTM(n_input=36, n_hidden=256, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
model_s2 = EasyLSTM(n_input=36+33, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
poser_model = BiPoser(net_s1=model_s1, net_s2=model_s2).to(device)

poser_model.restore(checkpoint_path=f'./checkpoint/LIP_10.pth')
poser_model.eval()


# load IMU data
data_path = './LIP_Dataset/s1_open_random'

# IMU order: left forearm, right forearm, back, waist
acc = torch.load(os.path.join(data_path, 'acc.pt')).reshape(-1, 4, 3).to(device)
rot = torch.load(os.path.join(data_path, 'rot.pt')).reshape(-1, 4, 3, 3).to(device)

# down sample to 30 Hz
acc = acc[::2]
rot = rot[::2]


h_1, c_1 = torch.zeros(2, 1, 256).to(device), torch.zeros(2, 1, 256).to(device)
h_2, c_2 = torch.zeros(2, 1, 256).to(device), torch.zeros(2, 1, 256).to(device)

server_for_unity = socket(AF_INET, SOCK_STREAM)
server_for_unity.bind(('127.0.0.1', 8888))
server_for_unity.listen(1)

print('Server start. Please run unity.')
conn, addr = server_for_unity.accept()

total_frame = len(acc)
for i in range(total_frame):
    time.sleep(1/40)
    print(f'\r{i} | {total_frame}', end='')

    _acc = acc[i]
    _rot = rot[i]

    # normalization
    _acc = torch.cat([_acc[:3] - _acc[[-1]], _acc[[-1]]], dim=0).matmul(_rot[[-1]]) / 30
    _rot = torch.cat([_rot[[-1]].transpose(-2, -1).matmul(_rot[:3]), _rot[[-1]]], dim=0)

    # convert to 6d representation
    _rot = rotation_matrix_to_r6d(_rot)

    x = torch.cat([_acc.flatten(0), _rot.flatten(0)], dim=-1).reshape(1, 1, 36).to(device)
    _joint, _pose, h_1, c_1, h_2, c_2 = poser_model(x, h_1, c_1, h_2, c_2)

    _pose = _pose.reshape(10, 6)

    # convert to axis angle
    _pose = r6d_to_rotation_matrix(_pose)
    _pose = rotation_matrix_to_axis_angle(_pose)
    _pose = _pose.cpu()

    # mapping the pred joint rot to SMPL template
    full_body_pose = torch.zeros(24, 3)
    full_body_pose[config.joint_set.index_pose] = _pose

    tran = torch.zeros(3)

    # send pose to unity
    s = ','.join(['%g' % v for v in full_body_pose.view(-1)]) + '#' + \
        ','.join(['%g' % v for v in tran]) + '$'

    conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here




