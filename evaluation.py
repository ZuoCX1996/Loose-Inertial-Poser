import torch.nn as nn
import torch
import random
import pandas as pd
import numpy as np
from my_model import *
from config import paths, joint_set
from my_data import *
from my_trainner import MyTrainer, MyEvaluator
from Aplus.models import EasyLSTM
from config import joint_set, paths
from my_trainner import PoseEvaluator, PoseEvaluatorWithStd

import os

seq_len = 128
use_elbow_angle = False
data_type = 'all'

evaluator = PoseEvaluatorWithStd()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lip_data = LipData.load_data(folder_path=paths.lip_dir, use_elbow_angle=use_elbow_angle, type=data_type)
# lip_data = LipData.load_data(folder_path='E:\DATA\LIP-tight', use_elbow_angle=use_elbow_angle)

data_test = LipData(x=lip_data['x_s1'],
                       y=lip_data['joint_upper_body'],
                       y2=lip_data['pose_all'], seq_len=seq_len, step=2)


# 网络定义
model_s1 = EasyLSTM(n_input=36, n_hidden=256, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
model_s2 = EasyLSTM(n_input=36+33, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
# lstm_initializer = Initiallizer(n_input=33).to(device)
# lstm_initializer.restore('./checkpoint/initializer_14.pth')
lstm_initializer = None
# --------合并训练--------
epoch = 1
poser_model = BiPoser(net_s1=model_s1, net_s2=model_s2, export_mode=True).to(device)
# model_all = PluginNet(poser_net=poser_model, rot_data_dim=24, stabilizer=None, export_mode=True)

poser_model.restore('./checkpoint/LIP_10.pth')
poser_model.eval()

# # --------合并训练--------
# poser_model = EasyLSTM(n_input=36, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
# model_all = PluginNet(poser_net=poser_model, rot_data_dim=24, stabilizer=None, export_mode=True)
# model_all.restore('./checkpoint/IMUPoser_10.pth')
# model_all = poser_model
# model_all.eval()



input = data_test.x.unsqueeze(0).to(device)
# print(input[0, :10, -24:].reshape(-1,4,6))
# print(input)
gt = data_test.y2.to(device)
pred = poser_model(input).squeeze(0)
# print(pred[100])

# torch.save(gt, './GT.pt')
evaluator(p=pred, t=gt)
#
# torch.save(pred, './IMUPoser_pred.pt')



