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
from my_model import SemoAE

import os


seq_len = 128
use_elbow_angle = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集准备
amass_data = AmassData.load_data(folder_path=paths.amass_dir, use_elbow_angle=use_elbow_angle)
lip_data = LipData.load_data(folder_path=paths.lip_dir, use_elbow_angle=use_elbow_angle, back_fix=False)
#
# train_size, test_size = int(len(dip_data['x_s1'])*train_split), int(len(dip_data['x_s1'])*train_split)
train_size = 34213 + 32529 + 33654 + 30755 + 33632 + 33283 + 30309 + 30699
test_size = 26401 + 25950

# 添加step=2 适才能应30帧输入

data_train = AmassData(x=amass_data['x_s1'],
                       y=amass_data['joint_upper_body'],
                       y2=amass_data['pose_all'], seq_len=seq_len, step=2)

data_test  = LipData(x=lip_data['x_s1'],
                       y=lip_data['joint_upper_body'],
                       y2=lip_data['pose_all'], seq_len=seq_len, step=2)

# 网络定义
model_s1 = EasyLSTM(n_input=36, n_hidden=256, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
model_s2 = EasyLSTM(n_input=36+33, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2).to(device)
# --------合练--------
epoch = 1
poser_model = BiPoser(net_s1=model_s1, net_s2=model_s2).to(device)

SemoAE_model = SemoAE(feat_dim=12+24, encode_dim=16).to(device)
SemoAE_model.restore(checkpoint_path='./checkpoint/SemoAE_10.pth')

optimizer = torch.optim.Adam(poser_model.parameters(), lr=5e-4, weight_decay=1e-5)

criterion = nn.MSELoss()
trainner = BiPoserTrainner(model=poser_model, data=data_train, optimizer=optimizer, batch_size=256, loss_func=criterion,
                        SemoAE=SemoAE_model)
evaluator = BiPoserEvaluator.from_trainner(trainner, data_eval=data_test)
# trainner.restore(checkpoint_path=f'./checkpoint/LIP_10.pth', load_optimizer=True)
# evaluator.run()

for i in range(10):
    model_name = 'LIP'
    trainner.run(epoch=epoch, evaluator=evaluator, data_shuffle=True, eta=2)
    trainner.save(folder_path='./checkpoint', model_name=model_name)
    trainner.log_export(f'./log/{model_name}.xlsx')



