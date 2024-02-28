
from config import paths, joint_set
from my_data import *
from Aplus.runner import BaseEvaluator
from my_trainner import *
from my_model import *


import os


train_split, test_split = 0.95, 0.05

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = SynPairedIMUData.load_data(folder_path=paths.amass_dir, shuffle=True, clothes_imu_calibration=True)
data_len = len(dataset['data_mesh'])
train_size, test_size = int(data_len*train_split), int(data_len*test_split)


model = SemoAE(feat_dim=12+24, encode_dim=16).to(device)

epoch = 1

data_train = SynPairedIMUData(x=dataset['data_garment'][:train_size], y=dataset['data_mesh'][:train_size], shuffle=True)
data_test  = SynPairedIMUData(x=dataset['data_garment'][train_size:train_size+test_size], y=dataset['data_mesh'][train_size:train_size+test_size])
print(data_train.__len__())
print(data_test.__len__())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()
trainner = SemoAETrainner(model=model, data=data_train, optimizer=optimizer, batch_size=512, loss_func=criterion)
evaluator = SemoAEEvaluator.from_trainner(trainner, data_eval=data_test)

# trainner.restore(checkpoint_path='./checkpoint/SemoAE_10.pth', load_optimizer=True)
evaluator.run()
model_name='SemoAE'
for i in range(10):
    trainner.run(epoch=epoch, evaluator=evaluator, data_shuffle=True, d_loss_weight=1)
    trainner.save(folder_path='./checkpoint', model_name=model_name)
    trainner.log_export(f'./log/{model_name}.xlsx')




