import numpy as np
import torch

from Aplus.tools.annotations import timing
from Aplus.data.process import add_gaussian_noise
from Aplus.runner import *

from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator, PerJointAccErrorEvaluator
from articulate.math.angular import RotationRepresentation, quaternion_to_rotation_matrix, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
import random
import quaternion
from articulate.evaluator import mean_vector_length
import articulate as art
from tqdm import tqdm

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

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
def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss

def DotF_loss(pred, real, vec_dim=3):
    pred = pred.view(-1, vec_dim)
    real = real.view(-1, vec_dim)

    norm_p = pred.detach().norm(dim=-1, keepdim=True) + 1e-2
    pred = pred / norm_p

    norm_r = real.detach().norm(dim=-1, keepdim=True) + 1e-2
    real = real / norm_r

    dot = (pred * real).sum(-1)
    loss_func = nn.MSELoss()
    return loss_func(dot, torch.ones(size=dot.shape).float().to('cuda:0'))


def r6d_global_y_rot(r, angle):
    sin_x = np.sin(angle)
    cos_x = np.cos(angle)
    r = r.reshape(-1, 6)
    r = torch.cat([cos_x*r[:, [0]]+sin_x*r[:, [2]], r[:, [1]], -sin_x*r[:, [0]]+cos_x*r[:, [2]],
                   cos_x*r[:, [3]]+sin_x*r[:, [5]], r[:, [4]], -sin_x*r[:, [3]]+cos_x*r[:, [5]]], dim=-1)

    # rot = [[ cos_x, sin_x, 0],
    #        [-sin_x, cos_x, 0],
    #        [0,      0,     1]]
    # rot = torch.tensor(rot).float()
    # print(r.shape)
    return r

# n x 1 x 3
def body_acc_global_y_rot(acc, root_oris, angle):
    global_acc = acc.bmm(root_oris.transpose(-2, -1))
    new_acc = global_acc.matmul(bulid_rot(theta=180*angle/np.pi, rotation_axis=[0, 1, 0]).to(global_acc.device)).bmm(root_oris)
    return new_acc


class MyTrainer(BaseTrainer):
    def __init__(self, model:nn.Module, data, optimizer, batch_size, loss_func, initializer=None):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'loss_eval', 'ang_err'])
        self.checkpoint = None
        if initializer is not None:
            self.initializer = initializer
            self.initializer_optim = torch.optim.Adam(self.initializer.parameters(), lr=1e-3, weight_decay=0)
        else:
            self.initializer = None

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None, noise_sigma=None):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                       drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_loss = DataMeter()

        for e in range(epoch):

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_loss.reset()
            try:
                if self.model.SemoAE is not None:
                    self.model.poser.train()
                    self.model.SemoAE.eval()
                else:
                    self.model.train()
            except:
                self.model.train()

            if self.initializer is not None:
                self.initializer.train()

            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                x, y = data
                x = x.to(device)
                y = y.to(device)
                batch_size, seq_len = x.shape[0], x.shape[1]
                try:
                    if self.model.SemoAE is not None:
                        if x.shape[2] == 36:
                            x = torch.cat([x[:, :, :12], self.model.SemoAE.make_artifacts(x[:, :, 12:36], eta=3)], dim=-1)
                        else:
                            x = torch.cat([x[:, :, :12], self.model.SemoAE.make_artifacts(x[:, :, 12:36], eta=3), x[:, :, 36:]], dim=-1)
                except:
                    pass

                if self.initializer is not None:
                    # h_0 = self.lstm_init_state(init_pose=y[:, 0, :])
                    # c_0 = torch.zeros(size=[2, batch_size, 256]).to(h_0.device)
                    h_0, c_0, h_1, c_1 = self.initializer(y[:, 0, :])
                    y_hat, _, _, _, _ = self.model(x[:, 1:, :], h_0, c_0, h_1, c_1)

                    y = y[:, 1:, :]
                else:
                    y_hat = self.model(x)


                # loss = self.loss_func(y_hat[:, 20:], y[:, 20:])
                loss = self.loss_func(y_hat[:, seq_len//2:], y[:, seq_len//2:])

                loss.backward()

                self.optimizer.step()

                # 每个batch记录一次
                avg_meter_loss.update(value=loss.item(), n_sample=len(y))

                print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # 获取整个epoch的loss
            loss_train = avg_meter_loss.get_avg()
            self.epoch += 1
            print('')

            if evaluator is not None:
                loss_eval, ang_err = evaluator.run(initializer=self.initializer)
            else:
                loss_eval, ang_err = -1, -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss_train, 'loss_eval': loss_eval, 'ang_err': ang_err})

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()

class MyEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size, rot_type='r6d'):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)

    @torch.no_grad()
    def run(self, device=None, initializer=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        avg_meter_loss = DataMeter()
        avg_meter_ang_err = DataMeter()
        avg_meter_per_joint_ang_err = DataMeter()
        self.model.eval()
        if initializer is not None:
            initializer.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            batch_size = x.shape[0]
            if initializer is not None:
                init_pose = y[:, 0, :]
                # h_0 = initializer(init_pose).reshape(-1, 2, 256)
                # h_0 = h_0.permute(1, 0, 2).contiguous()
                # c_0 = torch.zeros(size=[2, batch_size, 256]).to(h_0.device)
                h_0, c_0, h_1, c_1 = initializer(init_pose)
                y_hat, _, _, _, _ = self.model(x[:, 1:, :], h_0, c_0, h_1, c_1)
                y = y[:, 1:, :]
            else:
                y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)

            avg_meter_loss.update(value=loss.item(), n_sample=len(y))

            # 计算角度误差
            ang_err = self.rot_err_evaluator(p=y_hat[:, -1], t=y[:, -1]).cpu()
            avg_meter_ang_err.update(value=ang_err, n_sample=len(y))

            per_joint_ang_err = self.per_joint_rot_err_evaluator(p=y_hat[:, -1], t=y[:, -1], joint_num=10).cpu()
            avg_meter_per_joint_ang_err.update(value=per_joint_ang_err, n_sample=len(y))

        loss_train = avg_meter_loss.get_avg()
        ang_err = avg_meter_ang_err.get_avg()
        per_joint_ang_err = avg_meter_per_joint_ang_err.get_avg()
        print(per_joint_ang_err)
        # return loss_train, ang_err, per_joint_ang_err
        return loss_train, ang_err

    @classmethod
    def from_trainner(cls, trainner, data_eval, rot_type='r6d'):
        return cls(model=trainner.model, loss_func=trainner.loss_func, batch_size=trainner.batch_size, data=data_eval, rot_type=rot_type)


class BiPoserTrainner(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func, initializer=None, SemoAE=None):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'loss_eval', 'ang_err'])
        self.checkpoint = None
        self.SemoAE = SemoAE
        self.loss_func_joint = nn.MSELoss()

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None, noise_sigma=None, eta=1):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                 drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_loss = DataMeter()

        for e in range(epoch):

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_loss.reset()


            self.model.train()

            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                x, y_s1, y_s2 = data
                x = x.to(device)
                y_s1 = y_s1.to(device)
                y_s2 = y_s2.to(device)
                batch_size, seq_len = x.shape[0], x.shape[1]
                if self.SemoAE is not None:
                    x =self.SemoAE.secondary_motion_gen(x, eta=eta)
                y_s1_hat, y_s2_hat = self.model(x)

                loss_joint = self.loss_func_joint(y_s1_hat[:, seq_len//4:], y_s1[:, seq_len//4:])
                # print(y_s2_hat.shape, y_s2.shape)
                loss_pose = self.loss_func(y_s2_hat[:, seq_len//4:], y_s2[:, seq_len//4:])
                loss = loss_joint + loss_pose
                loss.backward()

                self.optimizer.step()

                # 每个batch记录一次
                avg_meter_loss.update(value=loss.item(), n_sample=len(y_s1))

                print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # 获取整个epoch的loss
            loss_train = avg_meter_loss.get_avg()
            self.epoch += 1
            print('')

            if evaluator is not None:
                loss_eval, ang_err = evaluator.run()
            else:
                loss_eval, ang_err = -1, -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update(
                {'epoch': self.epoch, 'loss_train': loss_train, 'loss_eval': loss_eval, 'ang_err': ang_err})

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()

class BiPoserEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size, rot_type='r6d'):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)
        self.loss_func_joint = nn.MSELoss()

    @torch.no_grad()
    def run(self, device=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        avg_meter_loss = DataMeter()
        avg_meter_ang_err = DataMeter()
        avg_meter_per_joint_ang_err = DataMeter()
        self.model.eval()
        for i, data in enumerate(data_loader):
            x, y_s1, y_s2 = data
            x = x.to(device)
            y_s1 = y_s1.to(device)
            y_s2 = y_s2.to(device)

            seq_len = x.shape[1]
            y_s1_hat, y_s2_hat = self.model(x)

            loss_joint = self.loss_func_joint(y_s1_hat[:, seq_len // 4:], y_s1[:, seq_len // 4:])
            loss_pose = self.loss_func(y_s2_hat[:, seq_len // 4:], y_s2[:, seq_len // 4:])
            
            loss = loss_joint+loss_pose

            avg_meter_loss.update(value=loss.item(), n_sample=len(y_s1))

            # 计算角度误差
            ang_err = self.rot_err_evaluator(p=y_s2_hat[:, -1], t=y_s2[:, -1]).cpu()
            avg_meter_ang_err.update(value=ang_err, n_sample=len(y_s1))

            per_joint_ang_err = self.per_joint_rot_err_evaluator(p=y_s2_hat[:, -1], t=y_s2[:, -1], joint_num=10).cpu()
            avg_meter_per_joint_ang_err.update(value=per_joint_ang_err, n_sample=len(y_s1))

        loss_train = avg_meter_loss.get_avg()
        ang_err = avg_meter_ang_err.get_avg()
        per_joint_ang_err = avg_meter_per_joint_ang_err.get_avg()
        print(per_joint_ang_err)
        # return loss_train, ang_err, per_joint_ang_err
        return loss_train, ang_err

    @classmethod
    def from_trainner(cls, trainner, data_eval, rot_type='r6d'):
        return cls(model=trainner.model, loss_func=trainner.loss_func, batch_size=trainner.batch_size, data=data_eval, rot_type=rot_type)


class SemoAETrainner(BaseTrainer):
    def __init__(self, model:nn.Module, data, optimizer, batch_size, loss_func):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'tight_err_train', 'loose_err_train', 'tight_err_eval', 'loose_err_eval'])
        self.checkpoint = None
        self.rot_err_evaluator = RotationErrorEvaluator(rep=RotationRepresentation.R6D)


    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None, d_loss_weight=1):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                       drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_tight = DataMeter()
        avg_meter_loose = DataMeter()
        avg_meter_domain = DataMeter()


        for e in range(epoch):
            optimizer = self.optimizer

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_tight.reset()
            avg_meter_loose.reset()
            avg_meter_domain.reset()

            self.model.train()

            for i, data in enumerate(data_loader):
                optimizer.zero_grad()
                loose_data, tight_data = data
                loose_data = loose_data.to(device)
                tight_data = tight_data.to(device)

                loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]

                # 根节点绕y(up)轴随机旋转
                x = random.uniform(-np.pi/2, np.pi/2)

                loose_rot = torch.cat([loose_rot[:, :-6], r6d_global_y_rot(r=loose_rot[:, -6:], angle=x)], dim=-1)
                tight_rot = torch.cat([tight_rot[:, :-6], r6d_global_y_rot(r=tight_rot[:, -6:], angle=x)], dim=-1)

                loose_data = torch.cat([loose_acc, loose_rot], dim=-1)
                tight_data = torch.cat([tight_acc, tight_rot], dim=-1)

                # 隐藏层表示
                loose_code = self.model.encode(loose_data)
                tight_code = self.model.encode(tight_data)

                loss_distribution = self.model.dis_normalizer(x1=tight_code, x2=loose_code)

                # 重映射结果 只使用一个decoder
                loose_data_recon = self.model.decode(loose_code)
                tight_data_recon = self.model.decode(tight_code)

                # 分离出加速度与姿态(r6d)
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]
                loose_acc_recon, loose_rot_recon = loose_data_recon[:, :12], loose_data_recon[:, 12:]
                tight_acc_recon, tight_rot_recon = tight_data_recon[:, :12], tight_data_recon[:, 12:]

                loss_loose = self.loss_func(loose_data_recon, loose_data)
                loss_tight = self.loss_func(tight_data_recon, tight_data)

                loss = loss_loose + loss_tight + loss_distribution * d_loss_weight

                loss.backward()

                optimizer.step()

                # 每个batch记录一次
                avg_meter_tight.update(value=self.rot_err_evaluator(p=tight_rot_recon, t=tight_rot).cpu(), n_sample=len(loose_data))
                avg_meter_loose.update(value=self.rot_err_evaluator(p=loose_rot_recon, t=loose_rot).cpu(), n_sample=len(loose_data))
                avg_meter_domain.update(value=loss_distribution.cpu(), n_sample=len(loose_data))
                # avg_meter_domain.update(value=self.rot_err_evaluator(p=tight_rot_noised, t=tight_rot).cpu(), n_sample=len(imu_rot))

                print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # 获取整个epoch的loss
            tight_err_train = avg_meter_tight.get_avg()
            loose_err_train = avg_meter_loose.get_avg()
            self.epoch += 1
            # print(loss_tight, loss_loose, loss_distribution)

            if evaluator is not None:
                tight_err_eval, loose_err_eval = evaluator.run()
            else:
                tight_err_eval, loose_err_eval = -1, -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update({'epoch': self.epoch, 'tight_err_train': tight_err_train, 'loose_err_train': loose_err_train,
                                     'tight_err_eval': tight_err_eval, 'loose_err_eval':loose_err_eval})

            # 打印最新一个epoch的训练记录
            print('mmd_loss:',  avg_meter_domain.get_avg())
            # print('std:', self.model.dis_normalizer.get_std())

            self.log_manager.print_latest()

class SemoAEEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.rot_err_evaluator = RotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_acc_err_evaluator = PerJointAccErrorEvaluator()

    @torch.no_grad()
    def run(self, device=None, noise_eta=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        # AverageMeter用于计算整个epoch的loss
        avg_meter_tight = DataMeter()
        avg_meter_loose = DataMeter()
        avg_meter_per_joint_tight = DataMeter()
        avg_meter_per_joint_loose = DataMeter()
        avg_meter_per_joint_acc_tight = DataMeter()
        avg_meter_per_joint_acc_loose = DataMeter()
        avg_meter_mmd = DataMeter()
        self.model.eval()
        # self.model.dis_normalizer.eval()
        for i, data in enumerate(data_loader):

            loose_data, tight_data = data
            loose_data = loose_data.to(device)
            tight_data = tight_data.to(device)

            # 隐藏层表示
            loose_code = self.model.encode(loose_data)
            tight_code = self.model.encode(tight_data)

            loss_distribution = self.model.dis_normalizer(x1=tight_code, x2=loose_code)

            # 重映射结果
            loose_tight_recon = self.model.decode(loose_code)
            tight_tight_recon = self.model.decode(tight_code)

            # 分离出加速度与姿态(r6d)
            tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]
            loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
            loose_acc_recon, loose_rot_recon = loose_tight_recon[:, :12], loose_tight_recon[:, 12:]
            tight_acc_recon, tight_rot_recon = tight_tight_recon[:, :12], tight_tight_recon[:, 12:]

            # 每个batch记录一次
            avg_meter_tight.update(value=self.rot_err_evaluator(p=tight_rot_recon, t=tight_rot).cpu(), n_sample=len(loose_data))
            avg_meter_loose.update(value=self.rot_err_evaluator(p=loose_rot_recon, t=loose_rot).cpu(), n_sample=len(loose_data))
            avg_meter_mmd.update(value=loss_distribution.cpu(),
                                   n_sample=len(loose_data))
            # 每个IMU的误差
            # 姿态误差
            per_joint_ang_err = self.per_joint_rot_err_evaluator(p=tight_rot_recon, t=tight_rot, joint_num=4).cpu()
            avg_meter_per_joint_tight.update(value=per_joint_ang_err, n_sample=len(loose_data))

            per_joint_ang_err = self.per_joint_rot_err_evaluator(p=loose_rot_recon, t=loose_rot, joint_num=4).cpu()
            avg_meter_per_joint_loose.update(value=per_joint_ang_err, n_sample=len(loose_data))

            # acc误差
            per_joint_acc_err = self.per_joint_acc_err_evaluator(p=tight_acc_recon, t=tight_acc, joint_num=4).cpu()
            avg_meter_per_joint_acc_tight.update(value=per_joint_acc_err, n_sample=len(loose_data))

            per_joint_acc_err = self.per_joint_acc_err_evaluator(p=loose_acc_recon, t=loose_acc, joint_num=4).cpu()
            avg_meter_per_joint_acc_loose.update(value=per_joint_acc_err, n_sample=len(loose_data))

            print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

        # 获取整个epoch的loss
        tight_err_train = avg_meter_tight.get_avg()
        loose_err_train = avg_meter_loose.get_avg()

        print('tight angle error', avg_meter_per_joint_tight.get_avg().mean())
        print('loose angle error', avg_meter_per_joint_loose.get_avg().mean())
        print('tight acc error', avg_meter_per_joint_acc_tight.get_avg().mean())
        print('loose acc error', avg_meter_per_joint_acc_loose.get_avg().mean())
        print('mmd', avg_meter_mmd.get_avg())

        return tight_err_train, loose_err_train

class PoseEvaluator:
    def __init__(self, rot_type='r6d', index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
                 index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]):
        self.index_joint = index_joint
        self.index_pose = index_pose
        self.body_model = art.ParametricModel('E:\DATA\smpl\smpl/SMPL_MALE.pkl')

        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_type = rot_type
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        p = p.cpu()
        t = t.cpu()
        joint_num = len(self.index_pose)
        mjre = self.rot_err_evaluator(p, t)
        mpjre = self.per_joint_rot_err_evaluator(p, t, joint_num=joint_num)
        if self.rot_type == 'r6d':
            p = p.reshape(-1, 6)
            t = t.reshape(-1, 6)

            p = r6d_to_rotation_matrix(p).reshape(-1, joint_num, 3, 3)
            # p = rotation_matrix_to_axis_angle(p).reshape(-1, joint_num, 3)

            t = r6d_to_rotation_matrix(t).reshape(-1, joint_num, 3, 3)
            # t = rotation_matrix_to_axis_angle(t).reshape(-1, joint_num, 3)

            p_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
            p_full_body[:, self.index_pose] = p

            t_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
            t_full_body[:, self.index_pose] = t


        shape = torch.zeros(10)
        tran = torch.zeros(len(p_full_body), 3)

        p_grot, p_joint = self.body_model.forward_kinematics(p_full_body, shape, tran, calc_tight=False)
        t_grot, t_joint = self.body_model.forward_kinematics(t_full_body, shape, tran, calc_tight=False)

        p_joint = p_joint[:, self.index_joint]
        t_joint = t_joint[:, self.index_joint]

        mjpe = torch.cat([mean_vector_length(p_joint[:, i, :] - t_joint[:, i, :]).unsqueeze(0) for i in range(len(self.index_joint))], dim=0)

        print(f'平均角度误差: {mjre}')
        print(f'平均各关节角度误差: {mpjre}')
        print(f'平均关节位置误差: {mjpe} avg: {mjpe.mean()}')

class PoseEvaluatorWithStd:
    def __init__(self, rot_type='r6d', index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
                 index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]):
        self.index_joint = index_joint
        self.index_pose = index_pose
        self.body_model = art.ParametricModel('E:\DATA\smpl\smpl/SMPL_MALE.pkl')

        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_type = rot_type
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        # self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)
    @torch.no_grad()
    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        p = p.cpu()
        t = t.cpu()
        p_all = p
        t_all = t

        joint_err = []
        position_err = []
        per_position_err = []

        for i in tqdm(range(len(p_all))):
            p = p_all[i]
            t = t_all[i]
            joint_num = len(self.index_pose)
            mjre = self.rot_err_evaluator(p, t)
            joint_err.append(mjre)
            # mpjre = self.per_joint_rot_err_evaluator(p, t, joint_num=joint_num)
            if self.rot_type == 'r6d':
                p = p.reshape(-1, 6)
                t = t.reshape(-1, 6)

                p = r6d_to_rotation_matrix(p).reshape(-1, joint_num, 3, 3)
                # p = rotation_matrix_to_axis_angle(p).reshape(-1, joint_num, 3)

                t = r6d_to_rotation_matrix(t).reshape(-1, joint_num, 3, 3)
                # t = rotation_matrix_to_axis_angle(t).reshape(-1, joint_num, 3)

                p_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
                p_full_body[:, self.index_pose] = p

                t_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
                t_full_body[:, self.index_pose] = t


            shape = torch.zeros(10)
            tran = torch.zeros(len(p_full_body), 3)

            p_grot, p_joint = self.body_model.forward_kinematics(p_full_body, shape, tran, calc_mesh=False)
            t_grot, t_joint = self.body_model.forward_kinematics(t_full_body, shape, tran, calc_mesh=False)

            p_joint = p_joint[:, self.index_joint]
            t_joint = t_joint[:, self.index_joint]

            mjpe = torch.cat([mean_vector_length(p_joint[:, i, :] - t_joint[:, i, :]).unsqueeze(0) for i in range(len(self.index_joint))], dim=0)
            # print(mjpe.mean())
            position_err.append(mjpe.mean().detach())
            per_position_err.append(mjpe.detach().cpu().unsqueeze(0))

        joint_err = np.array(joint_err, dtype=float)
        position_err = np.array(position_err, dtype=float)
        per_position_err = torch.cat(per_position_err, dim=0)
        # print(per_position_err.shape)
        print(f'平均角度误差: {joint_err.mean()} ± {joint_err.std()}')
        print(f'平均关节位置误差: {position_err.mean()} ± {position_err.std()}')
        print(f'平均各关节位置误差: {per_position_err.mean(dim=0)}')


class PoseVisualizer:
    def __init__(self, rot_type='r6d', index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
                 index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]):
        self.index_joint = index_joint
        self.index_pose = index_pose
        self.body_model = art.ParametricModel('../TransPose/models/SMPL_male.pkl')
        # '/media/brcao/eData4TB1/Repos/bryanbocao/sparseimu/TransPose/models/SMPL_male.pkl') # edit # art.ParametricModel('E:\DATA\smpl\smpl/SMPL_MALE.pkl')

        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_type = rot_type

        self.pred_24x3x3 = None
        self.gt_24x3x3 = None
        self.tran_3 = None

    @torch.no_grad()
    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        p = p.cpu()
        t = t.cpu()
        joint_num = len(self.index_pose)
        # mjre = self.rot_err_evaluator(p, t)
        # mpjre = self.per_joint_rot_err_evaluator(p, t, joint_num=joint_num)
        if self.rot_type == 'r6d':
            p = p.reshape(-1, 6)
            t = t.reshape(-1, 6)

            p = r6d_to_rotation_matrix(p).reshape(-1, joint_num, 3, 3)
            # p = rotation_matrix_to_axis_angle(p).reshape(-1, joint_num, 3)

            t = r6d_to_rotation_matrix(t).reshape(-1, joint_num, 3, 3)
            # t = rotation_matrix_to_axis_angle(t).reshape(-1, joint_num, 3)

            p_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
            p_full_body[:, self.index_pose] = p

            t_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
            t_full_body[:, self.index_pose] = t

        shape = torch.zeros(10)
        tran = torch.zeros(len(p_full_body), 3)

        self.pred_24x3x3, self.gt_24x3x3, self.tran_3 = p_full_body, t_full_body, tran
        
        # Save joints for visualization
        os.makedirs(paths.vis_dir, exist_ok=True)
        pred_24x3x3_path = f'{paths.vis_dir}/pred_24x3x3.pt' # edit
        gt_24x3x3_path = f'{paths.vis_dir}/gt_24x3x3.pt' # edit
        tran_3_path = f'{paths.vis_dir}/tran_3.pt' # edit
        torch.save(self.pred_24x3x3, pred_24x3x3_path); print(f'{pred_24x3x3_path} saved!')
        torch.save(self.gt_24x3x3, gt_24x3x3_path); print(f'{gt_24x3x3_path} saved!')
        torch.save(self.tran_3, tran_3_path); print(f'{tran_3_path} saved!')
