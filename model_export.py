from my_model import *
import torch.nn.functional as F


if __name__ == '__main__':
    # 导出Garment2Bone
    # model = DeArtifactor(feat_dim=24, encode_dim=16)
    # model.restore(checkpoint_path='./checkpoint/deArtifactor_relu_16_60.pth')
    # model.export_onnx(input_shapes={'garment_rot_r6d': [-1, 24]}, output_shapes={'bone_rot_r6d': [-1, 24]}, path='./Garment2Bone.onnx')


    # 导出Transpose版poser
    # 导出Initializer

    # lstm_initializer = Initiallizer(n_input=33)
    # lstm_initializer.restore('./checkpoint/initializer_transpose_15.pth')
    # lstm_initializer.restore('./checkpoint/initializer_lv3_9.pth')
    # lstm_initializer.export_onnx(input_shapes={'joint_position': [-1, 33]}, output_shapes={'h_1': [2, -1, 256],
    #                              'c_1': [2, -1, 256], 'h_2': [2, -1, 256], 'c_2': [2, -1, 256]},
    #                   path='./Initializer.onnx')
    #
    # 导出poser
    # stabilizer_model = Stabilizer(feat_dim=12 + 24, hidden_dim=16)
    stabilizer_model = None
    # deart = None

    model_s1 = EasyLSTM(n_input=36, n_hidden=256, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq',
                        dropout=0.2)
    model_s2 = EasyLSTM(n_input=36 + 33, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False,
                        output_type='seq', dropout=0.2)
    model = BiPoser(net_s1=model_s1, net_s2=model_s2)

    model.restore(checkpoint_path='./checkpoint2/LIP_10.pth')
    # model.stabilizer = None
    # model.restore(checkpoint_path='./checkpoint/garment_transpose_baseline_10.pth')

    poser = Poser(net=model)

    poser.export_onnx(input_shapes={'imu_data': [-1, 36], 'h_1': [2, -1, 256], 'c_1': [2, -1, 256],
                                    'h_2': [2, -1, 256], 'c_2': [2, -1, 256]},
                      output_shapes={'r6d': [-1, 24, 6], 'h_1_n': [2, -1, 256], 'c_1_n': [2, -1, 256],
                                    'h_2_n': [2, -1, 256], 'c_2_n': [2, -1, 256]}, path='./LIP_4.onnx')

    # # 带角度版本
    # deart = DeArtifactor(feat_dim=24, encode_dim=16)
    # # deart = None
    #
    # model_s1 = EasyLSTM(n_input=36+4, n_hidden=256, n_output=36, n_lstm_layer=2, bidirectional=False, output_type='seq',
    #                     dropout=0.2)
    # model_s2 = EasyLSTM(n_input=36+4 + 36, n_hidden=256, n_output=60, n_lstm_layer=2, bidirectional=False,
    #                     output_type='seq', dropout=0.2)
    # model_all = BiPoserNet_Transpose(net_s1=model_s1, net_s2=model_s2)
    #
    # model = PluginNet(poser_net=model_all, rot_data_dim=24, deart_plugin=deart, export_mode=True, use_elbow_angle=True)
    # model.restore(checkpoint_path='./checkpoint/garment_poser_angle_15.pth')
    # # model.restore(checkpoint_path='./checkpoint/garment_transpose_gacc_ft_15.pth')
    #
    # poser = Poser(net=model, type='axis_angle', use_elbow_angle=True)
    #
    # poser.export_onnx(input_shapes={'imu_data': [-1, 48+4], 'h_1': [2, -1, 256], 'c_1': [2, -1, 256],
    #                                 'h_2': [2, -1, 256], 'c_2': [2, -1, 256]},
    #                   output_shapes={'r6d': [-1, 24, 3], 'h_1_n': [2, -1, 256], 'c_1_n': [2, -1, 256],
    #                                  'h_2_n': [2, -1, 256], 'c_2_n': [2, -1, 256]}, path='./GarmentPoser_with_angle_axis.onnx')

