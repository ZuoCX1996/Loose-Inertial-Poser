from .base_models import *
from .layers import activation_layer

class EasyDNN(BaseModel):
    def __init__(self, n_input, n_hiddens, n_output, act_func='relu', dropout=None):
        """
        DNN network. The input size should be [batch, n_input]
        Args:
            n_input: dim of input
            n_hiddens: dim of hidden layers. e.g. [64, 128, 64]
            n_output: dim of output
            act_func: 'relu' | 'tanh' | 'LeakyReLu' | 'sigmoid'
            dropout: dropout rate, default:None
        """
        super(EasyDNN, self).__init__()
        channel_list = [n_input] + n_hiddens
        layers = []
        for i in range(len(channel_list)-1):
            mini_layer = [nn.Linear(in_features=channel_list[i], out_features=channel_list[i+1]),
                       activation_layer(act_name=act_func)]
            if dropout is not None:
                mini_layer += [nn.Dropout(dropout)]
            layers += mini_layer

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features=channel_list[-1], out_features=n_output)

    def forward(self, x):
        x = self.network(x)
        x = self.output_layer(x)
        return x