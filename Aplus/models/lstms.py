import torch

from .base_models import *
from .layers import activation_layer


class EasyLSTM(BaseModel):
    def __init__(self, n_input, n_output, n_hidden, n_lstm_layer, bidirectional=False, dropout=0.2,
                 output_type='seq', act_func='tanh', use_cell_state=True):
        """
        LSTM network with 2 Linear Layer. The input size should be [batch, seq_len, n_input]
        Args:
            n_input: Dim of input.
            n_output: Dim of output.
            n_hidden: Hidden size of lstm module.
            n_lstm_layer: Number of lstm layer.
            bidirectional: Use bidirectional lstm.
            dropout: Dropout rate.
            output_type: Choose 'seq' or 'feat'. 'seq' output will be [batch, seq_len, n_output];
            'feat' output will be [batch, n_output]
            use_cell_state: Using both [hidden state] and [cell state] of lstm for 'feat' computing.
        """
        super(EasyLSTM, self).__init__()

        self.use_cell_state = use_cell_state

        # self.bn = nn.BatchNorm1d(num_features=n_input)
        self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(n_input, n_hidden)
        self.act_func = activation_layer(act_name=act_func)

        self.lstm = nn.LSTM(n_hidden, n_hidden, n_lstm_layer, batch_first=True, bidirectional=bidirectional)

        self.h_0 = nn.Parameter(torch.zeros(size=[n_lstm_layer*(2 if bidirectional else 1), 1, n_hidden]))
        self.c_0 = nn.Parameter(torch.zeros(size=[n_lstm_layer*(2 if bidirectional else 1), 1, n_hidden]))
        self.h_0.requires_grad = False
        self.c_0.requires_grad = False

        if output_type == 'seq':
            lstm_out_dim = n_hidden * (2 if bidirectional else 1)
            self.forward = self.forward_seq
        elif output_type == 'feat':
            lstm_out_dim = n_lstm_layer * n_hidden * (2 if use_cell_state else 1) * \
                           (2 if bidirectional else 1)
            self.forward = self.forward_feat
        else:
            raise NameError("output_type should be 'seq' or 'feat'")

        self.fc_2 = nn.Linear(lstm_out_dim, n_output)

    def forward_feat(self, x):

        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.act_func(x)

        h_0 = self.h_0.repeat(1, x.shape[0], 1)
        c_0 = self.c_0.repeat(1, x.shape[0], 1)

        seq_output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # h/c: [n_rnn_layer*(2 if bidirectional else 1), batch_size, n_hidden]
        if self.use_cell_state:
            feat = torch.cat((h_n, c_n), dim=0)
        else:
            feat = h_n
        feat = feat.permute(1, 0, 2)
        feat = feat.reshape(feat.shape[0], -1)

        feat = self.fc_2(feat)

        return feat

    def forward_seq(self, x, *args):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.act_func(x)

        if len(args) > 0:
            h_0, c_0 = args
            seq_output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            seq_output = self.fc_2(seq_output)
            return seq_output, h_n, c_n
        else:
            h_0 = self.h_0.repeat(1, x.shape[0], 1)
            c_0 = self.c_0.repeat(1, x.shape[0], 1)
            seq_output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            seq_output = self.fc_2(seq_output)
            return seq_output




# class LSTM_backbone(BaseModel):
#     def __init__(self, n_hidden, n_lstm_layer, bidirectional=False, dropout=0.2,
#                  output_type='seq', use_cell_state=True):
#         """
#         LSTM network with 2 Linear Layer. The input size should be [batch, seq_len, n_input]
#         Args:
#             n_hidden: Hidden size of lstm module.
#             n_lstm_layer: Number of lstm layer.
#             bidirectional: Use bidirectional lstm.
#             dropout: Dropout rate.
#             output_type: Choose 'seq' or 'feat'. 'seq' output will be [batch, seq_len, n_output];
#             'feat' output will be [batch, n_output]
#             use_cell_state: Using both [hidden state] and [cell state] of lstm for 'feat' computing.
#         """
#         super(EasyLSTM, self).__init__()
#
#         self.use_cell_state = use_cell_state
#
#
#         self.lstm = nn.LSTM(n_hidden, n_hidden, n_lstm_layer, batch_first=True, bidirectional=bidirectional, dropout=dropout)
#
#         self.h_0 = nn.Parameter(torch.zeros(size=[n_lstm_layer*(2 if bidirectional else 1), 1, n_hidden]))
#         self.c_0 = nn.Parameter(torch.zeros(size=[n_lstm_layer*(2 if bidirectional else 1), 1, n_hidden]))
#         self.h_0.requires_grad = False
#         self.c_0.requires_grad = False
#
#         if output_type == 'seq':
#             self.out_dim = n_hidden * (2 if bidirectional else 1)
#             self.forward = self.forward_seq
#         elif output_type == 'feat':
#             self.out_dim = n_lstm_layer * n_hidden * (2 if use_cell_state else 1) * \
#                            (2 if bidirectional else 1)
#             self.forward = self.forward_feat
#         else:
#             raise NameError("output_type should be 'seq' or 'feat'")
#
#     def forward_feat(self, x):
#
#         h_0 = self.h_0.repeat(1, x.shape[0], 1)
#         c_0 = self.c_0.repeat(1, x.shape[0], 1)
#
#         seq_output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
#
#         # h/c: [n_rnn_layer*(2 if bidirectional else 1), batch_size, n_hidden]
#         if self.use_cell_state:
#             feat = torch.cat((h_n, c_n), dim=0)
#         else:
#             feat = h_n
#         feat = feat.permute(1, 0, 2)
#         feat = feat.reshape(feat.shape[0], -1)
#
#         return feat
#
#     def forward_seq(self, x):
#
#         h_0 = self.h_0.repeat(1, x.shape[0], 1)
#         c_0 = self.c_0.repeat(1, x.shape[0], 1)
#
#         seq_output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
#
#         return seq_output

class EasyRNN(BaseModel):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer, bidirectional=False, dropout=0.2, act_func='tanh',
                 output_type='seq'):
        """
        RNN network with 2 Linear Layer. The input size should be [batch, seq_len, n_input]
        Args:
            n_input: Dim of input.
            n_output: Dim of output.
            n_hidden: Hidden size of lstm module.
            n_rnn_layer: Number of rnn layer.
            dropout: Dropout rate.
            output_type: Choose 'seq' or 'feat'. 'seq' output will be [batch, seq_len, n_output];
            'feat' output will be [batch, n_output]
        """
        super(EasyRNN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(n_input, n_hidden)
        self.act_func = activation_layer(act_name=act_func)

        self.rnn = nn.RNN(n_hidden, n_hidden, n_rnn_layer, batch_first=True, bidirectional=bidirectional)

        self.h_0 = nn.Parameter(torch.zeros(size=[n_rnn_layer*(2 if bidirectional else 1), 1, n_hidden]))
        self.h_0.requires_grad = False

        if output_type == 'seq':
            lstm_out_dim = n_hidden * (2 if bidirectional else 1)
            self.forward = self.forward_seq
        elif output_type == 'feat':
            lstm_out_dim = n_rnn_layer * n_hidden * (2 if bidirectional else 1)
            self.forward = self.forward_feat
        else:
            raise NameError("output_type should be 'seq' or 'feat'")

        self.fc_2 = nn.Linear(lstm_out_dim, n_output)

    def forward_feat(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.act_func(x)

        h_0 = self.h_0.repeat(1, x.shape[0], 1)

        seq_output, h_n = self.rnn(x, h_0)

        # h/c: [n_rnn_layer*(2 if bidirectional else 1), batch_size, n_hidden]

        feat = h_n
        feat = feat.permute(1, 0, 2)
        feat = feat.reshape(feat.shape[0], -1)

        feat = self.fc_2(feat)

        return feat

    def forward_seq(self, x, *args):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.act_func(x)

        if len(args) > 0:
            h_0 = args[0]
            seq_output, h_n = self.rnn(x, h_0)
            seq_output = self.fc_2(seq_output)
            return seq_output, h_n
        else:
            h_0 = self.h_0.repeat(1, x.shape[0], 1)
            seq_output, h_n = self.rnn(x, h_0)
            seq_output = self.fc_2(seq_output)
            return seq_output
