from torch import nn


def activation_layer(act_name) -> nn.Module:
    """
    Get activation layer by name.

    Args:
        act_name: Name of activation.
    Return:
        act_layer: Activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(negative_slope=1e-2)
        elif act_name.lower() == 'elu':
            act_layer = nn.ELU()
        elif act_name.lower() == 'tanh':
            act_layer = nn.Tanh()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer
