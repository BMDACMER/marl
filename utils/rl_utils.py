from torch import nn


def soft_update(args, target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


def hard_update(target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


activate_funs = {'tanh': nn.Tanh(),
                 'relu': nn.ReLU()}
