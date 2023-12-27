
import torch
from torch import nn


class ActivationFactory:
    _MAP = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'elu': nn.ELU,
        'hardtanh': nn.Hardtanh,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'hardsigmoid': nn.Hardsigmoid,
    }

    def get(self, activation_name):
        return self._MAP[activation_name.lower()]()

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


activation_factory = ActivationFactory()


def ffw_factory(layer_sizes, activation=nn.ELU, final_activation=None):

    layers = []
    for ii in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1]))
        if isinstance(activation, str):
            layers.append(activation_factory.get(activation))
        else:
            layers.append(activation())

    # Remove final activation and re-add it
    layers.pop()
    if final_activation is not None:
        if isinstance(final_activation, str):
            layers.append(activation_factory.get(final_activation))
        else:
            layers.append(final_activation())

    return nn.Sequential(*layers)


class OptimizerFactory:
    _MAP = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "SGD": torch.optim.SGD,
    }

    def get(self, optimizer_name, optimizer_params=None, net=None, param_list=None):
        optimizer_params = optimizer_params or {}
        if not optimizer_name:
            optimizer_name = "sgd"
        constructor = self._MAP.get(optimizer_name.lower(), None)
        if constructor is None:
            constructor = getattr(torch.optim, optimizer_name)

        if net is not None:
            optimizer = constructor(
                net.parameters(),
                **optimizer_params
            )
        elif param_list is not None:
            optimizer = constructor(
                param_list,
                **optimizer_params
            )
        else:
            raise ValueError("Either net or param_list must be not None!")

        return optimizer

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


optimizer_factory = OptimizerFactory()
