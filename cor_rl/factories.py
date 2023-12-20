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
