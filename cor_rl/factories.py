
import gymnasium as gym
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


optimizer_factory = OptimizerFactory()


from cor_rl.atari import AtariEnvWrapper


class EnvironmentFactory:

    def __init__(self, use_gym_atari_wrapper=False):
        self.use_gym_atari_wrapper = use_gym_atari_wrapper

    def get(self, game_name, **kwargs):
        if game_name.startswith("ALE/"):
            if self.use_gym_atari_wrapper:
                env = gym.make(game_name, obs_type='rgb', frameskip=1, **kwargs)
                return gym.wrappers.AtariPreprocessing(
                    env, scale_obs=True
                )

            else:
                return AtariEnvWrapper(game_name, **kwargs)
        else:
            return gym.make(game_name, **kwargs)


environment_factory = EnvironmentFactory()
