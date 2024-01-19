from . import a2c
from . import atari_a2c
from .base import RepeatAgent


from torch import nn


class AgentFactory:
    _AGENTS = {
        # TODO: Add these
        # 'deepq': q_agents.FFWQAgent,
        # 'qtiles': q_agents.TiledLinearQAgent,
        # 'actor-critic': ac_agents.MountainCarActorCriticAgent,
        'a2c-ffw': a2c.AdvantageActorCriticAgent,
        'a2c-atari': atari_a2c.Mnih2016A2CAgent,
        'a2c-lstm': atari_a2c.Mnih2016LSTMA2CAgent,
        'a2c-lstmk': atari_a2c.KostikovLSTMA2CAgent,
        'repeater': RepeatAgent,
    }

    def get(self, agent_params, train_params, device="cpu") -> nn.Module:
        agent_type = agent_params['type']
        constructor = self._AGENTS[agent_type]
        agent = constructor(
            agent_params, train_params, device=device
        )
        load_file = agent_params.get('load_checkpoint', None)
        if load_file:
            agent.load(load_file)
        return agent

    def __call__(self, *args, **kwargs) -> nn.Module:
        return self.get(*args, **kwargs)
