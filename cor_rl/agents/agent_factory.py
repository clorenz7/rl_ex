from . import a2c
from . import atari_a2c


class AgentFactory:
    _AGENTS = {
        # TODO: Add these
        # 'deepq': q_agents.FFWQAgent,
        # 'qtiles': q_agents.TiledLinearQAgent,
        # 'actor-critic': ac_agents.MountainCarActorCriticAgent,
        'a2c-ffw': a2c.AdvantageActorCriticAgent,
        'a2c-atari': atari_a2c.Mnih2016A2CAgent,
    }

    def get(self, agent_params, train_params, device="cpu"):
        agent_type = agent_params['type']
        constructor = self._AGENTS[agent_type]
        agent = constructor(
            agent_params, train_params, device=device
        )
        load_file = agent_params.get('load_checkpoint', None)
        if load_file:
            agent.load(load_file)
        return agent

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)
