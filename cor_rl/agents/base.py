import torch


class BaseAgent:

    def __init__(self, agent_params):
        self.agent_params = agent_params or {}
        self.gamma = self.agent_params.get('gamma', 0.9)
        self.epsilon = self.agent_params.get('epsilon', 1e-8)
        self.min_vals = self.agent_params.get('min_vals', None)
        self.max_vals = self.agent_params.get('max_vals', None)
        self.n_actions = agent_params.get('n_actions')
        self.n_state = agent_params.get('n_state')
        self.grad_clip = self.agent_params.get('grad_clip', 1.0) or 0.0
        self.reward_clip = self.agent_params.get('reward_clip')
        self.entropy_weight = self.agent_params.get('entropy_weight', 0.01)
        if self.min_vals and self.max_vals:
            self.mu = (
                torch.tensor(self.max_vals) + torch.tensor(self.min_vals)
            ) / 2.0
            self.sigma = (
                torch.tensor(self.max_vals) - torch.tensor(self.min_vals)
            ) / 2.0
        else:
            self.mu = self.sigma = None

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

    def normalize_state(self, state):
        features = torch.from_numpy(state).float()
        if self.mu:
            features = (features - self.mu)/self.sigma

        return features.to(self.device)