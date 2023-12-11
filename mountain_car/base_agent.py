
import torch


class BaseAgent:

    def get_grid(self, n_mesh=100):
        x = torch.linspace(
            self.min_vals[0] + 1e-3,
            self.max_vals[0] - 1e-3,
            n_mesh
        )
        y = torch.linspace(
            self.min_vals[1] + 1e-3,
            self.max_vals[1] - 1e-3,
            n_mesh
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid_xf, grid_yf = grid_x.flatten(), grid_y.flatten()
        features = []
        for state in zip(grid_xf, grid_yf):
            features.append(self.state_to_features(state))

        features = torch.vstack(features).to(self.device)

        return features, grid_x.numpy(), grid_y.numpy()

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

