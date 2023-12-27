import gymnasium as gym

from cor_rl.atari import AtariEnvWrapper


class EnvironmentFactory:

    def __init__(self, use_gym_atari_wrapper=False):
        self.use_gym_atari_wrapper = use_gym_atari_wrapper

    def get(self, game_name, **kwargs):
        if game_name.startswith("ALE/"):
            if self.use_gym_atari_wrapper:
                env = gym.make(
                    game_name, obs_type='rgb', frameskip=1,
                    **kwargs
                )
                return gym.wrappers.AtariPreprocessing(
                    env, scale_obs=True, **kwargs
                )

            else:
                return AtariEnvWrapper(game_name, **kwargs)
        else:
            return gym.make(game_name, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


factory = EnvironmentFactory()
