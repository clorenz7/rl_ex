import gymnasium as gym

from cor_rl.atari import AtariEnvWrapper, FrameStacker


class EnvironmentFactory:

    def __init__(self):
        pass

    def get(self, game_name, **kwargs):
        if game_name.startswith("ALE/"):
            return AtariEnvWrapper(game_name, **kwargs)

        elif game_name.startswith("WALE/"):
            env = gym.make(
                game_name[1:], obs_type='rgb', frameskip=1,
                **kwargs
            )
            wenv = gym.wrappers.AtariPreprocessing(
                env, scale_obs=True
            )

            return FrameStacker(wenv)
        else:
            return gym.make(game_name, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


factory = EnvironmentFactory()
