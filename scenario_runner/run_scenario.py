import ray

import matplotlib.pyplot as plt

from .core.carla_env import RoutedCarlaEnv

from .core.carla_core import kill_all_servers
from .core.helper import get_checkpoint

from src.visualization.visualize import plot_frames


class RunScenario:
    def __init__(self, cfg, restore_config):
        # Setup the env and model
        self.config = cfg

        # TODO: Add pytorch lightning model restore logic
        checkpoint_path = restore_config['checkpoint_path']
        self.model = cfg['model'].load_from_checkpoint(
            checkpoint_path, hparams=cfg, net=cfg['net'], data_loader=None
        )

        # Kill all servers just in case
        kill_all_servers()

    def run(self):
        try:
            self.env = RoutedCarlaEnv(self.config['env_config'])
            obs = self.env.reset()
            done = False
            plot = False

            fig, ax = plt.subplots(nrows=1, ncols=obs.shape[0])

            for i in range(5000):
                action = self.agent(obs)
                obs, reward, done, info = self.env.step(action)

                if plot:
                    plot_frames(ax, obs)
                if done:
                    obs = self.env.reset()

        except KeyboardInterrupt:
            kill_all_servers()

        finally:
            kill_all_servers()
