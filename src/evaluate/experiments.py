import math
import pandas as pd

from benchmark.basic_experiment import BasicExperiment
from benchmark.navigation.path_planner import PathPlanner


try:
    import carla
except ModuleNotFoundError:
    pass


def read_txt_files(read_path):
    values = pd.read_csv(
        read_path, sep=" ", header=None, index_col=False,
    ).values.tolist()
    return values


class CORL2017(BasicExperiment):
    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.cfg = experiment_config
        self.max_time_idle = self.cfg["others"]["max_time_idle"]
        self.max_time_episode = self.cfg["others"]["max_time_episode"]

    def _construct_experiment_config(self, base_config, weather, town, navigation_type):
        # Update the spawn points
        data = pd.read_xml(
            f'benchmark/corl2017/{town}_{navigation_type}.xml', xpath=".//waypoint"
        ).values.tolist()
        path_points = []
        for i in range(0, len(data), 2):
            path_points.append([data[i], data[i + 1]])

        base_config['vehicle']['path_points'] = path_points
        base_config['town'] = town
        base_config['weather'] = weather

        return base_config

    def get_experiment_configs(self):
        all_configs = []
        for navigation_type in self.cfg['navigation_types']:
            for weather in self.cfg['weathers']:
                for town in self.cfg['towns']:
                    config = self._construct_experiment_config(
                        self.cfg.copy(), weather, town, navigation_type
                    )
                    all_configs.append(config)

        return all_configs

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Set episode parameters
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False
        # Set the planner
        self.route_planner = PathPlanner(
            self.hero, target_speed=self.cfg['vehicle']['target_speed'],
        )
        self.route_planner.set_destination(self.end_point.location)
        return None

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        observation = {}
        observation['image'] = sensor_data['rgb']
        command = self.route_planner.get_next_command()
        observation['command'] = command['direction'].value

        return observation

    def get_data_to_log(self, sensor_data, observation):
        #

        return None

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(self.hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = self.hero.get_location().z < -0.5
        return (
            self.done_time_idle
            or self.done_falling
            or self.done_time_episode
            or self.route_planner.done()
        )
