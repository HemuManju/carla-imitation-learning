import copy

import math
from collections import deque
import pandas as pd
import numpy as np


from PIL import Image

import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from benchmark.basic_experiment import BasicExperiment
from benchmark.navigation.path_planner import PathPlanner
from benchmark.navigation.utils import distance_vehicle, get_acceleration, get_speed

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
        self.image_deque = deque(maxlen=self.cfg['seq_length'])

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
        base_config['navigation_type'] = navigation_type

        return base_config

    def get_experiment_configs(self):
        all_configs = []
        for navigation_type in self.cfg['navigation_types']:
            for weather in self.cfg['weathers']:
                for town in self.cfg['towns']:
                    config = self._construct_experiment_config(
                        copy.deepcopy(self.cfg), weather, town, navigation_type
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
        self.n_collision = 0
        self.n_lane_invasion = 0
        self.distance_to_destination = 2000
        self.image_deque.clear()

        # Set the planner
        self.route_planner = PathPlanner(
            self.hero, target_speed=self.cfg['vehicle']['target_speed'],
        )
        self.route_planner.set_destination(self.end_point.location)
        return None

    def get_distance_to_destination(self):
        distance = distance_vehicle(
            self.route_planner._local_planner._waypoints_queue[-1][0],
            self.hero.get_transform(),
        )
        return distance

    def process_sensor_data(self, sensor_data):
        data = {}

        data['collision_predistrain'] = 0
        data['collision_vehicle'] = 0
        data['collision_other'] = 0

        # Get collision information
        if 'collision' in sensor_data.keys():
            self.n_collision += 1

            actor = sensor_data['collision'][0]
            if actor.semantic_tags == 4:
                data['collision_predistrain'] = 1
            elif actor.semantic_tags == 10:
                data['collision_vehicle'] = 1
            else:
                data['collision_other'] = 1

        # Get lane invasion information
        if 'lane_invasion' in sensor_data.keys():
            data['lane_invasion'] = 1
        else:
            data['lane_invasion'] = 0
        return data

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """

        observation = {}
        image = sensor_data['rgb']
        image_tensor = transforms.ToTensor()(image.copy())

        if not self.image_deque:
            for i in range(self.cfg['seq_length']):
                self.image_deque.append(image_tensor)
        else:
            self.image_deque.append(image_tensor)

        observation['image'] = torch.stack(list(self.image_deque), dim=0)
        command = self.route_planner.get_next_command(debug=False)
        observation['command'] = command['modified_direction'].value

        # Add speed to observation
        observation['speed'] = get_speed(self.hero)
        observation['steer'] = self.hero.get_control().steer

        # Get moving direction
        vehicle_transform = self.hero.get_transform()
        v_vec = vehicle_transform.get_forward_vector()
        observation['moving_direction'] = [v_vec.x, v_vec.y, 0.0]

        # Get location
        location = self.hero.get_location()
        observation['location'] = [location.x, location.y, location.z]

        # Get speed
        observation['current_speed'] = get_speed(self.hero)

        return observation

    def get_data_to_log(self, sensor_data, observation, control):
        data = {}
        data['command'] = observation['command']
        data['start_points'] = [
            self.start_point.location.x,
            self.start_point.location.y,
        ]
        data['end_points'] = [self.end_point.location.x, self.end_point.location.y]
        data['reached_destination'] = (
            self.distance_to_destination < 2.5
        ) or self.route_planner.done()
        data['done_time_episode'] = self.done_time_episode
        data['n_collisions'] = self.n_collision
        data['idle_time'] = self.time_idle

        # Add planner number of points
        data['len_path_points'] = self.route_planner.get_n_remaining_path_points()

        # Vehicle parameters
        location = self.hero.get_location()
        data['pos_x'] = location.x
        data['pos_y'] = location.y
        if control is None:
            data['steer'] = 0
            data['throttle'] = 0
            data['brake'] = 0
        else:
            data['steer'] = control.steer
            data['throttle'] = control.throttle
            data['brake'] = control.brake
        data['acceleration'] = get_acceleration(self.hero)
        data['speed'] = get_speed(self.hero)

        # Sensor parameters
        sensor_summary = self.process_sensor_data(sensor_data)
        data.update(sensor_summary)

        return data

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

        # Distance to final waypoint
        self.distance_to_destination = self.get_distance_to_destination()

        return (
            self.done_time_idle
            or self.done_falling
            or self.done_time_episode
            or self.route_planner.done()
            or (self.distance_to_destination < 2.5)
            or (self.n_collision > 200)
        )
