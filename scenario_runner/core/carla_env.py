#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import gym

from .carla_core import CarlaCore

from ..navigation.path_planner import PathPlanner


class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """

    def __init__(self, config):
        """Initializes the environment"""
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data, core=self.core)

        return observation

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data, core=self.core)
        done = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, done, info


class RoutedCarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """

    def __init__(self, config):
        """Initializes the environment"""
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()
        self.route_planner = PathPlanner(
            self.hero,
            target_speed=self.config['experiment']['constraints']['desired_speed'],
        )

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data)
        waypoint, direction = self.route_planner.get_next_command()

        # Filter our void, change left and right lane commands
        if direction.value in [-1, 5, 6]:
            observation['command'] = 4
        else:
            observation['command'] = direction.value

        return observation

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)

        # Get the observation and net command
        observation, info = self.experiment.get_observation(sensor_data)
        waypoint, direction = self.route_planner.get_next_command()

        # Filter our void, change left and right lane commands
        if direction.value in [-1, 5, 6]:
            observation['command'] = 4
        else:
            observation['command'] = direction.value

        done = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, done, info
