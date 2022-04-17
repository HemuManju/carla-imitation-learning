import math
import numpy as np
from gym.spaces import Box, Discrete, Dict

try:
    import carla
except ModuleNotFoundError:
    pass

from .base_experiment import ContinuousBase
from .core.helper import post_process_image


class FrontRGBContinuous(ContinuousBase):
    def get_observation_space(self):
        if self.config["hero"]["sensors_process"]["gray_scale"]:
            num_of_channels = 1
        else:
            num_of_channels = 3
        obs_space = Dict(
            {
                "images": Box(
                    low=0.0,
                    high=255.0,
                    shape=(
                        num_of_channels * self.frame_stack,
                        self.config["hero"]["sensors"]["rgb"]["image_size_x"],
                        self.config["hero"]["sensors"]["rgb"]["image_size_y"],
                    ),
                    dtype=np.float32,
                ),
                "command": Discrete(1),
            }
        )
        return obs_space

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5

        # Collision
        if self.n_collision > 3:
            self.collided = True

        return (
            self.done_time_idle
            or self.done_falling
            or self.done_time_episode
            or self.collided
        )

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False
        self.collided = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        # Collision and heading deviation
        self.n_collision = 0
        self.last_heading_deviation = 0

    def preprocess_sensor_data(self, sensor_data):

        # Check if there is a collision
        if "collision" in sensor_data.keys():
            self.n_collision += 1

    def get_observation(self, sensor_data, core=None):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """

        # Preprocess sensor data
        self.preprocess_sensor_data(sensor_data)

        image = post_process_image(
            sensor_data["rgb"][1],
            normalized=self.config["hero"]["sensors_process"]["normalized"],
            grayscale=self.config["hero"]["sensors_process"]["gray_scale"],
        )

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        images = np.swapaxes(images, 0, 2)
        self.obs["images"] = images

        return self.obs, {}

    def compute_reward(self, observation, core):
        reward = 0

        return reward
