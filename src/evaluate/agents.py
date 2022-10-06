from collections import deque

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

try:
    import carla
except ModuleNotFoundError:
    pass

from benchmark.agent import Agent

from src.dataset.preprocessing import get_preprocessing_pipeline
import matplotlib.pyplot as plt

from .controller import VehiclePIDController, PIController


class BaseAgent(Agent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__()
        self.debug = debug
        self.model = model
        self.config = config

        # Freeze the weights and put the model in eval mode
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.freeze()
        self.model.eval()

        # Preprocessing
        self._avoid_stopping = avoid_stopping
        self.preprocess = get_preprocessing_pipeline(config)

    def _control_function(self, image_input, command_input):
        with torch.no_grad():
            actions = (
                self.model(
                    image_input.cuda(), torch.tensor(command_input).unsqueeze(0).cuda()
                )
                .cpu()
                .numpy()
            )[0].tolist()
        return actions


class PIThetaNeaFarAgent(BaseAgent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__(model, config, avoid_stopping, debug)
        self.time_since_brake = 0
        self.pi_control = PIController()

    def post_process_action(self, acc, steer, brake, speed):

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > 0:
            brake = 0.0

        # Speed limit to 35 km/h
        if speed > 20.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc_scaling_factor = 0.75
        else:
            acc_scaling_factor = 1

        if self._avoid_stopping:
            # If time since bake is less than 50
            if brake > 0.45 and self.time_since_brake < 50:
                brake_scaling_factor = 1
            else:
                brake_scaling_factor = 0.0

            self.time_since_brake += 1
            if self.time_since_brake > 100:
                self.time_since_brake = 0

        # Carla vehicle control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = acc * acc_scaling_factor
        control.brake = 0 * brake * brake_scaling_factor
        control.hand_brake = 0
        control.reverse = 0
        return control

    def compute_control(self, observation):
        # Crop the image
        if self.config['crop']:
            crop_size = 256 - (2 * self.config['image_resize'][1])
            image = observation['image'][:, :, crop_size:, :]
        else:
            image = observation['image']

        image_input = torch.swapaxes(self.preprocess(image), 1, 0)

        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        actions = self._control_function(image_input, command)
        theta_near, theta_far, acc, steer, brake = (
            actions[0],
            actions[1],
            actions[2],
            actions[3],
            actions[4],
        )

        control = self.pi_control.run_step(theta_near, theta_far, acc, steer)
        if command == 3:
            control.steer = 0

        return control


class CILAgent(BaseAgent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__(model, config, avoid_stopping, debug)
        self.time_since_brake = 0

    def post_process_action(self, acc, steer, brake, speed):

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > 0:
            brake = 0.0

        # Speed limit to 35 km/h
        if speed > 20.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc_scaling_factor = 0.75
        else:
            acc_scaling_factor = 1

        if self._avoid_stopping:
            # If time since bake is less than 50
            if brake > 0.45 and self.time_since_brake < 50:
                brake_scaling_factor = 1
            else:
                brake_scaling_factor = 0.0

            self.time_since_brake += 1
            if self.time_since_brake > 100:
                self.time_since_brake = 0

        # Carla vehicle control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = acc * acc_scaling_factor
        control.brake = 0 * brake * brake_scaling_factor
        control.hand_brake = 0
        control.reverse = 0
        return control

    def compute_control(self, observation):
        # Crop the image
        if self.config['crop']:
            crop_size = 256 - (2 * self.config['image_resize'][1])
            image = observation['image'][:, :, crop_size:, :]
        else:
            image = observation['image']

        image_input = torch.swapaxes(self.preprocess(image), 1, 0)

        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        actions = self._control_function(image_input, command)
        acc, steer, brake = actions[0], actions[1], actions[2]

        if command == 3:
            steer = 2

        control = self.post_process_action(
            acc, (steer / 2) - 1, brake, observation['speed']
        )
        return control


class SteerAccCILAgent(BaseAgent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__(model, config, avoid_stopping, debug)
        self.past_steering = 0
        self.max_throttle = 0.50
        self.max_brake = 0.3
        self.max_steering = 0.8

    def post_process_action(self, acceleration, steering):

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if steering > self.past_steering + 0.1:
            steering = self.past_steering + 0.1
        elif steering < self.past_steering - 0.1:
            steering = self.past_steering - 0.1

        if steering >= 0:
            steering = min(self.max_steering, steering)
        else:
            steering = max(-self.max_steering, steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control

    def compute_control(self, observation):
        # Crop the image
        crop_size = 256 - (2 * self.config['image_resize'][1])
        image = observation['image'][:, :, crop_size:, :]

        image_input = torch.swapaxes(self.preprocess(image), 1, 0)

        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        actions = self._control_function(image_input, command)
        acc, steer = actions[0], actions[1]

        if command == 3:
            steer = 2

        control = self.post_process_action(acc, (steer / 2) - 1)

        return control


class PIDCILAgent(BaseAgent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__()

        # PID controller
        self.pid_controller = VehiclePIDController()

    def compute_control(self, observation):
        # Crop the image
        crop_size = 256 - (2 * self.config['image_resize'][1])
        image = observation['image'][:, :, crop_size:, :]

        image_input = torch.swapaxes(self.preprocess(image), 1, 0)

        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        actions = self._control_function(image_input, command)
        target_speed, steer = actions[0], actions[1]

        if command in [3] and command not in [1, 2]:
            steer = 0
        else:
            steer = observation['steer'] - 1

        control = self.pid_controller.run_step(
            target_speed=target_speed,
            current_steering=steer,
            current_speed=observation['speed'],
        )

        return control
