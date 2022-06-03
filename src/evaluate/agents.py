import numpy as np
from PIL import Image

import torch
from torchvision import transforms

try:
    import carla
except ModuleNotFoundError:
    pass

from benchmark.agent import Agent

from src.dataset.utils import get_preprocessing_pipeline
import matplotlib.pyplot as plt


class CustomCILAgent(Agent):
    def __init__(self, model, config, avoid_stopping=True, debug=False) -> None:
        super().__init__()
        Agent.__init__(self)
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
        self.time_since_brake = 0
        self.preprocess = get_preprocessing_pipeline(config)

    def compute_control(self, observation):
        image_input = torch.swapaxes(self.preprocess(observation['image']), 1, 0)

        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        acc, steer, brake = self._control_function(image_input, command)
        if command == 3:
            steer = 1

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # Speed limit to 35 km/h
        if observation['speed'] > 35.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer - 1) > 0.15:
            acc_scaling_factor = 0.50
        else:
            acc_scaling_factor = 0.75

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
        control.steer = steer - 1
        control.throttle = acc * acc_scaling_factor
        control.brake = brake * brake_scaling_factor
        control.hand_brake = 0
        control.reverse = 0
        return control

    def _control_function(self, image_input, command_input):
        with torch.no_grad():
            actions = (
                self.model(
                    image_input.cuda(), torch.tensor(command_input).unsqueeze(0).cuda()
                )
                .cpu()
                .numpy()
            )[0].tolist()

        predicted_acc = actions[0]
        predicted_steers = actions[1]
        predicted_brake = actions[2]

        return predicted_acc, predicted_steers, predicted_brake
