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

        # Freeze the weights and put the model in eval mode
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.freeze()
        self.model.eval()

        # Preprocessing
        self._avoid_stopping = avoid_stopping
        self.preprocess = get_preprocessing_pipeline(config)

    def compute_control(self, observation):
        image = Image.fromarray(observation['image'])
        convert = transforms.Compose(
            [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
        )
        image_input = self.preprocess(convert(image)).unsqueeze(0)
        # plt.imshow(image_input[0, :, :, :].permute(1, 2, 0))
        # plt.show()
        # print(image_input.shape)
        # afaf
        if observation['command'] in [-1, 5, 6]:
            command = 4
        else:
            command = observation['command']

        # Get the control
        acc, steer, brake = self._control_function(image_input, command)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        # if speed > 10.0 and brake == 0.0:
        #     acc = 0.0

        # Carla vehicle control
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = acc * 0.75
        control.brake = brake
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
