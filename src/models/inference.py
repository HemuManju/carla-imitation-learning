import numpy as np
from PIL import Image

import torch


from benchmark.benchmark_tools import run_driving_benchmark
from benchmark.driving_benchmarks import CoRL2017, CARLA100
from benchmark.benchmark_tools.experiment_suites.basic_experiment_suite import (
    BasicExperimentSuite,
)
from benchmark.benchmark_tools.agent import Agent
from benchmark.carla.client import VehicleControl


class CILCarlaAgent(Agent):
    def __init__(self, model, avoid_stopping=True, debug=False, image_cut=[115, 510]):
        Agent.__init__(self)
        self.debug = debug
        self.model = model

        # Freeze the weights and put the model in eval mode
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.freeze()
        self.model.eval()

        # Parameters
        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping
        self._image_cut = image_cut

    def run_step(self, measurements, sensor_data, directions, target):
        control = self.compute_action(
            sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed,
            directions,
        )
        return control

    def compute_action(self, rgb_image, speed, direction=None):
        rgb_image = rgb_image[self._image_cut[0] : self._image_cut[1], :]

        image_input = np.array(
            Image.fromarray(rgb_image).resize(
                [self._image_size[0], self._image_size[1]]
            )
        )

        image_input = image_input.astype(np.float32)
        image_input = np.expand_dims(np.transpose(image_input, (2, 0, 1)), axis=0)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        speed = np.array([[speed]]).astype(np.float32) / 30.0
        direction = int(direction - 2)

        steer, acc, brake = self._control_function(image_input, speed, direction)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc = acc * 0.4

        control = VehicleControl()
        control.steer = steer
        control.throttle = acc
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0
        return control

    def _control_function(self, image_input, speed, control_input):

        img_ts = torch.from_numpy(image_input).cuda()
        speed_ts = torch.from_numpy(speed).cuda()

        with torch.no_grad():
            branches, pred_speed = self.model(img_ts, speed_ts)

        pred_result = (
            branches[0][3 * control_input : 3 * (control_input + 1)].cpu().numpy()
        )

        predicted_steers = pred_result[0]
        predicted_acc = pred_result[1]
        predicted_brake = pred_result[2]

        if self._avoid_stopping:
            predicted_speed = pred_speed.squeeze().item()
            real_speed = speed * 30.0

            real_predicted = predicted_speed * 30.0
            if real_speed < 2.0 and real_predicted > 3.0:

                predicted_acc = 1 * (5.6 / 30.0 - speed) + predicted_acc
                predicted_brake = 0.0
                predicted_acc = predicted_acc * 0.5

        return predicted_steers, predicted_acc, predicted_brake


def run_benchmark(agent, benchmark_config):
    # Experiment suit
    if benchmark_config['benchmark'] == 'corl_2017':
        experiment_suite = CoRL2017(benchmark_config['city_name'])
    elif benchmark_config['benchmark'] == 'carla100':
        experiment_suite = CARLA100(benchmark_config['city_name'])
    else:
        print('WARNING: running the basic driving benchmark')
        experiment_suite = BasicExperimentSuite(benchmark_config['city_name'])

    run_driving_benchmark(
        agent,
        experiment_suite,
        benchmark_config['city_name'],
        benchmark_config['log_name'],
        benchmark_config['continue_experiment'],
        benchmark_config['host'],
        benchmark_config['port'],
    )

