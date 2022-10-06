# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np

try:
    import carla
except ModuleNotFoundError:
    pass
from benchmark.navigation.utils import get_speed


class PIController:
    def __init__(
        self, max_throttle=0.50, max_brake=0.3, max_steering=0.8,
    ):
        self._dt = 1.0 / 20.0
        self.controller_params = {
            'K_P_far': 0.5249113,
            'K_P_near': -0.53373547,
            'K_I_near': 0.04510265,
            'dt': self._dt,
        }

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0
        self.theta_near_history = deque([0, 0, 0, 0], maxlen=4)

    def run_step(self, theta_near, theta_far, acceleration, steer):
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Calculate the steering
        self.theta_near_history.append(theta_near)
        steering = (
            self.controller_params['K_P_far'] * theta_far
            + self.controller_params['K_P_near'] * theta_near
            + self.controller_params['K_I_near']
            * self._dt
            * sum(self.theta_near_history)
        )
        print(steering)
        current_steering = steering
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        return control


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(
        self,
        args_longitudinal=None,
        offset=0,
        max_throttle=0.75,
        max_brake=0.3,
        max_steering=0.8,
    ):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        if args_longitudinal is None:
            self._dt = 1.0 / 20.0
            self._target_speed = 20.0  # Km/h
            self._sampling_radius = 2.0
            self._args_longitudinal = {
                'K_P': 1.0,
                'K_I': 0.05,
                'K_D': 0,
                'dt': self._dt,
            }
        else:
            self._args_longitudinal = args_longitudinal

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0
        self._lon_controller = PIDLongitudinalController(**self._args_longitudinal)

    def run_step(self, target_speed, current_steering, current_speed):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        # Acceleration is calculated using the logitudinal controller
        acceleration = self._lon_controller.run_step(target_speed, current_speed)

        print(target_speed, current_speed)
        print(acceleration, self.max_throt)
        print('--------------------')

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control

    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_lateral)


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
