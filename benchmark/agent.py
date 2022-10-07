from __future__ import print_function
import abc


class Agent(object):
    def __init__(self):
        self.__metaclass__ = abc.ABCMeta

    def reset(self):
        pass

    @abc.abstractmethod
    def compute_control(self, observation):
        """
        Function to be redefined by an agent.
        :param The measurements like speed, the image data and a target
        :returns A carla Control object, with the steering/gas/brake for the agent
        """
        raise NotImplementedError
