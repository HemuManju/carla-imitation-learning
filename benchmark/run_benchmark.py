import os


from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers

from benchmark.navigation.path_planner import PathPlanner


class Benchmarking:
    def __init__(self, cfg, agent, experiment_suite):
        # Kill all servers just in case
        kill_all_servers()

        # Setup the env and model
        self.config = cfg
        self.agent = agent

        # Setup carla core and experiment
        os.environ["CARLA_ROOT"] = self.config['carla_server']['carla_path']
        self.core = CarlaCore(self.config['carla_server'])
        self.experiment = experiment_suite

    def setup_experiment(self, experiment_config):
        client = self.core.get_client()
        server_port = self.core.get_server_port()
        self.experiment.setup_experiment(client, server_port, experiment_config)
        self.experiment.spawn_hero(experiment_config["vehicle"])

    def reset(self):
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.experiment.tick(None)
        observation = self.experiment.get_observation(sensor_data)
        return observation, False

    def run_single_episode(self):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        try:
            observation, done = self.reset()
            # Weather and other setting logic

            while not done:
                control = self.agent.compute_control(observation)
                sensor_data = self.experiment.tick(control)

                # Get the observation from the experiment
                observation = self.experiment.get_observation(sensor_data)
                data_to_log = self.experiment.get_data_to_log(sensor_data, observation)

                # Post processing and logging

                # Check if episode is done
                done = self.experiment.get_done_status(observation, self.core)

        except KeyboardInterrupt:
            kill_all_servers()

        return done

    def run(self):
        try:
            # Get all the experiment configs
            experiment_configs = self.experiment.get_experiment_configs()

            for config in experiment_configs:
                self.setup_experiment(config)
                self.run_single_episode()

                # Destroy actors and sensors
                client = self.core.get_client()
                self.experiment.destroy_actors_sensors(client)
                # self.core.clean_up(self.experiment.tm_port)

            # Post processing and summarizing
        finally:
            kill_all_servers()

