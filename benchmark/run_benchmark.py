import os


from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers
from benchmark.summary import DataRecorder


class Benchmarking:
    def __init__(self, cfg, agent, experiment_suite):
        # Setup the env and model
        self.config = cfg
        self.agent = agent

        # Setup the data_writer
        self.data_recorder = DataRecorder(self.config)

        # Setup carla core and experiment
        os.environ["CARLA_ROOT"] = self.config['carla_server']['carla_path']
        # Kill all servers just in case
        kill_all_servers()
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
        return observation, sensor_data, False

    def run_single_episode(self, exp_id, iteration):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        try:
            observation, sensor_data, done = self.reset()
            data_to_log = self.experiment.get_data_to_log(
                sensor_data, observation, control=None
            )
            data_to_log.update({'exp_id': exp_id, 'iteration': iteration})
            if exp_id == 0:
                self.data_recorder.create_csv_file(data_to_log)

            while not done:
                control = self.agent.compute_control(observation)
                sensor_data = self.experiment.tick(control)

                # Get the observation from the experiment
                observation = self.experiment.get_observation(sensor_data)
                data_to_log = self.experiment.get_data_to_log(
                    sensor_data, observation, control
                )

                # Post processing and logging
                data_to_log.update({'exp_id': exp_id, 'iteration': iteration})
                self.data_recorder.write(data_to_log)

                # Check if episode is done
                done = self.experiment.get_done_status(observation, self.core)

        except KeyboardInterrupt:
            kill_all_servers()

        return done

    def run(self):
        try:
            # Get all the experiment configs
            experiment_configs = self.experiment.get_experiment_configs()
            for exp_id, config in enumerate(experiment_configs):
                # for iteration in range(config['repeat']):
                self.setup_experiment(config)
                self.run_single_episode(exp_id, iteration=0)

                # Destroy actors and sensors
                client = self.core.get_client()
                self.experiment.destroy_actors_sensors(client)

            # Post processing and summarizing
        finally:
            self.data_recorder.close()
            kill_all_servers()

