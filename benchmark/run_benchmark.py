import os
import csv
import gc

import pandas as pd

from benchmark.core.carla_core import kill_all_servers


def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    return new_fname


def create_directory(write_path):
    if not os.path.exists(write_path):

        # Create a new directory because it does not exist
        os.makedirs(write_path)
        print("Created new data directory!")


class DataRecorder:
    def __init__(self, summary_config) -> None:
        self.cfg = summary_config
        self.write_path = self.cfg['write_path']

        if self.cfg['directory'] is None:
            directory = 'run'
        else:
            directory = self.cfg['directory']

        if self.write_path is None:
            self.write_path = get_nonexistant_path(
                f'logs/benchmark_results/{directory}'
            )
        # Create a directory
        create_directory(self.write_path)
        self.all_data = []

    def is_csv_empty(self, path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for i, _ in enumerate(reader):
                if i:  # found the second row
                    return False
        return True

    def create_csv_file(self, init_data):
        if self.cfg['file_name'] is None:
            file_name = 'measurements'

        # Create a folder
        self.path_to_file = self.write_path + f'/{file_name}.csv'
        if not os.path.isfile(self.path_to_file):
            self.csvfile = open(self.path_to_file, 'a', newline='')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=init_data.keys())
            self.writer.writeheader()

    def log_data(self, data):
        self.all_data.append(data)

    def write(self):
        df = pd.DataFrame(self.all_data)
        df.to_csv(
            self.path_to_file,
            mode='a',
            index=False,
            header=self.is_csv_empty(self.path_to_file),
        )
        del self.all_data
        gc.collect()
        self.all_data = []

    def close(self):
        self.csvfile.close()


class Benchmarking:
    def __init__(self, core, cfg, agent, experiment_suite):
        # Setup the env and model
        self.config = cfg
        self.agent = agent

        # Kill all servers just in case
        self.core = core
        self.experiment = experiment_suite

    def setup_experiment(self, experiment_config):
        client = self.core.get_client()
        server_port = self.core.get_server_port()
        self.experiment.setup_experiment(client, server_port, experiment_config)
        self.experiment.spawn_hero(experiment_config["vehicle"])

    def reset(self):
        self.experiment.reset()

        self.agent.reset()

        # Tick once and get the observations
        sensor_data = self.experiment.tick(None)
        observation = self.experiment.get_observation(sensor_data)
        return observation, sensor_data, False

    def run_single_episode(self, exp_id, iteration, config):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        # Setup the experiment
        self.setup_experiment(config)

        try:
            observation, sensor_data, done = self.reset()
            data_to_log = self.experiment.get_data_to_log(
                sensor_data, observation, control=None
            )
            data_to_log.update(
                {
                    'exp_id': exp_id,
                    'iteration': iteration,
                    'navigation_type': config['navigation_type'],
                    'weather': config['weather'],
                    'town': config['town'],
                }
            )

            # Create csv file
            self.data_recorder.create_csv_file(data_to_log)

            while not done:
                control = self.agent.compute_control(observation)
                sensor_data = self.experiment.tick(control)

                # Get the observation from the experiment
                observation = self.experiment.get_observation(sensor_data)

                # Check if episode is done
                done = self.experiment.get_done_status(observation, self.core)

                data_to_log = self.experiment.get_data_to_log(
                    sensor_data, observation, control
                )

                # Post processing and logging
                data_to_log.update(
                    {
                        'exp_id': exp_id,
                        'iteration': iteration,
                        'navigation_type': config['navigation_type'],
                        'weather': config['weather'],
                        'town': config['town'],
                    }
                )
                self.data_recorder.log_data(data_to_log)

            # Write to CSV
            self.data_recorder.write()

        except KeyboardInterrupt:
            kill_all_servers()

        return done

    def run(self, experiment_config, exp_id=0):
        try:
            # Setup the data_writer
            self.data_recorder = DataRecorder(experiment_config['summary_writer'])

            # Run the simulations
            for i in range(len(experiment_config['vehicle']['path_points'])):
                self.run_single_episode(exp_id, iteration=i, config=experiment_config)

                # Destroy actors and sensors
                client = self.core.get_client()
                self.experiment.destroy_actors_sensors(client)

        finally:
            # Close the recorder
            self.data_recorder.close()
