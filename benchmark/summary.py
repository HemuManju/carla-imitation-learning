import os
import csv


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
    print(filename, file_extension)
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
    def __init__(self, config, directory='run', write_path=None) -> None:
        self.cfg = config
        self.write_path = write_path

        if self.write_path is None:
            self.write_path = get_nonexistant_path(
                f'logs/benchmark_results/{directory}'
            )

        # Create a directory
        create_directory(self.write_path)

    def create_csv_file(self, init_data, file_name='measurements'):
        # Create a folder
        path_to_file = self.write_path + f'/{file_name}.csv'
        if not os.path.isfile(path_to_file):
            self.csvfile = open(path_to_file, 'a', newline='')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=init_data.keys())
            self.writer.writeheader()

    def write(self, data):
        self.writer.writerow(data)

    def close(self):
        self.csvfile.close()
