import os
import random
import signal
import subprocess
import time
import psutil

import collections

try:
    import carla
except ModuleNotFoundError:
    pass

import torch
import torch.nn as nn


def train(model, data_loader, optimizer, criterion, device=None):

    # Setup train and device
    device = device or torch.device("cpu")
    model.train()

    # Metrics
    running_loss = 0.0
    epoch_steps = 0

    for data, target in data_loader:

        # get the inputs; data is a list of [inputs, labels]
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1

    return running_loss, epoch_steps


def test(model, data_loader, criterion, device=None):
    # Setup eval and device
    device = device or torch.device("cpu")
    model.eval()
    predicted = []
    targets = []
    losses = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            # Append data
            targets.append(target)
            predicted.append(outputs)
            losses.append(loss)

    return predicted, targets, losses


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def load_checkpoint(net, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # NOTE: This is a hacky way to loading the net weights
    # TODO: Find a better way to do
    target_dict = {
        k.replace('net.', '', 1): checkpoint['state_dict'][k]
        for k in checkpoint['state_dict'].keys()
        if k.startswith('net.')
    }
    net.load_state_dict(target_dict, strict=True)
    return net


def number_parameters(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    m(m.example_input_array, m.example_command)
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class CarlaServer:
    def __init__(self, config) -> None:
        BASE_CORE_CONFIG = {
            "host": "localhost",  # Client host
            "timeout": 10.0,  # Timeout of the client
            "timestep": 0.05,  # Time step of the simulation
            "retries_on_error": 10,  # Number of tries to connect to the client
            "resolution_x": 600,  # Width of the server spectator camera
            "resolution_y": 600,  # Height of the server spectator camera
            "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
            "enable_map_assets": False,  # enable / disable all town assets except for the road
            "enable_rendering": True,  # enable / disable camera images
            "show_display": False,  # Whether or not the server will be displayed
        }

        self.config = join_dicts(BASE_CORE_CONFIG, config)

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = 2000  #  random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + str(self.server_port))
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl",  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "-world-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"]),
            "-benchmark",
            # "-fps=10",
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        self.process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(
                    " Waiting for server to be ready: {}, attempt {} of {}".format(
                        e, i + 1, self.config["retries_on_error"]
                    )
                )
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration"
        )

    def start_server(self):
        """Start the server

        Returns
        -------
        str, float
            host and server port
        """
        self.init_server()
        # self.connect_client()
        return self.config["host"], self.server_port


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        """
        The inputs are sets of d-dimensional points:
        x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.
        Arguments:
            x: a float tensor with shape [b, d, n].
            y: a float tensor with shape [b, d, m].
        Returns:
            a float tensor with shape [].
        """
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # compute pairwise l2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)
