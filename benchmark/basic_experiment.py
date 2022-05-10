#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import random
import sys
import time
import psutil
import logging


try:
    import carla
except ModuleNotFoundError:
    pass

from .core.sensors.sensor_interface import SensorInterface
from .core.sensors.factory import SensorFactory
from .core.helper import join_dicts

BASE_EXPERIMENT_CONFIG = {
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
    "hero": {
        "blueprint": "vehicle.lincoln.mkz_2017",
        "sensors": {  # Go to sensors/factory.py to check all the available sensors
            # "sensor_name1": {
            #     "type": blueprint,
            #     "attribute1": attribute_value1,
            #     "attribute2": attribute_value2
            # }
            # "sensor_name2": {
            #     "type": blueprint,
            #     "attribute_name1": attribute_value1,
            #     "attribute_name2": attribute_value2
            # }
        },
        "spawn_points": [
            # "0,0,0,0,0,0",  # x,y,z,roll,pitch,yaw
        ],
    },
    "background_activity": {
        "n_vehicles": 0,
        "n_walkers": 0,
        "tm_hybrid_mode": True,
        "seed": None,
    },
    "town": "Town05_Opt",
    "weather": 'ClearNoon',
}


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


class BasicExperiment(object):
    def __init__(self, config):
        self.config = join_dicts(BASE_EXPERIMENT_CONFIG, config)
        self.world = None
        self.hero = None
        self.traffic_manager = None
        self.sensor_interface = SensorInterface()

    def setup_experiment(self, client, server_port, experiment_config):
        """Initialize the hero and sensors"""

        self.world = client.load_world(
            map_name=experiment_config["town"],
            reset_settings=False,
            map_layers=carla.MapLayer.All
            if self.config["enable_map_assets"]
            else carla.MapLayer.NONE,
        )

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, experiment_config["weather"])
        self.world.set_weather(weather)

        self.tm_port = server_port // 10 + server_port % 10
        while is_used(self.tm_port):
            print(
                "Traffic manager's port "
                + str(self.tm_port)
                + " is already being used. Checking the next one"
            )
            self.tm_port += 1

        if self.traffic_manager is None:
            self.traffic_manager = client.get_trafficmanager(self.tm_port)
            print("Traffic manager connected to port " + str(self.tm_port))

        self.traffic_manager.set_hybrid_physics_mode(
            experiment_config["background_activity"]["tm_hybrid_mode"]
        )
        seed = experiment_config["background_activity"]["seed"]
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

        # Spawn the background activity
        self.spawn_npcs(
            client,
            experiment_config["background_activity"]["n_vehicles"],
            experiment_config["background_activity"]["n_walkers"],
        )

    def spawn_hero(self, hero_config):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface.destroy()
        self.world.tick()

        self.hero_blueprints = self.world.get_blueprint_library().find(
            hero_config['blueprint']
        )
        self.hero_blueprints.set_attribute("role_name", "hero")
        random.shuffle(hero_config['path_points'], random.random)
        for points in hero_config['path_points']:
            # If already spawned, destroy it
            try:
                # Get the start and end points of the
                self.start_point = carla.Transform(
                    carla.Location(points[0][0], points[0][1], points[0][2]),
                    carla.Rotation(points[0][4], points[0][5], points[0][3]),
                )
                self.end_point = carla.Transform(
                    carla.Location(points[1][0], points[1][1], points[1][2]),
                    carla.Rotation(points[1][4], points[1][5], points[1][3]),
                )
                self.hero = self.world.try_spawn_actor(
                    self.hero_blueprints, self.start_point
                )
                if self.hero is not None:
                    print("Hero spawned!")
                    break
                else:
                    print("Could not spawn hero, changing spawn point")
            except IndexError:
                pass

        if self.hero is None:
            print("We ran out of spawn points")
            return

        self.world.tick()

        # Part 3: Spawn the new sensors
        for name, attributes in hero_config["sensors"].items():
            _ = SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)

        return self.hero

    def spawn_npcs(self, client, n_vehicles, n_walkers):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > n_spawn_points:
            logging.warning(
                "{} vehicles were requested, but there were only {} available spawn points".format(
                    n_vehicles, n_spawn_points
                )
            )
            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            v_blueprint = random.choice(v_blueprints)
            if v_blueprint.has_attribute('color'):
                color = random.choice(
                    v_blueprint.get_attribute('color').recommended_values
                )
                v_blueprint.set_attribute('color', color)
            v_blueprint.set_attribute('role_name', 'autopilot')

            transform.location.z += 1
            v_batch.append(
                SpawnActor(v_blueprint, transform).then(
                    SetAutopilot(FutureActor, True, self.tm_port)
                )
            )

        results = client.apply_batch_sync(v_batch, True)
        if len(results) < n_vehicles:
            logging.warning(
                "{} vehicles were requested but could only spawn {}".format(
                    n_vehicles, len(results)
                )
            )
        vehicles_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walkers
        spawn_locations = [
            self.world.get_random_location_from_navigation() for i in range(n_walkers)
        ]

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = client.apply_batch_sync(w_batch, True)
        if len(results) < n_walkers:
            logging.warning(
                "Could only spawn {} out of the {} requested walkers.".format(
                    len(results), n_walkers
                )
            )
        walkers_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walker controllers
        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = client.apply_batch_sync(wc_batch, True)
        if len(results) < len(walkers_id_list):
            logging.warning(
                "Only {} out of {} controllers could be created. Some walkers might be stopped".format(
                    len(results), n_walkers
                )
            )
        controllers_id_list = [r.actor_id for r in results if not r.error]

        self.world.tick()

        for controller in self.world.get_actors(controllers_id_list):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.world.tick()
        self.actors = self.world.get_actors(
            vehicles_id_list + walkers_id_list + controllers_id_list
        )

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(
                    pitch=server_view_pitch, yaw=server_view_yaw, roll=server_view_roll
                ),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calcualted at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))

        return sensor_data

    def destroy_actors_sensors(self, client):
        self.sensor_interface.destroy()
        client.apply_batch(
            [
                carla.command.DestroyActor(x)
                for x in self.world.get_actors()
                if x.is_alive
            ]
        )
        self.world.tick()
        time.sleep(0.5)

    def get_experiment_configs(self):
        return NotImplementedError

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return NotImplementedError

    def get_data_to_log(self, observation):
        return None

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return NotImplementedError

