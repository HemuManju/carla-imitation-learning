import numpy as np
import webdataset as wds
import torch
import scipy.interpolate

import matplotlib.pyplot as plt

from .preprocessing import get_preprocessing_pipeline

from .utils import (
    rotate,
    get_dataset_paths,
    generate_seqs,
    find_in_between_angle,
    show_image,
)


def post_process_action(data, config):
    if config['action_processing_id'] == 1:
        action = torch.tensor(
            [data['throttle'], (data['steer'] + 1) * 2, data['brake']]
        )
    elif config['action_processing_id'] == 2:
        action = torch.tensor(
            [
                data['speed'] / 20,
                data['throttle'],
                (data['steer'] + 1) * 2,
                data['brake'],
            ]
        )
    elif config['action_processing_id'] == 3:
        action = torch.tensor([data['speed'] / 5.55, (data['steer'] + 1)])
    elif config['action_processing_id'] == 4:
        # Calculate theta near and theta far
        theta_near, theta_middle, theta_far = calculate_theta_near_far(
            data['waypoints'], data['location']
        )
        action = torch.tensor([theta_near, theta_middle, theta_far, data['steer']])

    elif config['action_processing_id'] == 5:
        ego_frame_waypoints = project_to_ego_frame(data)
        points = ego_frame_waypoints[0:5, :].astype(np.float32)
        action = torch.from_numpy(points)
    else:
        action = torch.tensor([data['throttle'], data['steer'], data['brake']])

    return action


def calc_ego_frame_projection(x, moving_direction):
    theta = find_in_between_angle(moving_direction, np.array([0.0, 1.0, 0.0]))
    projected = rotate(x[0:2].T, theta)
    projected[0] *= -1
    return projected


def project_to_ego_frame(data):
    # Direction vector
    moving_direction = np.array(data['moving_direction'])

    # Origin shift
    shifted_waypoints = np.array(data['waypoints']) - np.array(data['location'])

    # Projected points
    projected_waypoints = np.zeros((len(shifted_waypoints), 2))
    for i, waypoint in enumerate(shifted_waypoints):
        projected_waypoints[i, :] = calc_ego_frame_projection(
            waypoint, moving_direction
        )
    return projected_waypoints


def calc_world_projection(ego_frame_coord, moving_direction, ego_location):

    ego_frame_coord[0] *= -1
    #  Find the rotation angle such the movement direction is always positive
    theta = find_in_between_angle(moving_direction, np.array([0.0, 1.0, 0.0]))

    # Rotate the pointd back
    re_projected = rotate(ego_frame_coord.T, -theta)
    re_projected += ego_location
    return re_projected


def project_to_world_frame(ego_frame_waypoints, data):
    ego_location = np.array(data['location'])
    v_vec = np.array(data['moving_direction'])

    # Projected points
    projected_waypoints = np.zeros((len(ego_frame_waypoints), 2))
    for i, waypoint in enumerate(ego_frame_waypoints):
        projected_waypoints[i, :] = calc_world_projection(
            waypoint, v_vec, ego_location[0:2]
        )
    return projected_waypoints


def calculate_angle(v0, v1):
    theta = np.arctan2(np.cross(v0, v1), np.dot(v0, v1)).astype(np.float32)
    if theta > 3.0:
        theta = 0.0
    return theta


def resample_waypoints(waypoints, current_location, resample=False):
    if resample:
        xy = np.array(waypoints)
        x = xy[:, 0]
        y = xy[:, 1]

        # Add initial location
        x = np.insert(x, 0, current_location[0])
        y = np.insert(y, 0, current_location[1])

        # Shift the frame to origin
        # x = x - x[0]
        # y = y - y[0]

        # get the cumulative distance along the contour
        dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))

        # build a spline representation of the contour
        try:
            spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)
            # resample it at smaller distance intervals
            interp_d = np.linspace(dist_along[0], dist_along[-1], 10)
            interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)
        except ValueError:
            interp_x = x
            interp_y = y

        processed_waypoints = np.vstack((interp_x, interp_y)).T
    else:
        processed_waypoints = np.array(waypoints)[:, 0:2]
    return processed_waypoints


def calculate_theta_near_far(waypoints, location):
    # NOTE: The angles returned are in radians.
    # The x and y position are given in world co-ordinates, but theta near and theta far should
    # be calculated in the direction of movement.

    if len(waypoints) > 1:
        # resample waypoints
        current_location = np.array(location[0:2])
        waypoints = resample_waypoints(waypoints, current_location)

        # From the vectors taking ego's location as origin
        v0 = waypoints[0] - current_location

        # Select the second point as the near points
        point_select = 1
        v1 = waypoints[point_select] - current_location
        theta_near = calculate_angle(v0, v1)

        point_select = 2
        v1 = waypoints[point_select] - current_location
        theta_middle = calculate_angle(v0, v1)

        # Select the fourth point as the far point
        point_select = 4
        v1 = waypoints[point_select] - current_location
        theta_far = calculate_angle(v0, v1)
    else:
        theta_far, theta_middle, theta_near = 0.0, 0.0, 0.0

    return float(theta_near), float(theta_middle), float(theta_far)


def concatenate_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images).squeeze(1)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :crop_size, :]

    last_data = samples[-1]['json']

    if last_data['modified_direction'] in [-1, 5, 6]:
        command = 4
    else:
        command = last_data['modified_direction']

    # Post processing according to the ID
    action = post_process_action(last_data, config)
    n_waypoints = config['n_waypoints']

    return images, command, action[0:n_waypoints, :]


def concatenate_test_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images).squeeze(1)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :crop_size, :]

    last_data = samples[-1]['json']

    if last_data['modified_direction'] in [-1, 5, 6]:
        command = 4
    else:
        command = last_data['modified_direction']

    # Post processing according to the ID
    action = post_process_action(last_data, config)
    n_waypoints = config['n_waypoints']

    return images, command, action[0:n_waypoints, :], last_data


def webdataset_data_test_iterator(config, file_path):
    # Get dataset path(s)
    paths = get_dataset_paths(config)

    # Parameters
    SEQ_LEN = config['obs_size']

    dataset = (
        wds.WebDataset(file_path, shardshuffle=False)
        .decode("torchrgb")
        .then(generate_seqs, concatenate_test_samples, SEQ_LEN, config)
    )
    return dataset


def webdataset_data_iterator(config):
    # Get dataset path(s)
    paths = get_dataset_paths(config)

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['obs_size']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .then(generate_seqs, concatenate_samples, SEQ_LEN, config)
            )
            data_loader = wds.WebLoader(
                dataset,
                num_workers=number_workers,
                shuffle=False,
                batch_size=BATCH_SIZE,
            )
            if key in ['training', 'validation']:
                dataset_size = 6250 * len(path)
                data_loader.length = dataset_size // BATCH_SIZE

            data_iterator[key] = data_loader

    return data_iterator
