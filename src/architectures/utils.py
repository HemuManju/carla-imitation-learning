import torch

from torch import nn


def build_model(layer_config):
    """Get model from layer config dictionary."""
    modules = []

    for layer in layer_config:
        layer_type = layer.pop("type")
        module = getattr(torch.nn, layer_type)(**layer)
        modules.append(module)
    return nn.Sequential(*modules)
