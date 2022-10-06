from copy import deepcopy
import torch

from torch import nn


def build_conv_model(image_size, layer_config):
    """Get model from layer config dictionary."""
    modules = []
    image_sizes = []

    for layer in deepcopy(layer_config):
        layer_type = layer.pop("type")
        module = getattr(torch.nn, layer_type)(**layer)
        modules.append(module)

        if layer_type in ["Conv2d"]:
            out_image_size = conv_output_shape(
                image_size, layer['kernel_size'], layer['stride']
            )
            image_sizes.append(out_image_size)
            image_size = out_image_size

    return nn.Sequential(*modules), image_sizes


def build_deconv_model(out_size, image_sizes, conv_layer_config):
    """Get model from layer config dictionary."""
    modules = []

    for layer in reversed(conv_layer_config):
        layer_type = layer.pop("type")

        if layer_type in ['Conv2d']:

            # Find the kernel size
            module = nn.ConvTranspose2d(
                layer["out_channels"],
                layer["in_channels"],
                kernel_size=layer["kernel_size"],
                stride=layer["stride"],
                output_padding=1,
            )

        else:
            module = getattr(torch.nn, layer_type)(**layer)

        modules.append(module)
    return nn.Sequential(*modules)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def convtransp_kernel_shape(
    in_size, out_size, stride=1, pad=0, dilation=1, output_pad=0
):
    """
    Utility function for computing output the required kernel size
    give the input and output image takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if dilation == 0:
        raise ValueError('Dilation cannot be zero')

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(output_pad) is not tuple:
        output_pad = (output_pad, output_pad)

    if type(dilation) is not tuple:
        dilation = (dilation, dilation)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (
        out_size[1] - (in_size[1] - 1) * stride[0] + 2 * pad[0] - output_pad[0] - 1
    ) / dilation[0] + 1
    w = (
        out_size[2] - (in_size[2] - 1) * stride[1] + 2 * pad[1] - output_pad[1] - 1
    ) / dilation[1] + 1

    return [h, w]


def get_model(config) -> nn.Module:
    """Get model from layer config dictionary."""
    modules = []
    for l in config:
        layer_type = l.pop("type")
        layer = getattr(torch.nn, layer_type)(**l)
        modules.append(layer)
    return nn.Sequential(*modules)


def conv_output_shape(in_size, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    h = floor(
        ((in_size[1] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((in_size[2] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return [in_size[0], h, w]
