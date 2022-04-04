import torch.nn as nn
from homology import rips
from math import prod


class TopologicalAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        # Input_shape is tuple with: (Amount of simplices, input dimensions)
        # Latent_space_size is: latent dimensions
        super().__init__()
        self.act = nn.ReLU()

        intermediate_layer_dimensions = (kwargs["input_shape"][1] + kwargs["latent_space_size"]) // 2

        self.lin_1 = nn.Linear(
            in_features=prod(kwargs["input_shape"]),
            out_features=kwargs["input_shape"][0] * intermediate_layer_dimensions
        )

        self.lin_2 = nn.Linear(
            in_features=kwargs["input_shape"][0] * intermediate_layer_dimensions,
            out_features=kwargs["input_shape"][0] * kwargs["latent_space_size"]
        )

        self.lin_3 = nn.Linear(
            in_features=kwargs["input_shape"][0] * kwargs["latent_space_size"],
            out_features=kwargs["input_shape"][0] * intermediate_layer_dimensions
        )

        self.lin_4 = nn.Linear(
            in_features=kwargs["input_shape"][0] * intermediate_layer_dimensions,
            out_features=prod(kwargs["input_shape"])
        )

    def forward(self, x):
        out = self.act(self.lin_1(x))
        latent = self.act(self.lin_2(out))
        out = self.act(self.lin_3(latent))
        out = self.act(self.lin_4(out))
        return out, latent


class TopAELoss(nn.Module):
    def forward(ctx, input, latent, output):
        # flattened layers into dimensional vectors
        # Select "important" simplices
        ctx.A_input = rips(input)
        ctx.A_latent = rips(latent)
        ctx.A_output = rips(output)


    def backward(ctx, out):
        # grad = 0 - Lx - Ly
        # Lx, Ly = 0 if not relevant
        for x in relevant_points:

        None
