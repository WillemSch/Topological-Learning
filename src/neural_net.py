import torch
import torch.nn as nn
import torch.nn.functional as functional
from homology import rips
from math import prod
import numpy as np
from util import create_distance_matrix


class TopologicalAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        # Input_shape is tuple with: (Amount of simplices, input dimensions)
        # Latent_space_size is: latent dimensions

        assert(kwargs["input_shape"] is not None, "parameter input_shape required")
        assert(kwargs["latent_space_size"] is not None, "parameter latent_space_size required")

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
    def __init__(self):
        super().__init__()
        self.relevant_input = None
        self.relevant_latent = None
        self.A_input = None
        self.A_latent = None

    def forward(self, input, latent, output, point_count):
        # assert(dimensions == 1, "This implementation only supports 1 dimensional homology")

        # flattened layers into dimensional vectors
        # Select "important" simplices
        self.relevant_input = self.__relevant_points(rips(input, dimensions=1)[0])
        self.relevant_latent = self.__relevant_points(rips(latent, dimensions=1)[0])

        self.A_input = create_distance_matrix(output.reshape((point_count, len(input) // point_count)))
        self.A_latent = create_distance_matrix(output.reshape((point_count, len(latent) // point_count)))

        loss_reconstruction = functional.mse_loss(input, output)

        loss_x = .5 * torch.norm(self.__relevant_distances(self.relevant_input, self.A_input) - self.__relevant_distances(self.relevant_input, self.A_latent))
        loss_z = .5 * torch.norm(self.__relevant_distances(self.relevant_latent, self.A_latent) - self.__relevant_distances(self.relevant_latent, self.A_input))

        return loss_reconstruction + loss_x + loss_z

    def __relevant_distances(self, relevent_indices, distance_matrix):
        return np.array([distance_matrix[i[0], i[1]] for i in relevent_indices])

    def __relevant_points(self, diagram):
        non_zeros = np.count_nonzero(diagram, axis=0)
        simplex_indices = torch.where(diagram.T[np.where(non_zeros == 2)] == 1)[1]
        index_tuples = simplex_indices.reshape((len(simplex_indices) // 2, 2))
        return index_tuples

