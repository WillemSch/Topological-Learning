import torch
import torch.nn as nn
import torch.nn.functional as functional
from homology import Rips
from math import prod
import numpy as np
from util import create_distance_matrix


class PersLay(nn.Module):
    """A Pytorch implementation of the PersLay. This implementation will use a small fully connected network as weight
    function, and a different small fully connected network as phi. This is used to vectorize a persistence diagram
    in a neural network.

    :param output_dim: The amount of output nodes of the PersLay.
    """

    def __init__(self, output_dim):
        """Initializes The persLay class, with it the weight- and phi network. Note that the networks use convolutional
        layers calculate the weights and phis for all points at once.

        :param output_dim: The amount of output nodes of the PersLay.
        """
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 2), stride=(1, 2), out_channels=8, in_channels=1),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(1, 1), stride=(1, 1), out_channels=1, in_channels=8),
            nn.ReLU()
        )

        self.phi = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 2), stride=(1, 2), out_channels=output_dim//2, in_channels=1),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(1, 1), stride=(1, 1), out_channels=output_dim, in_channels=output_dim//2),
            nn.ReLU()
        )

    def forward(self, x):
        """Passes a dataset Tensor through the PersLay.

        :param x: A Tensor containing the data.
        :return: A Tensor containing the output of this PersLay.
        """
        weight = self.weight.forward(x.to(dtype=torch.float))
        phi = torch.swapaxes(self.phi.forward(x.to(dtype=torch.float)), 1, 3)
        weight = weight.repeat(1, 1, 1, phi.shape[3])
        out = weight * phi
        out = torch.squeeze(torch.sum(out, dim=2))
        return out


class TopologicalAutoEncoder(nn.Module):
    """Implement a Topological AutoEncoder in pytorch.

    :param input_shape: Tuple containing: (Amount of simplices, input dimensions)
    :param latent_space_size: An integer stating the amount of dimensions in the latent space.
    """

    def __init__(self, **kwargs):
        """Initializes a topological autoencoder with 4 fully connected layers (2 for encoder, 2 for decoder).

        :param input_shape: Tuple containing: (Amount of simplices, input dimensions)
        :param latent_space_size: An integer stating the amount of dimensions in the latent space.
        """

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
        """Passes a dataset Tensor through the PersLay.

        :param x: A Tensor containing the data.
        :return: A Tensor containing the output of the AutoEncoder, and a Tensor containing the output at the latent
            space.
        """
        out = self.act(self.lin_1(x))
        latent = self.act(self.lin_2(out))
        out = self.act(self.lin_3(latent))
        out = self.act(self.lin_4(out))
        return out, latent


class TopAELoss(nn.Module):
    """A pytorch implementation of the Topological AutoEncoder loss function.
    """

    def __init__(self):
        """Initializes the loss function.
        """
        super().__init__()
        self.relevant_input = None
        self.relevant_latent = None
        self.A_input = None
        self.A_latent = None

    def forward(self, input, latent, output, point_count):
        """Calculates the loss for a given input and its corresponding latent-space, and output. This is a combination
        of the reconstruction loss between input and output, and the homology loss between input and latent space.

        :param input: The input of the AutoEncoder.
        :param latent: The latent space of the AutoEncoder when fed with input.
        :param output: The output of the AutoEncoder when fed with input
        :param point_count: The amount of points in the diagram that is the input.
        :return: A Tensor with the loss of the AutoEncoder.
        """

        # assert(dimensions == 1, "This implementation only supports 1 dimensional homology")
        rips = Rips(dimensions=1)
        self.relevant_input = self.__relevant_points(rips.fit(input.numpy()).transform(input.numpy())[0])
        self.relevant_latent = self.__relevant_points(rips.fit(latent.numpy()).transform(latent.numpy())[0])

        self.A_input = create_distance_matrix(output.reshape((point_count, len(input) // point_count)))
        self.A_latent = create_distance_matrix(output.reshape((point_count, len(latent) // point_count)))

        loss_reconstruction = functional.mse_loss(input, output)

        loss_x = .5 * torch.norm(self.__relevant_distances(self.relevant_input, self.A_input) - self.__relevant_distances(self.relevant_input, self.A_latent))
        loss_z = .5 * torch.norm(self.__relevant_distances(self.relevant_latent, self.A_latent) - self.__relevant_distances(self.relevant_latent, self.A_input))

        return loss_reconstruction + loss_x + loss_z

    def __relevant_distances(self, relevant_indices, distance_matrix):
        """Get a list of relevant distances from a distance_matrix with pre-determined relevant simplexes.

        :param relevant_indices: The indices of the relevant simplexes.
        :param distance_matrix: A distance matrix of all simplexes.
        :return: Numpy array of distances between the relevant simplexes.
        """
        return np.array([distance_matrix[i] for i in relevant_indices])

    def __relevant_points(self, diagram):
        """Find the relevant simplexes in a persistence diagram. (Only works for 1 dimensional homology at the moment)

        :param diagram: A persistence diagram without column reduction applied.
        :return: A list of index tuples of the relevant simplexes.
        """
        non_zeros = np.count_nonzero(diagram, axis=0)
        # All edges are "destroyers" of 0 dimensional simplexes, and are therefore relevant
        simplex_indices = torch.where(diagram.T[np.where(non_zeros == 2)] == 1)[1]
        index_tuples = simplex_indices.reshape((len(simplex_indices) // 2, 2))
        return index_tuples

