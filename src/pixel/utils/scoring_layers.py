import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProduct:
    def __init__(
            self,
            func=lambda raw_embs, cands_embs:torch.einsum('bxeh,bceh->bc', raw_embs, cands_embs)
    ):
        self.func = func

    def __call__(self, raw_embs, cands_embs):
        return self.func(raw_embs, cands_embs)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        residual = x  # Save the input as the residual
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        return F.relu(out + residual)  # Add the residual


def create_nn(hidden_size, num_layers, res_net=False):
    """
    Creates a feedforward or residual neural network based on the parameters.
    
    Args:
        hidden_size (int): The size of each hidden layer.
        num_layers (int): The number of layers in the network.
        res_net (bool): If True, returns a residual network; otherwise, a feedforward network.

    Returns:
        nn.Module: The constructed neural network.
    """
    layers = []
    
    if res_net:
        # Residual network
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_size))
    else:
        # Feedforward network
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


class ScoringLayer(nn.Module):
    def __init__(self, raw_size, cands_size, num_layers, inner_product, res_net=False):
        super(ScoringLayer, self).__init__()
        
        self.num_layers = num_layers
        # create the networks
        if self.num_layers > 0:
            self.raw_nn = create_nn(raw_size, num_layers, res_net=res_net)
            self.cands_nn = create_nn(cands_size, num_layers, res_net=res_net)
        
        # define projection layers
        self.raw_size = raw_size
        self.cands_size = cands_size
        if self.raw_size != self.cands_size:
            self.projection_size = min(raw_size, cands_size)
            self.raw_projection = nn.Linear(raw_size, self.projection_size)
            self.cands_projection = nn.Linear(cands_size, self.projection_size)

        # define the inner product
        self.inner_product = inner_product
    
    def forward(self, raw_input, cands_input):
        # pass inputs through their respective networks, if needed
        if self.num_layers > 0:
            raw_output = self.raw_nn(raw_input)
            cands_output = self.cands_nn(cands_input)
        else:
            raw_output = raw_input
            cands_output = cands_input
        
        # project outputs to the same dimensions, if needed
        if self.raw_size != self.cands_size:
            raw_project = self.raw_projection(raw_output)
            cands_project = self.cands_projection(cands_output)
        else:
            raw_project = raw_output
            cands_project = cands_output

        # compute the inner product (dot product) of the projected vectors
        score = self.inner_product(raw_project, cands_project)
        
        return score
