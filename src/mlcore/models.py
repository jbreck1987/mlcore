"""
Model definitions to be used in runs.
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

from torch import nn
import torch
    
class BranchedConvReg(nn.Module):
    def __init__(self, in_channels, h_hidden_units, h_hidden_layers) -> None:
        super().__init__()
        self.h_hidden_units = h_hidden_units
        self.h_hidden_layers = h_hidden_layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=5) , 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5), 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2), 
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        # Arrival network is straightforward
        self.arrival_regression = nn.Sequential(
            nn.Linear(in_features=128*5, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=1)
        )
        
        # Height regression network has its hidden layers dynamically built
        self.height_regression = nn.Sequential(nn.Linear(128*5, h_hidden_units),
                                               *self.hidden_layers_list(), # access values of the resulting list
                                               nn.Linear(h_hidden_units, 1))
    def hidden_layers_list(self):
        """
        Dynamically builds the hidden layers for the height regression network.
        Based off of the original model architecture, we need to add a dropout
        layer after each linear layer. In addition, the previous model had a
        non-linear activation layer sandwiched in the middle of the linear
        layers, which is replicated here.
        """
        l = nn.ModuleList()
        for idx in range(self.h_hidden_layers):
            # Add the ReLU layer if in middle of building hidden layers
            if idx == (self.h_hidden_layers // 2):
                l.extend([nn.ReLU()])
            l.extend([nn.Linear(self.h_hidden_units, self.h_hidden_units), nn.Dropout(0.5)])
        return l
        

    
    def forward(self, x) -> torch.Tensor:
        # Want to keep the parameters in the arrival and height FC branches separate from each other,
        # the two graphs will be merged in the feature extractor
        out_arrival = self.arrival_regression(self.feature_extractor(x))
        out_height = self.height_regression(self.feature_extractor(x))
        return out_arrival, out_height
