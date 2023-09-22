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


class BranchedConvRegv2(nn.Module):
    def __init__(self, in_channels, h_hidden_units, h_hidden_layers) -> None:
        super().__init__()
        self.h_hidden_units = h_hidden_units
        self.h_hidden_layers = h_hidden_layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1, padding=2) , 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2), # Output is length 500
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2), 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2), # output is length 250
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=5, padding=2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2), # output length is 25
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=2),
            nn.LeakyReLU(),
            nn.Flatten(), # output is 5*64
            nn.Dropout(0.5)
        )
        # Arrival network is straightforward
        self.arrival_regression = nn.Sequential(
            nn.Linear(in_features=64*5, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=1)
        )
        
        # Height regression network has its hidden layers dynamically built
        self.height_regression = nn.Sequential(nn.Linear(64*5, h_hidden_units, bias=True),
                                               *self.hidden_layers_list(), # access values of the resulting list
                                               nn.Linear(h_hidden_units, 1, bias=True))
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
                l.extend([nn.LeakyReLU()])
            l.extend([nn.Linear(self.h_hidden_units, self.h_hidden_units), nn.Dropout(0.5)])
        return l
        

    
    def forward(self, x) -> torch.Tensor:
        # Want to keep the parameters in the arrival and height FC branches separate from each other,
        # the two graphs will be merged in the feature extractor
        out_arrival = self.arrival_regression(self.feature_extractor(x))
        out_height = self.height_regression(self.feature_extractor(x))
        return out_arrival, out_height


class ConvRegHeight(nn.Module):
    def __init__(self, in_channels, h_hidden_units, h_hidden_layers) -> None:
        super().__init__()
        self.h_hidden_units = h_hidden_units
        self.h_hidden_layers = h_hidden_layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1, padding=2) , 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2), # Output is length 500
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2), 
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2), # output is length 250
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=5, padding=2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2), # output length is 25
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=2),
            nn.LeakyReLU(),
            nn.Flatten(), # output is 5*64
            nn.Dropout(0.5)
        )
        # Height regression network has its hidden layers dynamically built
        self.height_regression = nn.Sequential(nn.Linear(64*5, h_hidden_units, bias=True),
                                               *self.hidden_layers_list(), # access values of the resulting list
                                               nn.Linear(h_hidden_units, 1, bias=True))
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
                l.extend([nn.LeakyReLU()])
            l.extend([nn.Linear(self.h_hidden_units, self.h_hidden_units), nn.Dropout(0.5)])
        return l
        
    def forward(self, x) -> torch.Tensor:
        return self.height_regression(self.feature_extractor(x))


class ConvAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10, stride=1, padding='same') , 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=5), # Output is length L/8
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=1, padding='same'), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=5), # output is length L/64
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=5), # output length is L/512
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(), # Latent dims, output is (L/512) * 64
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=5.0), # Output length is 250
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=5.0), # Output length is 500
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=5.0), # Output length is 1000
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=10, stride=1, padding='same'),
            nn.Sigmoid(), # sigmoid needed since inputs are normalized
        )
    def forward(self, x):
        # Reshape is necessary to "unflatten" the latent space vector
        return self.decoder(self.encoder(x).reshape((x.shape[0], 64, x.shape[2] // 125)))

class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=10000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=250), 
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=125),
            nn.ReLU(),
            nn.Linear(in_features=125, out_features=75), # latent dimension is 75
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=75, out_features=125),
            nn.ReLU(),
            nn.Linear(in_features=125, out_features=250), 
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=10000),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

#### Modular Convolutional Autoencoder ####

class ConvEncoderLayer1D(nn.Module):
    """
    This is a convience class used to make code a little easier to parse when creating
    models that use the Convolutional Autoencoder architecture. It combines commonly used encoder
    layers into one layer. The class packages the Conv1d, ReLU, MaxPool1d, and Dropout layers
    together.

    parameters:
    -in_ch: Number of input channels to the layer group
    -out_ch: Number of output channels from the layer group
    -kernel_size: Kernel size used in convolution layer
    -stride: Stride length used in convolution layer
    -pad: Padding used in the convolution layer; uses symmetrical 0 padding; See Conv1d docs for acceptable inputs.
    -pool_kernel_size: Kernel size used in the pooling layer
    -dropout_prob: Probability that a weight will be ignored during training step
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int,
                 stride: int,
                 pad: int | str,
                 pool_kernel_size: int,
                 dropout_prob: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.pool_kernel_size = pool_kernel_size
        self.dropout_prob = dropout_prob

        # Define container to store layers
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=pad),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool1d(kernel_size=pool_kernel_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ConvDecoderLayer1D(nn.Module):
    """
    This is a convience class used to make code a little easier to parse when creating
    models that use the Convolutional Autoencoder architecture. It combines commonly used Decoder
    layers into one layer. The class packages the Conv1d, ReLU, Upsample1d, and Dropout layers
    together.

    parameters:
    -in_ch: Number of input channels to the layer group
    -out_ch: Number of output channels from the layer group
    -kernel_size: Kernel size used in convolution layer
    -stride: Stride length used in convolution layer
    -pad: Padding used in the convolution layer; uses symmetrical 0 padding; See Conv1d docs for acceptable inputs.
    -ups_kernel_size: Kernel size used in the upsampling layer
    -dropout_prob: Probability that a weight will be ignored during training step
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int,
                 stride: int,
                 pad: int | str,
                 upsample_scale: int,
                 dropout_prob: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.upsample_scale = upsample_scale
        self.dropout_prob = dropout_prob

        # Define container to store layers
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=pad),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Upsample(scale_factor=upsample_scale)
        )
    
    def forward(self, x):
        return self.net(x)

class AELatentLayer(nn.Module):
    """
    This class is used to encapsulate the latent layer of a (non-variational) AutoEncoder.
    
    parameters:
    -in_feat: Number of input features to the linear layer
    -out_feat: Number of output features (number of activation units in the layer)
    -bias: Flag for determining whether to use a bias unit in the layer.
    -out_shape: Used to reshape the output vector into the appropriate shape for the
    subsequent Decoder layer.

    """
    def __init__(self, in_feat: int, out_feat: int, out_shape: tuple, bias: bool) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.out_shape = out_shape

    def forward(self, x):
        return nn.Linear(in_features=self.in_feat, out_features=self.out_feat, bias=self.bias)(x).reshape(self.out_shape)