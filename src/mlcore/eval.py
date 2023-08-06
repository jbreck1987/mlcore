"""
This module contains utilities that help with evaluating the performance of a model
"""

# Author: Josh Breckenridge
# Date: 7/26/2023

import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List

def plot_stream_data(units: str, **kwargs) -> None:
    """
    Plots the training data. Assumes training samples are
    I/Q/photon arrival/qp density/phase response timestreams.
    The functions allows for submitting any combination of these
    timestreams and will plot accordingly.
    """

    # Define the allowed keys that can be passed into the function
    # Also making this a dictionary of the labels to be used in the plots
    # for a given passed in key.
    allowed_kw = {'i': 'I',
                  'q': 'Q',
                  'phase_response': 'Phase Response',
                  'photon_arrivals': 'Photon Arrivals',
                  'pred': 'Model Prediction',
                  'qp_density': r'$\Delta$ Quasiparticle Density'}
    
    # Want to get a list of all the keys that are passed as kwargs but arent in the
    # allowed kwarg dict above
    unknown_kw = [passed_key for passed_key in kwargs.keys() if passed_key not in allowed_kw.keys()]
    if len(unknown_kw) > 0:
        raise KeyError(f"Unknown keys {str(unknown_kw).strip('[]')}")
    
    # Loop over the passed in keys and add subplot for the associated
    # quantity
    _, ax = plt.subplots(len(kwargs.keys()),figsize=(10, 5 * len(kwargs.keys())))
    for idx, key in enumerate(kwargs.keys()):
        if len(kwargs.keys()) == 1:
            ax.plot(np.arange(0, kwargs[key].size), kwargs[key])
            ax.set_xlabel(f'Time ({units})')
            ax.set_ylabel(allowed_kw[key], fontweight = 'bold', size = 'small')
        else:
            ax[idx].plot(np.arange(0, kwargs[key].size), kwargs[key])
            ax[idx].set_xlabel(f'Time ({units})')
            ax[idx].set_ylabel(allowed_kw[key], fontweight = 'bold', size = 'medium')
    plt.show()

def accuracy_regression(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Returns the accuracy [0, 1] of the predicted value compared to the true
    value for a regularized regression model (targets are in range [0,1])
    """
    mag = torch.abs(y_pred.mean() - y_true.mean()).item()

    # Difference between outputs > 1 implies 0% accuracy
    return 0.0 if mag > 1 else 1 - mag

def regression_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    This function first finds the mean prediction and target vectors in the
    batch and then returns the absolute difference between them (lower is better)
    """
    return torch.abs(y_pred.mean().abs() - y_true.mean().abs()).item()

def add_noise(pulse_list: List[np.array], range: float) -> None:
    """
    Adds uniform noise to photon arrival data. The pulse_list is expected to have the shape returned by
    the make_dataset function (with photon arrival data in dimension 2).

    Inputs:
        pulse_list: list of np arrays, each containing I/Q/photon timestream data
        range: max value of sampled values to add as noise
    """
    rng = np.random.default_rng() 
    for sample in pulse_list:
        # Save the index with the pulse
        pulse_idx = sample[2] == 1 
        # Add the noise 
        sample[2] = rng.random(sample[2].shape) * (range - 0)
        # Make sure the pulse is still 1 
        sample[2][pulse_idx] = 1