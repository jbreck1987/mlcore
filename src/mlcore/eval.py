"""
This module contains utilities that help with evaluating the performance of a model
"""

# Author: Josh Breckenridge
# Date: 7/26/2023

import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List

def plot_stream_data(i_stream: np.array,
                     q_stream: np.array,
                     label_stream: np.array,
                     units: str
                     ) -> None:
    """
    Plots the training data. Assumes training samples are I and Q timestreams
    and the labels are timestreams of photon events (based on the I/Q values).
    """

    _, ax = plt.subplots(3,1,figsize = (10,10))
    ax[0].plot(np.arange(0, i_stream.size), i_stream)
    ax[0].set_xlabel(f'Time ({units})')
    ax[0].set_ylabel('I Timestream', fontweight = 'bold', size = 'large')

    ax[1].plot(np.arange(0, q_stream.size), q_stream)
    ax[1].set_xlabel(f'Time ({units})')
    ax[1].set_ylabel('Q Timestream', fontweight = 'bold', size = 'large')

    ax[2].plot(np.arange(0, label_stream.size), label_stream)
    ax[2].set_xlabel(f'Time ({units})')
    ax[2].set_ylabel('Photon Timestream', fontweight = 'bold', size = 'large')
    plt.show()




def accuracy_regression(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Returns the accuracy [0, 1] of the predicted value compared to the true
    value for a regularized regression model (targets are in range [0,1])
    """
    mag = torch.abs(y_pred.mean() - y_true.mean()).item()

    # Difference between outputs > 1 implies 0% accuracy
    return 0.0 if mag > 1 else 1 - mag


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