"""
This module contains functions/classes that can be used to manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import torch
import numpy
import math
import random
import datetime as dt
import pathlib
from typing import List

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream
from mkidreadoutanalysis.resonator import Resonator, FrequencyGrid, RFElectronics, ReadoutPhotonResonator, LineNoise

# Define constant resonator, readout electronics, noise, and frequency sweep objects
_RES = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1e2)
_FREQ_GRID = FrequencyGrid(fc=_RES.f0, points=1000, span=500e6)
_LINE_NOISE = LineNoise(freqs=[60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3],
                        amplitudes=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01],
                        phases=[0, 0.5, 0,1.3,0.5, 0.2, 2.4],
                        n_samples=100,
                        fs=1e3)
_RF = RFElectronics(gain=(3.0, 0, 0),
                    phase_delay=0,
                    white_noise_scale=10,
                    line_noise=_LINE_NOISE,
                    cable_delay=50e-9)

def gen_iqp(qp_timestream: QuasiparticleTimeStream, norm: bool = True):
    """
    Generate I, Q, and Phase Response time streams using the mkidreadoutanalysis library

    Inputs: 
        qp_timestream: This object should have the photons generated before passing
        norm: Determines whether to return normalized or non-normalized I/Q response
    
    Returns: tuple of three numpy arrays containing the I, Q, and Phase Response timestreams respectively.
    """

    # Create Photon Resonator Readout
    readout = ReadoutPhotonResonator(_RES, qp_timestream, _FREQ_GRID, _RF, noise_on=True)

    # Return I, Q, and Phase Response timestreams
    if bool:
        return readout.normalized_iq.real, readout.normalized_iq.imag, readout.basic_coordinate_transformation()[0]
    return readout.iq_response.real, readout.iq_response.imag, readout.basic_coordinate_transformation()[0]

def create_windows(i: numpy.array,
                   q: numpy.array,
                   photon_arrivals: numpy.array,
                   qp_densities: numpy.array,
                   phase_responses: numpy.array,
                   with_pulses: list,
                   no_pulses: list,
                   single_pulse: bool,
                   num_samples: int,
                   no_pulse_fraction: float,
                   edge_padding: int,
                   window_size: int,
                   ) -> None:
    """
    This function takes the output of the mkidreadoutanalysis objects (I, Q, and Photon Arrival vectors) and chunks it into smaller arrays. The code
    also separates chunks with photons and those without photons with the goal of limiting the number of samples
    without photon pulses since there are vastly more windows in the synthetic data in this category. It uses "scanning" logic to scan over the full
    arrays with a given window size and inspects that window for a photon event. The window is then added to the appropriate container (photon/no photon).
    """

    # First determine the last index in the scanning range (need to have length of photon arrivals array be multiple of window_size)
    end_idx = math.floor(len(photon_arrivals) / window_size) * window_size

    # Now scan across the photon arrival vector and look at windows of length window_size with and without photon events
    for window in range(0, end_idx - window_size + 1, window_size):
        window_pulses = np.sum(photon_arrivals[window : window + window_size] == 1)
        window_pulse_idxs = np.argwhere(photon_arrivals[window : window + window_size])
        valid_window_start = window + edge_padding
        valid_window_end = window + window_size - edge_padding
        valid_pulse = ((window_pulse_idxs > valid_window_start) & (window_pulse_idxs < valid_window_end)).all()

        # If there are more than one pulses in the entire window and we only want single pulse data, skip this window
        if window_pulses > 1 and single_pulse:
            continue

        # Now any window with a pulse is valid as long as the pulse(s) dont live in the edge padding area
        elif window_pulses > 0 and valid_pulse:
            # If so add the window to the with_pulses container
            with_pulses.append(np.vstack((i[window : window + window_size],
                                          q[window : window + window_size],
                                          photon_arrivals[window : window + window_size],
                                          qp_densities[window : window + window_size],
                                          phase_responses[window : window + window_size])).reshape(5, window_size)) # Reshaping to get in nice form for CNN

        # If no pulses are in the window and the no-pulse fraction hasn't been met,
        # add to the no_pulses container
        elif len(no_pulses) < num_samples * no_pulse_fraction and window_pulses == 0:
            no_pulses.append(np.vstack((i[window : window + window_size],
                                        q[window : window + window_size],
                                        photon_arrivals[window : window + window_size],
                                        qp_densities[window : window + window_size],
                                        phase_responses[window : window + window_size])).reshape(5, window_size)) # Reshaping to get in nice form for CNN


def make_dataset(qp_timestream: QuasiparticleTimeStream,
                 magnitudes: List[int],
                 num_samples: int,
                 no_pulse_fraction: float,
                 with_pulses: list,
                 no_pulses: list,
                 single_pulse: bool,
                 cps=500,
                 edge_padding=0,
                 window_size=150,
                 normalize: bool = True
            ) -> None:
    # Generate the training set in the following format: [np.array([i,q, photon_arrivals, qp_densities]), ...] where i,q,... are all WINDOW_SIZE length arrays.
    # Each list element is a 5 x 1 x WINDOW_SIZE numpy array.
    count = len(with_pulses)
    while len(with_pulses) < num_samples - (num_samples * no_pulse_fraction):
        # We want the training data to be varied, so lets use the Poisson sampled
        # gen_photon_arrivals method to change the photon flux per iteration and also modulate the
        # quasiparticle density to get different pulse heights
        # Note that the magniutdes are referenced to the lowest in the list (I.E. shortest wavelength has largest energy -> highest change in qp density)
        qp_timestream.gen_quasiparticle_pulse(magnitude=min(magnitudes)/random.choice(magnitudes))
        photon_arrivals = qp_timestream.gen_photon_arrivals(cps=cps, seed=None)
        qp_densities = qp_timestream.populate_photons() 
        I, Q, phase_resp = gen_iqp(qp_timestream, normalize)
        # print(f'I: {I.size}, Q: {Q.size}, P: {phase_resp.size}')
        create_windows(I,
                       Q,
                       photon_arrivals,
                       qp_densities,
                       phase_resp,
                       with_pulses,
                       no_pulses,
                       single_pulse,
                       num_samples,
                       no_pulse_fraction,
                       window_size=window_size,
                       edge_padding=edge_padding)
        # Give status update on number of samples with photons
        if len(with_pulses) > count:
            print(f'Num samples with photons: {len(with_pulses)}/{num_samples - (num_samples * no_pulse_fraction)}', end='\r')
            count = len(with_pulses)
    print(f'\nNumber of samples with pulses: {len(with_pulses)}')
    print(f'Number of samples without pulses: {len(no_pulses)}')


def save_model(model_dir: pathlib.Path, filename: str,  model: torch.nn.Module, ext: str) -> None:

    # Append unix epoch to the filename and add the given extension
    filename = f'{filename}_{str(int(dt.datetime.now().timestamp()))}.{ext}'
    print(f'Saving model state_dict to {str(model_dir)} as {filename}')
    torch.save(obj=model.state_dict(), f=model_dir / filename)


