"""
This module contains functions/classes that can be used to manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import torch
from itertools import repeat
from functools import reduce
from warnings import warn
import datetime as dt
import pathlib
from typing import List, Any, Tuple
from collections.abc import Iterable
from types import FunctionType
from hashlib import md5
import os
from shutil import copy2
from inspect import getmembers, isclass, getfullargspec

import ruamel.yaml as yaml

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream
from mkidreadoutanalysis.resonator import Resonator, FrequencyGrid, RFElectronics, ReadoutPhotonResonator, LineNoise
from mkidreadoutanalysis.optimal_filters.make_filters import Calculator
import mlcore.dataset.yaml_constructors as yaml_constructors

### DATASET GENERATION FUNCTIONS ###
#----------------------------------#

def gen_iqp(qp_timestream: QuasiparticleTimeStream, conf_var: dict, white_noise_scale: float):
    """
    Generate I, Q, and Phase Response time streams using the mkidreadoutanalysis library

    Inputs: 
        qp_timestream: This object should have the photons generated before passing
        conf_var: Dict that has all of the configuration values to generate the appropriate
        objects necessary go generate the resulting time streams.
    
    Returns: tuple of three numpy arrays containing the I, Q, and Phase Response timestreams respectively.
    """

    # Decompose the configuration variable dict into sub-components
    res_conf = conf_var['resonator']
    f_grid_conf = conf_var['freq_grid']
    noise_conf = conf_var['noise']
    coords_conf = conf_var['coordinates']

    res = Resonator(f0=res_conf['f0'],
                    qi=res_conf['qi'],
                    qc=res_conf['qc'],
                    xa=res_conf['xa'],
                    a=res_conf['a'],
                    tls_scale=res_conf['tls_scale']
                    )
    f_grid = FrequencyGrid(fc=f_grid_conf['fc'], points=f_grid_conf['points'], span=f_grid_conf['span'])
    line_noise = LineNoise(freqs=noise_conf['line_noise']['freqs'],
                            amplitudes=noise_conf['line_noise']['amplitudes'],
                            phases=noise_conf['line_noise']['phases'],
                            n_samples=noise_conf['line_noise']['n_samples'],
                            fs=noise_conf['line_noise']['fs']
                            )
    rf = RFElectronics(gain=tuple(noise_conf['rf_electronics']['gain']),
                        phase_delay=noise_conf['rf_electronics']['phase_delay'],
                        white_noise_scale=white_noise_scale,
                        line_noise=line_noise,
                        cable_delay=noise_conf['rf_electronics']['cable_delay']
                        )
    # Create Photon Resonator Readout
    readout = ReadoutPhotonResonator(res, qp_timestream, f_grid, rf, noise_on=noise_conf['noise_on'])

    # Return I, Q, and Phase Response timestreams
    if coords_conf['normalize_iq']:
        return readout.normalized_iq.real, readout.normalized_iq.imag, readout.basic_coordinate_transformation()[0]
    return readout.iq_response.real, readout.iq_response.imag, readout.basic_coordinate_transformation()[0]


def make_arrivals(num_samples: int,
                  window_size: int,
                  edge_pad: int,
                  lam: float,
                  single_pulse: bool = True,
                  flatten: bool= False,
                  shuffle: bool = True,
                  random_seed = None) -> np.array:
    """
    Creates photon arrival timestream in training sample format (E.g. In the shape (num_samples, window_size), where window_size 
    is the length of the training sample). The function will also shuffle the indices if desired and will flatten the output. This is 
    useful if you need the timestream for other tools that most likely do not need the sample format. The complexity is due to guaranteeing
    that all indices within the valid window will have at least one sample as long as the number of samples is larger than the valid window size.
    """
    # Catch incorrect inputs
    if 2 * edge_pad >= window_size:
       raise ArithmeticError('Window size needs to be > than 2 * edge pad')
    
    # If requesting multiple pulses per window, call function defined to handle that case.
    if single_pulse is False:
        return make_arrivals_multi(num_samples, window_size, edge_pad, lam, flatten, random_seed)

    # Create identity array. If the window size is larger than the number of samples
    # only take the number of rows up to the size of number of samples and return.
    id_mat = np.identity(window_size - (2 * edge_pad), dtype=bool)
    if num_samples < window_size:
        samples = id_mat[:num_samples]
        if shuffle:
            rng = np.random.default_rng() if random_seed is None else np.random.default_rng(random_seed=random_seed)
            rng.shuffle(samples) # in-place
        # Pad the samples to get to the appropriate requested dimensions
        samples = np.pad(samples, ((0,0),(edge_pad, edge_pad)), 'constant', constant_values=(0,))
        if flatten:
            samples = samples.flatten()
        return samples

   
    # If number of requested samples is larger than window size, stack multiple identiy matrices to get the number of samples necessary. 
    # Due to the identity matrix being square, need to do some checking on whether the side length of the identity matrix divides
    # the number of samples requested. This clause catches the case where there will be a remainder.
    if num_samples % id_mat.shape[0] != 0:
        remain = num_samples % id_mat.shape[0]

        # Find largest value less than the number of requested samples that the identity array side length divides
        # and create a stacked array to give that number of samples.
        # This is a foldr on an iterable of repeated identity matrixes.
        prim_arr = reduce(lambda acc, x: np.vstack((acc, x)), repeat(id_mat, int((num_samples - remain) / id_mat.shape[0])))

        # The remanining number of samples to create will always be less than the side length of the identity matrix from above,
        # get the rest of the samples to reach the requested value from one identity matrix and stack with the primary array
        # created previously.
        rem_arr = id_mat[:remain]
    
        # Stack the two arrays
        samples = np.vstack((prim_arr, rem_arr))
    
        if shuffle:
            rng = np.random.default_rng() if random_seed is None else np.random.default_rng(random_seed=random_seed)
            rng.shuffle(samples) # in-place
        # Pad the samples to get to the appropriate requested dimensions
        samples = np.pad(samples, ((0,0),(edge_pad, edge_pad)), 'constant', constant_values=(0,))
        if flatten:
            samples = samples.flatten()
        return samples
   
    # Otherwise, window size divides number of samples
    samples = reduce(lambda acc, x: np.vstack((acc, x)), repeat(id_mat, int(num_samples / id_mat.shape[0])))
    if shuffle:
       rng = np.random.default_rng() if random_seed is None else np.random.default_rng(random_seed=random_seed)
       rng.shuffle(samples) # in-place shuffling.

    # Pad the samples
    samples = np.pad(samples, ((0,0),(edge_pad, edge_pad)), 'constant', constant_values=(0,))
    if flatten:
        samples = samples.flatten()
    return samples

def make_arrivals_multi(num_samples: int, window_size: int, edge_pad: int, lam: float, flatten: bool = False, seed=None):
    """
    Creates photon arrival timestream in training sample format (E.g. In the shape (num_samples, window_size),
    for scenarios where multiple pulses per window are requried. The function will also flatten the output if desired.
    This is useful if you need the timestream for other tools that most likely do not need the sample format.
    """

    # Find length of array where pulses are allowed
    mat_len = window_size - 2 * edge_pad

    # Create photon arrivals using Poisson statistics
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    arrivals_mat = np.array(np.vstack([rng.poisson(lam=lam, size=mat_len) for _ in range(num_samples)]), dtype=bool)

    # Pad the photon arrivals matrix with the appropriate edge padding and return
    if flatten:
        return np.pad(arrivals_mat, ((0,0),(edge_pad, edge_pad)), 'constant', constant_values=(0,)).flatten()
    return np.pad(arrivals_mat, ((0,0),(edge_pad, edge_pad)), 'constant', constant_values=(0,))


def gen_data_dir(file, out_parent_path: pathlib.Path):
    """
    This file generates a directory based on the md5 hash of the data_conf.yaml file
    object that is passed in. This hash will be used to determine whether or not data is unique
    to reduce data duplication. The directory name will be the md5 hash of the data_conf.yaml
    file and will contain a copy of the data_conf.yaml file and the npz file with the actual data.

    Returns True if the directory already exists, otherwise returns Path of the newly created
    directory.
    """

    # Generate the md5 hash of the passed in data_conf.yaml file.
    hash = 'a' + str(md5(file.read()).hexdigest())

    # Try to create a new subdirectory under the passed in parent directory
    # with the hash name. If it exists already, os.mkdir will raise.
    try:
        os.mkdir(pathlib.Path(out_parent_path, hash))
    except FileExistsError as e:
        print(f'Data already exists with hash: {hash}')
        return True
    # Otherwise, directory was created, return path
    return pathlib.Path(out_parent_path, hash)

def make_data(data_conf_path: pathlib.Path, out_parent_path: pathlib.Path, low_pass_coe: list = None, optimal_filt: Calculator = None) -> None:

    # Check to see if path exists
    with open(data_conf_path, 'rb') as f:
        # Generate the new data directory (or return if it already exists)
        path_ret = gen_data_dir(f, out_parent_path)
        if not isinstance(path_ret, pathlib.Path):
            return None

    # Load and parse the yaml file
    with open(data_conf_path, 'r') as f:
        # yaml needs the file as text, not bytes
        conf_var = yaml.safe_load(f)

    # Decompose the configuration variable dict into sub-components
    general_conf = conf_var['general']
    qpt_conf = conf_var['quasiparticle']

    # Generate photon arrival timestream
    photon_arrivals = make_arrivals(
        general_conf['num_samples'],
        general_conf['window_size'],
        general_conf['edge_pad'],
        lam= qpt_conf['cps'] / general_conf['fs'],
        single_pulse=general_conf['single_pulse'],
        flatten=True,
        shuffle=True,
        random_seed=general_conf['random_seed']
    )

    # Using config dict and photon arrivals, generate a new quasiparticle timestream
    # object. The size of the "data" member must match the length of the flattened photon
    # arrivals array.
    qpt = QuasiparticleTimeStream(fs=general_conf['fs'], ts=int(photon_arrivals.size / general_conf['fs']))
    qpt.photon_arrivals = photon_arrivals

    # quasiparticle shift magnitude and white noise scale are the two major
    # variables that will be changed in the training data. Loop over all the
    # passed in values for these variables.
    ret_arr = []
    for mag in qpt_conf['qp_shift_magnitudes']:
        for scale in conf_var['noise']['rf_electronics']['white_noise_scale']:
            qpt.gen_quasiparticle_pulse(magnitude=mag)
            _ = qpt.populate_photons()

            # Generate time streams
            print(f'Generating time streams for mag: {mag}, noise_scale: {scale}...')

            # If there is no optimal filter object passed in,
            # I/Q streams should be noisy but the phase_response should
            # be ideal (no noise).
            if optimal_filt is None:
                # Get noisy I/Q data
                i, q, _ = gen_iqp(qpt, conf_var, white_noise_scale=scale)

                # Toggle noise switch to get ideal phase response
                # timestream, then reactivate
                conf_var['noise']['noise_on'] = False
                _, _, phase_response = gen_iqp(qpt, conf_var, white_noise_scale=scale)
                conf_var['noise']['noise_on'] = True

            # Save the data to the appropriate location in the appropriate format
            # (See save_training_data docstring for format)
            ret_arr.append(np.stack((i.reshape(general_conf['num_samples'], general_conf['window_size']),
                                     q.reshape(general_conf['num_samples'], general_conf['window_size']),
                                     photon_arrivals.reshape(general_conf['num_samples'], general_conf['window_size']),
                                     qpt.data.reshape(general_conf['num_samples'], general_conf['window_size']),
                                     phase_response.reshape(general_conf['num_samples'], general_conf['window_size'])), axis=1))


    print(f'Saving data...')
    save_training_data(np.vstack(ret_arr),
                       path_ret,
                       path_ret.stem)
    
    print(f'Saved data to {path_ret}.')

    # Copy the data_conf.yaml file to the newly created directory
    copy_dir = copy2(data_conf_path, path_ret)
    print(f'Copied yaml config file to {copy_dir}.')




def save_training_data(in_array: Any,
                       dir: pathlib.Path,
                       filename: str,
                       labels: Tuple[str] = ('i', 'q', 'photon_arrivals', 'qp_density', 'phase_response'),
                       ) -> None:
    """
    Saves the training/test data to disk as an npz file after generation.
    This function expects the data to be structured such that the different timestreams
    are stacked along the second axis while the first axis denotes the training sample
    number (this format matches the make_dataset function.)

    Example: in_array is numpy array with shape (100, 5, 200) => there are 100
    training samples, 5 timestreams per example, and the training sample length for each
    stream is 200 elements long.

    The streams will be stored in their own arrays within the npz file so that they can
    be accessed individually to ease data manipulation after loading.

    Inputs:
        -in_array: array_like (anything acceptable by numpy API): See example above
        -dir: Directory to store the file in
        -filename: The name to be given to the file (extension will be appended)
        -labels: These are the labels that will be assigned to the different streams when they are
        saved as arrays in the npz file.
    """

    if not isinstance(in_array, np.ndarray):
        warn(f'Input array is not a numpy array, making copy. This can have memory usage implications...', Warning)
        temp_arr = np.array(in_array)
    else:
        temp_arr = in_array
    try:
        # Try to match the the streams in the given array with the given labels index-wise.
        # If the number of labels doesn't match number of streams, zip() will raise.
        kws = {label: temp_arr[:,stream,:].view() for label, stream in zip(labels, range(temp_arr.shape[1]), strict=True)}
    except ValueError:
        # Suppress the ValueError from zip() and raise a more informative IndexError
        raise IndexError('Number of given labels does not match number of streams in input array.') from None
        
    # If all works out, save the data with the given labels
    np.savez(dir / filename, **kws)


def load_training_data(filepath: pathlib.Path,
                       labels: Tuple[str] = ('i', 'q', 'photon_arrivals', 'qp_density', 'phase_response')) -> Tuple[np.ndarray]:
    """
    Loads training/test data from disk based on the format used in
    the save_training_data() function. The reconstructed array will
    be in the shape (num_training_samples, num_timestreams, timestream length).
    The file format is assumed to npz.

    Inputs:
        -filepath: path to the npz file on disk
        -labels: These are the labels that will be assigned to the different streams when they are
         saved as arrays in the npz file.
    
    Returns:
        -Tuple of the arrays that corresspond to each passed label, matched index-wise.
    """
    
    # Load with given labels. Not doing any special handling of exceptions, will let the numpy API raise.
    with np.load(filepath) as f:
        return tuple([f[label] for label in labels])

#### YAML Constructors ####
def extend_yaml_loader(loader: yaml.SafeLoader, constructor: type, tag: str = None) -> None:
    """
    Takes a YAML loader, YAML constructor tag, and constructor class and adds the constructor to the loader.
    Note that the constructor ID must begin with "!"; E.g. "!TimestreamConf". Uses the name of the class
    to create the tag by default (this dunder needs to be defined in your constructors!)
    """
    if tag is not None:
        loader.constructor.add_constructor(tag, yaml_constructor(constructor))
    else:
        loader.constructor.add_constructor(f'!{constructor.__name__}', yaml_constructor(constructor))

def yaml_constructor(cls: type) -> FunctionType:
    """
    Builds the YAML constructor function for the passed in class since the
    yaml "add_constructor" method only expects two arguments (loader and node).
    Removes the need to define a unique constructor function for each class.
    E.g. `loader.add_constructor('!my_loader', yaml_constructor(my_loader_class))`
    """
    def f(a, b):
        return cls(**a.construct_mapping(b))
    return f


### NEW DATASET GENERATION FUNCTIONS ###

def data_config_loader(data_conf_path: pathlib.Path, constructors: Iterable[type] = None, tags: Iterable[str] = None) -> dict:
    """
    Performs YAML handling, such as adding custom constructors to the YAML loader, and loads the
    data creation configuration file. Will load and add all the constructors in the yaml_constructors
    module by default. Otherwise, will need to pass specific constructors (and, optionally, their tags).
    """

    # Build loader with appropriate custom constructors
    loader = yaml.YAML(typ='safe')
    if constructors is None:
        # Get iterable of all pre-defined constructors
        cons = [con for _, con in getmembers(yaml_constructors) if isclass(con)]

        # Add all constructors to the loader
        _ = list(map(lambda x: extend_yaml_loader(loader, x), cons))
    if constructors is not None and tags is not None:
        # Only add the passed in constructors and their associated tags to the loader
        _ = list(map(lambda x, y: extend_yaml_loader(loader, x, y), constructors, tags))

    if constructors is not None:
        # Only add the passed in constructors with the default tag as the class name
        _ = list(map(lambda x: extend_yaml_loader(loader, x), constructors))
    
    # Load the YAML config file and return the config dict
    with open(data_conf_path, 'rb') as f:
        ret = loader.load(f)
    return ret

def photon_generator(data_conf: dict) -> np.array:
    """
    This function makes use of other functions defined earlier to take an input data configuration dict
    and return a flattened array of photon arrival times.
    """
    # Get the arguments from the make_arrivals function
    # that match the keys in data_conf to be expanded.
    # This is to reduce verbosity.
    kwargs = [key for key in getfullargspec(make_arrivals)[0] if key in data_conf['general'].keys()]
    return make_arrivals(shuffle=True,
                         lam=data_conf['quasiparticle']['cps'] / data_conf['general']['fs'],
                         flatten=True,
                         **{kwarg: data_conf['general'][kwarg] for kwarg in kwargs})

    


### MODEL LOADING AND SAVING FUNCTIONS ###
#----------------------------------------#
def save_model(model_dir: pathlib.Path, filename: str,  model: torch.nn.Module, ext: str) -> None:

    # Append unix epoch to the given filename and add the given extension
    filename = f'{filename}_{str(int(dt.datetime.now().timestamp()))}.{ext}'
    print(f'Saving model state_dict to {str(model_dir)} as {filename}')
    torch.save(obj=model.state_dict(), f=model_dir / filename)


### DATA TRANSFORMATION FUNCTIONS ###
#-----------------------------------#
def stream_to_height(photon_arrivals: np.ndarray,
                     pulse_stream: np.ndarray,
                     norm: bool = True) -> np.ndarray:
    """
    Takes a training/test sample that is a timestream (such as quasiparticle density
    or phase response) and returns the height of the pulse(s) in the training sample.
    If input vectors are 2D, the function assumes the first axis denotes the training
    sample while the second axis is length of the streams. ONLY WORKS FOR SAMPLES WITH
    SINGLE PULSES.

    Inputs:
        -photon_arrivals: Used to determine the locations of the pulse(s)
        -pulse_stream: The timestream containing the pulse(s)
    
    returns:
        -singleton numpy array with the pulse height value. if the input arrays are
        2d, the returned numpy array has shape (num_training_samples, 1), where the second
        axis contains the pulse height for the associated training sample
    """

    # Define auxiliary function that contains the normalization logic to simplify code
    def aux(pa, ps):
        # We need to find the largest absolute value in the pulse height list
        # to do the normalization. 
        max = np.abs(ps[pa == 1]).max()
        return ps[pa == 1] / max # Perform normalization

    # Single training sample case
    if photon_arrivals.shape[0] == 1 or len(photon_arrivals.shape) == 1:
        if norm:
            return aux(photon_arrivals, pulse_stream)
        return pulse_stream[photon_arrivals == 1]
    # Batch case
    if norm:
        return aux(photon_arrivals, pulse_stream).reshape(photon_arrivals.shape[0], 1)
    return pulse_stream[photon_arrivals == 1].reshape(photon_arrivals.shape[0], 1)


def stream_to_arrival(photon_arrivals: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    Takes the photon_arrivals timestream from a training/test sample and returns the 
    location of the pulse. If input vector is 2D, the function assumes the first axis denotes the training
    sample number while the second axis is length of the streams. ONLY WORKS FOR SAMPLES WITH
    SINGLE PULSES.

    Inputs:
        -photon_arrivals: Used to determine the locations of the pulse(s)
        -norm: If true, normalize the pulse location value.
    Returns:
        -singleton numpy array with the pulse height value. if the input arrays are
        2d, the returned numpy array has shape (num_training_samples, 1), where the second
        axis contains the pulse height for the associated training sample
    """
    if photon_arrivals.shape[0] == 1 or len(photon_arrivals.shape) == 1:
        return np.argwhere(photon_arrivals == 1) / photon_arrivals.size if norm else np.argwhere(photon_arrivals == 1)
    ret = np.argwhere(photon_arrivals == 1)[:, 1] / photon_arrivals.shape[1] if norm else np.argwhere(photon_arrivals == 1)[:, 1]
    return ret.reshape(photon_arrivals.shape[0], 1)
    
