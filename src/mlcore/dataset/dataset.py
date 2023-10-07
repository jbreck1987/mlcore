"""
This module contains functions/classes that can be used to generate and manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import torch
from itertools import repeat
from functools import reduce 
from warnings import warn
import datetime as dt
import pathlib
from typing import Any, Tuple
from collections.abc import Iterable
from types import FunctionType
import copy
from inspect import getmembers, isclass, getfullargspec

import ruamel.yaml as yaml

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream
from mkidreadoutanalysis.resonator import Resonator, FrequencyGrid, RFElectronics, ReadoutPhotonResonator, LineNoise
import mlcore.dataset.yaml_constructors as yaml_constructors
from mlcore.dataset.yaml_constructors import TimestreamConf

#### YAML Handling ####

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

### DATASET GENERATION FUNCTIONS ###
#----------------------------------#

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


def save_training_data(in_array: Any, dir: pathlib.Path, filename: str, labels: Tuple[str]) -> None:
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


def load_training_data(filepath: pathlib.Path, labels: Tuple[str]) -> Tuple[np.ndarray]:
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


def photon_generator(data_conf: dict) -> np.ndarray:
    """
    Takes an input data configuration dict and return a flattened array of photon arrival times.
    """
    # Get the arguments from the make_arrivals function
    # that match the keys in data_conf to be expanded.
    # This is to reduce verbosity.
    kwargs = [key for key in getfullargspec(make_arrivals)[0] if key in data_conf['general'].keys()]
    return make_arrivals(shuffle=True,
                         lam=data_conf['quasiparticle']['cps'] / data_conf['general']['fs'],
                         flatten=True,
                         **{kwarg: data_conf['general'][kwarg] for kwarg in kwargs})


def iqp_generator(timestream_confs: Iterable[TimestreamConf],
                  readouts: Iterable[ReadoutPhotonResonator],
                  phase_transform: str) -> tuple[tuple[TimestreamConf, np.ndarray]]:
    """
    Generates I, Q, and Phase Response timestreams based on the passed TimestreamConf objects and
    the passed ReadoutPhotonResonator objects. Phase transform should be either 'basic' or 'nicks'.
    """
    # Auxiliary function to be mapped over the TimestreamConf objects
    def f(res: ReadoutPhotonResonator, tconf: TimestreamConf):
        # Create shallow copy of the readout object with noise toggled
        # appropriately. Shallow copy due to mutation of noise boolean.
        res = copy.copy(res)
        res.noise_on = tconf.noise_on

        # Check stream type and normalization condition
        if tconf.stream_type == 'i':
            if tconf.normalize:
                return tconf, res.normalized_iq.real
            return tconf, res.iq_response.real
        if tconf.stream_type == 'q':
            if tconf.normalize:
                return tconf, res.normalized_iq.imag
            return tconf, res.iq_response.imag
        if tconf.stream_type == 'phase':
            if phase_transform == 'basic':
                return tconf, res.basic_coordinate_transformation()[0]
            return tconf, res.nick_coordinate_transformation()[0]
        
    # Auxiliary function for the second map over the readout objects
    def g(res: ReadoutPhotonResonator):
        return tuple(map(lambda x: f(res, x), timestream_confs))

    # The map over all the readout objects will result in a 3-level nested tuple,
    # where we want to return a 2-level nested tuple. This comprehension will unpack
    # the second level.
    ret = []
    _ = [tuple(map(ret.append, x)) for x in map(g, readouts)]
    return tuple(ret)
        

def build_qp_timestreams(data_conf: dict, photon_arrivals: np.ndarray) -> tuple[QuasiparticleTimeStream]:
    """
    Builds the QuasiparticleTimestream objects that are appropriate based on the generated photon
    arrivals and parameters in the data configuration file. This is necessary due to the coupling
    between the QuasiparticleTimestream object and the ReadoutPhotonResonator object. This function
    looks specifically at the quasiparticle shift magnitudes in the data configuration dict to determine
    how many objects to return in the container (one for each magnitude).
    """
    qpt = QuasiparticleTimeStream(fs=data_conf['general']['fs'],
                                  ts=photon_arrivals.size / data_conf['general']['fs'])
    qpt.photon_arrivals = photon_arrivals
    
    # Want to return a tuple of objects with the given magnitudes in the 
    # data conf dict with the same photon arrivals.
    def f(mag: float):
        # Make shallow copy since we will only be manipulating the magnitude
        # data member and eventually throwing these objects away.
        temp_qpt = copy.copy(qpt)

        # Generate the qp shift based on the passed in magnitude
        temp_qpt.gen_quasiparticle_pulse(magnitude=mag)
        _ = temp_qpt.populate_photons()

        return temp_qpt
    
    return tuple(map(f, data_conf['quasiparticle']['qp_shift_magnitudes']))


def build_readouts(data_conf: dict, qpts: Iterable[QuasiparticleTimeStream]) -> tuple[ReadoutPhotonResonator]:
    """
    Builds the ReadoutPhotonResonator objects necessary to generate the I, Q, and Phase Response timestreams.
    The number of readout objects generated is calculated by multiplying the number of passed in QuasiparticleTimeStream
    objects by the number of white_noise_scale values in the data config dict.
    """
    # Define the ancillary objects needed to create the ReadoutPhotonResonator object.
    # These will be shared among all objects generated (only the quasiparticles will differ.)
    # Decompose the configuration variable dict into sub-components.
    res_conf = data_conf['resonator']
    f_grid_conf = data_conf['freq_grid']
    noise_conf = data_conf['noise']

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
    # White noise scale is a variable that will be unique to each readout object.
    # Will need to map over all the passed in values of white noise to generate
    # unique RFElectronics objects.
    def f(noise_scale: float):
        rf = RFElectronics(gain=tuple(noise_conf['rf_electronics']['gain']),
                            phase_delay=noise_conf['rf_electronics']['phase_delay'],
                            white_noise_scale=noise_scale,
                            line_noise=line_noise,
                            cable_delay=noise_conf['rf_electronics']['cable_delay']
                            )
        return rf
    rfs = tuple(map(f, noise_conf['rf_electronics']['white_noise_scale']))

    # Lower level aux function to map over all the RFElectronics (E.g. white noise scales)
    def g(qpt: QuasiparticleTimeStream, rf: RFElectronics):
        return ReadoutPhotonResonator(res, qpt, f_grid, rf)
    
    # Upper level aux function to map over all the QuasiparticleTimeStream
    # objects (E.g. Quasiparticle shift magnitudes).
    def h(qpt: QuasiparticleTimeStream):
        return tuple(map(lambda x: g(qpt, x), rfs))

    # The map over all the quasiparticle objects will result in a nested
    # tuple, but we want to return a flat tuple. This comprehension unpacks
    # the second level tuples into a flat list, which gets returned as a tuple.
    ret = []
    _ = [tuple(map(ret.append, x)) for x in map(h, qpts)]
    return tuple(ret)


def data_writer(data_conf: dict,
                out_path: pathlib.Path,
                out_name: str,
                streams: tuple[tuple[TimestreamConf, np.ndarray]]) -> None:
    """
    Takes the input data structure from the iqp_generator function and writes the arrays to the given
    location with the given name. The streams will be saved as a .npz file in the format defined in the
    save_training_data function defined above.
    """
    
    # First need to reshape the flattened input arrays to be in training sample form;
    # shape is (num_samples, window_size). 
    shape = (data_conf['general']['num_samples'], data_conf['general']['window_size'])
    streams = [(stream[0].name, stream[1].reshape(shape)) for stream in streams]

    # Now need to create a function that stacks all the individual streams based on stream name in the first dimension.
    # Can map over the reshaped iterable and stack the name-like streams in a hash table.
    def f(d: dict, x: tuple[str, np.ndarray]) -> None:
        # Stack on already existing array
        if x[0] in d.keys():
            d[x[0]] = np.vstack((d[x[0]], x[1]))
        # Add new array to hash table otherwise
        else:
            d[x[0]] = x[1]
        return None
    
    stacked = dict()
    _ = tuple(map(lambda x: f(stacked, x), streams))

    # Finally, need to stack all the arrays into one large array along the name axis
    # to get in the correct format for the save_training_data function.
    keys = list(stacked.keys())
    keys.sort()
    out_arr = np.stack([stacked[key] for key in keys], axis=1)

    # Save the training data
    save_training_data(out_arr, out_path, out_name, labels=tuple(keys))


def make_dataset(data_conf_path: pathlib.Path, out_path: pathlib.Path, out_name: str) -> None:
    """
    This is a pipeline function that utilizes the functions defined above to create a dataset
    based on the data configuration yaml file. The dataset format is a .npz file containing different
    "streams" packaged as numpy arrays of shape (num_training_samples, stream length). The possible streams
    are I, Q, Phase Response and (optionally, to be defined) the photon arrivals and quasiparticle density shift
    for each photon. The streams are defined as "TimestreamConf" tags within the data configuration yaml file,
    see the example for usage.
    """

    # Load the yaml config file
    print(f'Loading configuration data...')
    config_data = data_config_loader(data_conf_path)

    # Generate the photon arrivals that will be used to build quasiparticle shifts
    print(f'Generating photons...')
    photons = photon_generator(config_data)

    # Build the QuasiparticleTimeStream object necessary for building the 
    # noise and readout objects
    print(f'Generating quasiparticle timestreams...')
    qpts = build_qp_timestreams(config_data, photons)

    # Create noise and readout objects that will be used to generate the streams
    print(f'Generating noise and readout objects...')
    readouts = build_readouts(config_data, qpts)

    # Generate the required streams
    print(f'Generating streams...')
    streams = iqp_generator(config_data['timestreams'], readouts, config_data['coordinates']['coord_transform'])

    # Write the streams to disk
    print(f'Writing streams to disk...')
    data_writer(config_data, out_path, out_name, streams)
    print(f'Write complete!')


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
    
