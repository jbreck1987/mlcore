"""
Contains custom YAML constructors used in data creation utilities.
"""

## Author: Josh Breckenridge
## Date: 9/2723

### Constructor Classes ###

class TimestreamConf():
    """
    This simple data class is used as a constructor for YAML when parsing the config file
    when creating data for a specific experiment. It represents the local properties of timestreams
    (I, Q, Phase Response), allowing a completely custom dataset to be generated.
    E.g. The goal is to train a denoising model for I and Q data, so both noisy and
    denoised I and Q timestreams are needed for the dataset.

    Data members:
    -name: The name given to the particular object; Will be used when saving the data to disk in
    a .npz file archive
    -stream_type: Specifies which timestream type this object refers to. This is currently restricted to 'i',
    'q', and 'phase'.
    -normalize: Specifies whether the timestream will be normalized. This toggle does NOT normalize the
    noise.
    -noise_on: Specifies whether to add noise or not to the timestream.
    """
    def __init__(self, name: str, stream_type: str, normalize: bool, noise_on: bool):
        self.name = name
        self.stream_type = stream_type
        self.normalize = normalize
        self.noise_on = noise_on
    
    @classmethod
    def __name__(cls):
        return f'TimestreamConf'