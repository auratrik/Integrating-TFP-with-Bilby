#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:06:30 2021

@author: Eric Chassande-Mottin and Cyril Cano
"""

import os
import sys
import copy
import pickle

import numpy as np

import lal
import lalsimulation as lalsim

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from scipy.stats import describe

# from scipy.interpolate import interp1d

DEFAULT_COMPACTBINARY_PARAMETERS = {
    "m1": 20,
    "m2": 20,
    "s1x": 0.0,
    "s1y": 0.0,
    "s1z": 0.0,
    "s2x": 0.0,
    "s2y": 0.0,
    "s2z": 0.0,
}

DEFAULT_RANDOM_COMPACTBINARY_PARAMETERS = {
    "m1_range": [10, 20],
    "m2_range": [10, 20],
    "mass_ratio_range": None,
    "total_mass_range": None,
    "s1_range": [0.4, 0.9],
    "s2_range": [0.4, 0.9],
    "s1": None,
    "s2": None,
    "tilt_range": [0.0, np.pi / 2.0],
}

# Maximum tilt should be smaller than pi/2 otherwise it interferes
# with s_range

DEFAULT_WAVEFORM_PARAMETERS = {
    "distance": 1.0,
    "inclination": 0.0,
    "phi": 0.0,
    "longAscNodes": 0.0,
    "eccentricity": 0.0,
    "deltaT": 1 / 16384.0,
    "f_min": None,
    "approximant": "SEOBNRv4",
    "amplitude0": -1,
    "phase0": -1,
    "lambda1": 0,
    "lambda2": 0,
    "modearray": [[2, 2]],
}

MINIMAL_COMPACTBINARY_PARAMETERS = [
    "m1",
    "m2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
]

ALL_COMPACTBINARY_PARAMETERS = [
    "m1",
    "m2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "total_mass",
    "effective_precession_spin",
    "mass_ratio",
    "effective_spin",
    "chirp_mass",
    "m1_inv",
    "m2_inv",
    "symmetric_mass_ratio",
    "reduced_mass",
]

DEFAULT_TRANSFORMER_PARAMETERS = {"name": "bivar2circ", "unwrap": True}

#DEFAULT_TIMEALIGN_PARAMETERS = {"criterion": "max", "attribute": "a"}

DEFAULT_NOMINALTIME_PARAMETERS = {
    "n_samples": 2 ** 12,
    "t_merge": 1.0,
    "power": 0.35,
    "t_end": 3e-4,
}

# Dict utils ##################################################################


def merge_dict(params, default_params):
    """
    Merge two dictionaries.
    
    default_params is used to complete params and to sort valid keys in params.
    """
    valid_keys = set(params.keys()).intersection(set(default_params.keys()))
    valid_params = {key: params[key] for key in valid_keys}
    missing_keys = set(default_params.keys()) - set(params.keys())
    missing_params = {key: default_params[key] for key in missing_keys}
    p = {**valid_params, **missing_params}
    return p


def remove_keys(params, keys_to_remove):
    for key in keys_to_remove:
        params.pop(key)
    return params


# Create random vector ########################################################


def random_vector(scale_range, tilt_range):
    """
    Generate random vector in R^3.
    
    Isotropic if tilt_range=[0, pi]
    
    Parameters
    ----------
    scale_range
        range for random vector's norm
    tilt_range
        range for tilt between z axis and vector's head, ex : [0,pi/6]
    
    Returns
    -------
    s2x, s2y, s2z
        random vector's coordinates
    """
    # /!\ Is it more stable for PCA to fix S1 on x-z plane ? (It does not change uniformity of configurations)
    scale = np.random.uniform(*scale_range)
    elevation = np.arccos(np.random.uniform(*np.cos(tilt_range)))
    azimuth = np.random.uniform(0, 2 * np.pi)
    s2x = scale * np.sin(elevation) * np.cos(azimuth)
    s2y = scale * np.sin(elevation) * np.sin(azimuth)
    s2z = scale * np.cos(elevation)
    return s2x, s2y, s2z


# CompactBinary ###############################################################


class CompactBinary(object):
    """
    A CompactBinary object characterises a binary formed by two compact objects.
    """

    def __init__(self, **params):
        params = merge_dict(params, DEFAULT_COMPACTBINARY_PARAMETERS)
        for key in params:
            setattr(self, key, params[key])
        self.waveform = None

    def random_parameters(self, **params):
        """
        Generate random parameters and return the CompactBinary object itself.
        
        If total_mass_range = None and mass_ratio_range = None:
            m1 ~ Unif(m1_range), m2 ~ Unif(m2_range)
        If total_mass_range = None and mass_ratio_range != None:
            m1 ~ Unif(m1_range), q ~ Unif(mass_ratio_range) and m2=m1/q
        If total_mass_range != None and mass_ratio_range = None:
            m1 ~ Unif(m1_range), M ~ Unif(total_mass_range) and m2=M-m1
        If total_mass_range != None and mass_ratio_range != None:
            M ~ Unif(total_mass_range), q ~ Unif(mass_ratio_range) and m1=q*M/(q+1), m2=M/(q+1)
        
        if si = None:
            si ~ Unif(si_range)
        else:
            six = siy = 0 and siz ~ Unif(si_range) * si / ||si||
        """
        params = merge_dict(params, DEFAULT_RANDOM_COMPACTBINARY_PARAMETERS)

        # Generate m1 and m2
        if params["total_mass_range"] == None:
            self.m1 = np.random.uniform(*params["m1_range"])
            if params["mass_ratio_range"] == None:
                self.m2 = np.random.uniform(*params["m2_range"])
            # if self.m2 > self.m1:
            # Do we really want m1 >= m2 ??
            # self.m1, self.m2 = self.m2, self.m1
            else:
                q = np.random.uniform(*params["mass_ratio_range"])
                self.m2 = self.m1 / q
        else:
            if params["mass_ratio_range"] == None:
                self.m1 = np.random.uniform(*params["m1_range"])
                m = np.random.uniform(*params["total_mass_range"])
                self.m2 = m - self.m1
            else:
                q = np.random.uniform(*params["mass_ratio_range"])
                m = np.random.uniform(*params["total_mass_range"])
                self.m1 = q * m / (q + 1)
                self.m2 = m / (q + 1)

        # Generate dimensionless spins
        if params["s1"] == None:
            self.s1x, self.s1y, self.s1z = random_vector(
                scale_range=params["s1_range"], tilt_range=params["tilt_range"]
            )
        else:
            if np.linalg.norm(params["s1"]) == 0:
                self.s1x, self.s1y, self.s1z = params["s1"]
            else:
                self.s1x, self.s1y, self.s1z = np.multiply(
                    np.random.uniform(*params["s1_range"])
                    / np.linalg.norm(params["s1"]),
                    params["s1"],
                )
        if params["s2"] == None:
            self.s2x, self.s2y, self.s2z = random_vector(
                scale_range=params["s2_range"], tilt_range=params["tilt_range"]
            )
        else:
            if np.linalg.norm(params["s2"]) == 0:
                self.s2x, self.s2y, self.s2z = params["s2"]
            else:
                self.s2x, self.s2y, self.s2z = np.multiply(
                    np.random.uniform(*params["s2_range"])
                    / np.linalg.norm(params["s2"]),
                    params["s2"],
                )
        return self

    def get_waveform(self, **params):
        """
        Compute time-domain template model of the gravitational wave for a given compact binary.
        Ref: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
        
        Returns a waveform and creates an attribute CompactBinary.waveform.
        
        Parameters
        ----------
        approximant
            model approximant
        amplitude0
            amplitude pN order: -1 means include all
        phase0
            phase pN order: -1 means include all
        modearray
            modes to consider, ex: [[2,2],[2,1]]
        distance
            Distance of the source in Mpc
        inclination
            inclination of the binary
        phi
            initial phase
        longAscNodes
            longitude of ascending nodes, degenerated with the polarization angle psi
        eccentricity
        
        deltaT
            sampling rate in Hz
        f_min
            start frequency in Hz
        
        Returns
        -------
        waveform
            dict with hp, hc and reduced time axis
        
        
        If approx="SEOBNRv4P" or "SEOBNRv4PHM" is used 
        lalsim.SimInspiralChooseTDWaveform get the result from 
        lalsim.SimIMRSpinPrecEOBWaveform which get the result from 
        lalsim.SimIMRSpinPrecEOBWaveformAll.
        
        Here the wiki of SEOBNRv4P(HM):
        https://git.ligo.org/waveforms/reviews/seobnrv4p/-/wikis/home
        
        For an example of how to use lalsim.SimInspiralChooseTDWaveform and 
        lalsim.SimIMRSpinPrecEOBWaveformAll, see :
        https://git.ligo.org/waveforms/reviews/seobnrv4p/blob/master/example_SEOBNRv4P_HM_plotall.ipynb
        
        lalsim.SimInspiralChooseTDWaveform is defined here :
        https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L346
        
        lalsim.SimIMRSpinPrecEOBWaveform is defined here :
        https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRSpinPrecEOBv4P.c#L763
        
        lalsim.SimIMRSpinPrecEOBWaveformAll is defined here :
        https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRSpinPrecEOBv4P.c#L4571
        """

        params = merge_dict(params, DEFAULT_WAVEFORM_PARAMETERS)

        approximant = lalsim.GetApproximantFromString(params["approximant"])

        # Build structure containing variable with default values
        LALparams = lal.CreateDict()
        lal.DictInsertREAL8Value(LALparams, "Lambda1", params["lambda1"])
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(LALparams, params["lambda1"])
        lal.DictInsertREAL8Value(LALparams, "Lambda2", params["lambda2"])
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(LALparams, params["lambda2"])
        lal.DictInsertINT4Value(LALparams, "amplitude0", params["amplitude0"])
        lal.DictInsertINT4Value(LALparams, "phase0", params["phase0"])

        # Include only needed modes
        modearray = lalsim.SimInspiralCreateModeArray()
        for i in range(len(params["modearray"])):
            lalsim.SimInspiralModeArrayActivateMode(modearray, *params["modearray"][i])
        lalsim.SimInspiralWaveformParamsInsertModeArray(LALparams, modearray)

        if params["f_min"] == None:  # Why ??
            params["f_min"] = 200 / (
                self.m1 + self.m2
            )  # freq_min = 20 Hz x (10 Msun/M_total)

        if params["f_min"] > lalsim.EOBHighestInitialFreq(self.m1 + self.m2):
            print(
                "Oops! f_min=",
                params["f_min"],
                " should be smaller than ",
                lalsim.EOBHighestInitialFreq(self.m1 + self.m2),
                ".",
            )

        try:
            output = lalsim.SimInspiralChooseTDWaveform(
                m1=self.m1 * lalsim.lal.MSUN_SI,
                m2=self.m2 * lalsim.lal.MSUN_SI,
                s1x=self.s1x,
                s1y=self.s1y,
                s1z=self.s1z,
                s2x=self.s2x,
                s2y=self.s2y,
                s2z=self.s2z,
                distance=params["distance"] * 1e6 * lalsim.lal.PC_SI,
                inclination=params["inclination"],
                phiRef=params["phi"],
                longAscNodes=params["longAscNodes"],
                eccentricity=params["eccentricity"],
                meanPerAno=0.0,
                deltaT=params["deltaT"],
                f_min=params["f_min"],
                f_ref=params["f_min"],
                params=LALparams,
                approximant=approximant,
            )
            self.waveform = {
                "hp": output[0].data.data,
                "hc": output[1].data.data,
                # TO DO: get_time_axis() to have a time axis with t=0 at merger
                "time": float(output[0].epoch) + np.arange(output[0].data.length) * params["deltaT"]
                # "time": np.arange(len(output[0].data.data)) * params["deltaT"],
            }
            return self.waveform

        except:
            print("Oops! ", sys.exc_info()[0], "occurred.")
            print(self.__dict__)
            return None

    def get_parameters(self, names=ALL_COMPACTBINARY_PARAMETERS):
        """
        Return compact binary parameters.
        
        Parameters
        ----------
        names
            array of names
        
        Returns
        -------
        parameters
            dict
        """
        parameters = dict.fromkeys(names)
        for name in names:
            attr = getattr(self, name)
            if callable(attr):
                parameters[name] = attr()
            else:
                parameters[name] = attr
        return parameters

    def total_mass(self):
        """Compute total mass in solar mass"""
        return self.m1 + self.m2

    def effective_precession_spin(self):
        """
        Compute effective precession spin
        DOI: 10.1103/PhysRevD.91.024043
        """
        q = self.m1 / self.m2
        A1 = 2 + 3 * q / 2
        A2 = 2 + 3 / 2 / q
        Sp = np.max(
            (
                A1 * self.m1 ** 2 * np.linalg.norm((self.s1x, self.s1y)),
                A2 * self.m2 ** 2 * np.linalg.norm((self.s2x, self.s2y)),
            )
        )
        return Sp / A1 / self.m1 ** 2  # norm(S1 /m1**2)**2

    def mass_ratio(self):
        """Compute mass ratio in solar mass"""
        if self.m1 >= self.m2:
            return self.m1 / self.m2
        else:
            return self.m2 / self.m1

    def effective_spin(self):
        """Compute effective spin"""
        return (self.m1 * self.s1z + self.m2 * self.s2z) / self.total_mass()

    def chirp_mass(self):
        """Compute chirp mass"""
        return (self.m1 * self.m2) ** (3 / 5) / (self.m1 + self.m2) ** (1 / 5)

    def reduced_mass(self):
        """Compute reduced mass"""
        return (self.m1 * self.m2) / (self.m1 + self.m2)

    def symmetric_mass_ratio(self):
        """Compute symmetric mass ratio"""
        return (self.m1 * self.m2) / (self.m1 + self.m2) ** 2

    def m1_inv(self):
        """Compute 1/m1"""
        return 1 / self.m1

    def m2_inv(self):
        """Compute 1/m2"""
        return 1 / self.m2


# Transformer #################################################################


class Transformer(object):
    """
    A Transformer object allows to map the data (dict) using various transformations
    """

    def __init__(self, name="identity", **kwargs):
        self.name = name
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def transform(self, **data):
        if self.name == "identity":
            return data
        if self.name == "offset":
            attr = offset(data=data[self.key], delta=self.delta)
            data.update({self.key: attr})
        if (self.name == "resample"):
            # Resample is the only transform that has to be applied to all keys in data
            if "old_time" not in self.__dir__():
                data = resample(new_time=self.new_time, old_time=data["time"], **data)
            else:
                data = resample(new_time=self.new_time, old_time=self.old_time, **data)
        if self.name == "bivar2circ":
            a, phi = bivar2circ(hp=data["hp"], hc=data["hc"], unwrap=self.unwrap)
            data = remove_keys(data, ["hp", "hc"])
            data.update({"a": a, "phi": phi})
        if self.name == "circ2bivar":
            hp, hc = circ2bivar(a=data["a"], phi=data["phi"])
            data = remove_keys(data, ["a", "phi"])
            data.update({"hp": hp, "hc": hc})
        return data

    def inverse(self):
        switcher = {
            "identity": "identity",
            "offset": "offset",
            "resample": "resample",
            "bivar2circ": "circ2bivar",
            "circ2bivar": "bivar2circ",
        }
        new_transform = copy.deepcopy(self)
        new_transform.name = switcher[self.name]
        if new_transform.name == "offset":
            new_transform.delta *= -1
        if new_transform.name == "resample":
            new_transform.old_time, new_transform.new_time = (
                self.new_time,
                self.old_time,
            )
        return new_transform

    def inverse_transform(self, **data):
        new_transform = self.inverse()
        return new_transform.transform(**data)

    def transform_dataset(self, dataset):
        for index in dataset:
            dataset[index] = self.transform(**dataset[index])
        return dataset

    def inverse_transform_dataset(self, dataset):
        for index in dataset:
            dataset[index] = self.inverse_transform(**dataset[index])
        return dataset


def post_process_22(hp, hc, new_params, old_params):
    """Rotate and amplify waveform with respect to physical parameters"""
    assert old_params['inclination']==0
    assert old_params['phi']==0
    
    hp /= (
        old_params["total_mass"]
        / old_params["distance"]
        * lal.SpinWeightedSphericalHarmonic(0,0,-2,2,2).real
    )
    hc /= (
        old_params["total_mass"]
        / old_params["distance"]
        * lal.SpinWeightedSphericalHarmonic(0,0,-2,2,2).real
    )
    
    hp *= new_params["total_mass"] / new_params["distance"]
    hc *= new_params["total_mass"] / new_params["distance"]    
    
    h = (hp + 1j * hc) * lal.SpinWeightedSphericalHarmonic(new_params['inclination'],new_params['phi'],-2,2,2)
    h += (hp - 1j * hc) * lal.SpinWeightedSphericalHarmonic(new_params['inclination'],new_params['phi'],-2,2,-2)
    
    return h.real, h.imag


def offset(data, delta):
    """Apply an offset of delta to data"""
    return data + delta


def resample(new_time, old_time, **data):
    """
    Resample attribute to nominal time grid
    """
    # interpolate from original onto nominal time grid
    for key in data.keys():
        if key != "time" and np.size(data[key]) > 1:
            #             interpolator = interp1d(old_time, data[key], kind='cubic')
            #             data[key] = interpolator(new_time)
            data[key] = np.interp(new_time, old_time, data[key])
    if "time" in data:
        data["time"] = new_time
    return data


def bivar2circ(hp, hc, unwrap=False):
    """
    Compute amplitude and phase from bivariate polarized signal (hp, hc)
    
    Parameters
    ----------
    hp
        real part of the bivariate signal
    hc
        complex part of the bivariate signal
    
    Returns
    -------
    a
        amplitude of the bivariate signal
    phi
        phase of the bivariate signal
    """
    a = np.sqrt(hp ** 2 + hc ** 2)
    phi = np.arctan2(hc, hp)
    if unwrap:
        phi = np.unwrap(phi)
    return a, phi


def circ2bivar(a, phi):
    """
    Compute polarized bivariate signal from amplitude a and phase phi
    
    Parameters
    ----------
    a
        amplitude of the bivariate signal
    phi
        phase of the bivariate signal
    unwrap boolean
        whether or not the phase is unwrap (not used)
    
    Returns
    -------
    hp
        real part of the bivariate signal
    hc
        complex part of the bivariate signal
    """

    phi = np.arctan2(np.sin(phi), np.cos(phi))

    h = a * np.exp(1j * phi)

    hp = np.real(h)
    hc = np.imag(h)

    return hp, hc


# Time grid stuff #############################################################


def get_nominal_time(**params):
    """
    Computes the nominal time axis used to resample all waveforms in the training set.
    """
    params = merge_dict(params, DEFAULT_NOMINALTIME_PARAMETERS)
    t = np.linspace(
        -params["t_merge"] ** params["power"],
        params["t_end"] ** params["power"],
        params["n_samples"],
    )
    return np.sign(t) * np.abs(t) ** (1 / params["power"])


# def resample_dataset(dataset, old_time, new_time):
#    resampler=Transformer(name='resample', new_time=new_time)
#    for index in dataset:                                           # /!\ As dataset is an object, the original dataset is directly
#        dataset[index].update({'time': old_time})                   # modified by this function. We can create a copy to avoid this behaviour
#        dataset[index] = resampler.transform(**dataset[index])
#        dataset[index].pop('time')
#    return dataset


# def particular_time_align(scale=1.0, criterion="max", attribute="a", **data):
#     switcher = {"max": np.argmax}
#     # set time origin to max of ref attribute
#     t_origin = switcher[criterion](data[attribute])
#     data["time"] = (data["time"] - data["time"][t_origin]) * scale
#     return data


# Dataset and Binaryset #######################################################


def random_binaryset(size, seed=None, **random_binary_parameters):
    np.random.seed(seed)
    random_binary_parameters = merge_dict(
        random_binary_parameters, DEFAULT_RANDOM_COMPACTBINARY_PARAMETERS
    )
    binaryset = {
        str(index): CompactBinary()
        .random_parameters(**random_binary_parameters)
        .get_parameters()
        for index in range(size)
    }
    info = {
        "size": size,
        "random_binary_parameters": random_binary_parameters,
        "seed": seed,
    }
    return binaryset, info


def binaryset_from_dataset(dataset):
    return {
        index: CompactBinary(**dataset[index]).get_parameters() for index in dataset
    }


def random_dataset(
    size,
    transformer=Transformer(**DEFAULT_TRANSFORMER_PARAMETERS),
    nominal_time=get_nominal_time(**DEFAULT_NOMINALTIME_PARAMETERS),
#    time_align_parameters=DEFAULT_TIMEALIGN_PARAMETERS,
    parameters_of_interest=ALL_COMPACTBINARY_PARAMETERS,
    waveform_parameters=DEFAULT_WAVEFORM_PARAMETERS,
    random_binary_parameters=DEFAULT_RANDOM_COMPACTBINARY_PARAMETERS,
    seed=None,
):

    random_binary_parameters = merge_dict(
        random_binary_parameters, DEFAULT_RANDOM_COMPACTBINARY_PARAMETERS
    )  # Needed for info output
    binaryset, info = random_binaryset(size=size, seed=seed, **random_binary_parameters)
    info = {
        "size": size,
        "random_binary_parameters": random_binary_parameters,
        "seed": seed,
    }

    return dataset_from_binaryset(
        binaryset=binaryset,
        info=info,
        transformer=transformer,
        nominal_time=nominal_time,
#        time_align_parameters=time_align_parameters,
        parameters_of_interest=parameters_of_interest,
        waveform_parameters=waveform_parameters,
    )


def dataset_from_binaryset(
    binaryset,
    info={},
    transformer=Transformer(**DEFAULT_TRANSFORMER_PARAMETERS),
    nominal_time=get_nominal_time(**DEFAULT_NOMINALTIME_PARAMETERS),
#    time_align_parameters=DEFAULT_TIMEALIGN_PARAMETERS,
    parameters_of_interest=ALL_COMPACTBINARY_PARAMETERS,
    waveform_parameters=DEFAULT_WAVEFORM_PARAMETERS,
):

    waveform_parameters = merge_dict(
        waveform_parameters, DEFAULT_WAVEFORM_PARAMETERS
    )  # Needed for info output

    resampler = Transformer(name="resample", new_time=nominal_time)

    dataset = {}
    for index in binaryset:
        # Get waveform
        bivar = CompactBinary(**binaryset[index]).get_waveform(**waveform_parameters)
        # Apply transformer
        data = transformer.transform(**bivar)
        # Time align with criterion
#        data = particular_time_align(
#            1 / binaryset[index]["total_mass"], **time_align_parameters, **data
#        )
        data["time"] /= binaryset[index]["total_mass"]
        # Resample
        data = resampler.transform(**data)
        #        return data, resampled_data     # debug
        # Remove time attribute
        data.pop("time")
        dataset.update(
            {
                index: {
                    **data,
                    **CompactBinary(**binaryset[index]).get_parameters(
                        parameters_of_interest
                    ),
                }
            }
        )

    info.update(
        {
            "transformer": transformer.__dict__,  # save_obj will also work with the transformer
            "nominal_time": nominal_time,
 #           "time_align_parameters": time_align_parameters,
            "parameters_of_interest": parameters_of_interest,
            "waveform_parameters": waveform_parameters,
        }
    )

    return dataset, info


def add_binary_parameters(dataset, names=ALL_COMPACTBINARY_PARAMETERS):
    for index in dataset:
        dataset[index].update(CompactBinary(**dataset[index]).get_parameters(names))
    return dataset


# Match metric ################################################################


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def overlap(h1, h2):
    """
    Compute the overlap metric between two time-series
    """

    N = next_power_of_2(len(h1) + len(h2) - 1)
    h1f = np.fft.fft(h1 / np.linalg.norm(h1), N)
    h2f = np.fft.fft(h2 / np.linalg.norm(h2), N)

    return np.max(np.abs(np.fft.ifft(h1f * np.conj(h2f))))


def _match_metric(bivar1, bivar2):
    """
    Computes the mismatch metric between two waveforms or attributes
    """
    # compute average for both polarizations:
    h1 = bivar1["hp"] + 1j * bivar1["hc"]
    h2 = bivar2["hp"] + 1j * bivar2["hc"]
    return overlap(h1, h2)


def mismatch_metric(dataset_true, dataset_pred, mode="full"):

    #    # To do before:
    #    # Resample datasets
    #    dataset_true = resampler.transform_dataset(dataset_true, old_time, new_time)
    #    dataset_pred = resampler.transform_dataset(dataset_pred, old_time, new_time)
    #    # Transform datasets to have bivariate signals
    #    dataset_true = transformer.transform_dataset(dataset_true)
    #    dataset_pred = transformer.transform_dataset(dataset_pred)

    metric = np.array(
        [
            1 - _match_metric(dataset_true[index], dataset_pred[index])
            for index in dataset_true
        ]
    )

    if mode == "min":
        return np.min(metric)
    elif mode == "argmin":
        return np.argmin(metric)
    elif mode == "max":
        return np.max(metric)
    elif mode == "argmax":
        return np.argmax(metric)
    elif mode == "mean":
        return np.mean(metric)
    elif mode == "median":
        return np.median(metric)
    elif mode == "describe":
        return describe(metric)
    elif mode == "full":
        return metric


def mismatch_scorer(
    dataset_true, dataset_pred, transformer, resampler
):  # Not used for instance
    # To do before:
    # Create dictionnaries
    dataset_true = dataset_true.to_dict("index")
    dataset_pred = dataset_pred.to_dict("index")
    # Resample datasets
    dataset_true = resampler.transform_dataset(dataset_true)
    dataset_pred = resampler.transform_dataset(dataset_pred)
    # Transform datasets to have bivariate signals
    dataset_true = transformer.transform_dataset(dataset_true)
    dataset_pred = transformer.transform_dataset(dataset_pred)

    return mismatch_metric(dataset_true, dataset_pred)


# Save ########################################################################


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


# Model #######################################################################


class Model(object):
    """
    A Model object provides a regression model of gravitational waveforms
    """

    def __init__(
        self,
        pca,
        regressor,
        features,
        regressor_keys,
        nominal_time,
        total_mass,
        transformer,
        distance,
        inclination,
        phi,
    ):
        self.pca = pca
        self.regressor = regressor
        self.regressor_keys = regressor_keys
        self.features = features
        self.attr_keys = list(pca.keys())
        self.attr_pca_keys = {}
        for key in self.attr_keys:
            self.attr_pca_keys.update(
                {
                    key: [
                        "pca_" + key + "_" + str(comp)
                        for comp in range(self.pca[key].n_components)
                    ]
                }
            )
        self.resampler_nominal2regular = Transformer(
            name="resample", old_time=nominal_time
        )
        self.transformer = transformer
        self.total_mass = total_mass
        self.distance = distance
        self.inclination = inclination
        self.phi = phi
        self.nominal_time = nominal_time
    
    def transform2(self, X):
        check_is_fitted(self)

        X = self._validate_data(
            X, order="F", dtype=FLOAT_DTYPES, reset=False, accept_sparse=("csr", "csc")
        )

        n_samples, n_features = X.shape

        if sparse.isspmatrix_csr(X):
            if self._max_degree > 3:
                return self.transform(X.tocsc()).tocsr()
            to_stack = []
            if self.include_bias:
                to_stack.append(
                    sparse.csc_matrix(np.ones(shape=(n_samples, 1), dtype=X.dtype))
                )
            if self._min_degree <= 1 and self._max_degree > 0:
                to_stack.append(X)
            for deg in range(max(2, self._min_degree), self._max_degree + 1):
                Xp_next = _csr_polynomial_expansion(
                    X.data, X.indices, X.indptr, X.shape[1], self.interaction_only, deg
                )
                if Xp_next is None:
                    break
                to_stack.append(Xp_next)
            if len(to_stack) == 0:
                # edge case: deal with empty matrix
                XP = sparse.csr_matrix((n_samples, 0), dtype=X.dtype)
            else:
                XP = sparse.hstack(to_stack, format="csr")
        elif sparse.isspmatrix_csc(X) and self._max_degree < 4:
            return self.transform(X.tocsr()).tocsc()
        elif sparse.isspmatrix(X):
            combinations = self._combinations(
                n_features=n_features,
                min_degree=self._min_degree,
                max_degree=self._max_degree,
                interaction_only=self.interaction_only,
                include_bias=self.include_bias,
            )
            columns = []
            for combi in combinations:
                if combi:
                    out_col = 1
                    for col_idx in combi:
                        out_col = X[:, col_idx].multiply(out_col)
                    columns.append(out_col)
                else:
                    bias = sparse.csc_matrix(np.ones((X.shape[0], 1)))
                    columns.append(bias)
            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
        else:
            # Do as if _min_degree = 0 and cut down array after the
            # computation, i.e. use _n_out_full instead of n_output_features_.
            XP = np.empty(
                shape=(n_samples, self._n_out_full), dtype=X.dtype, order=self.order
            )

            if self.include_bias:
                XP[:, 0] = 1
                current_col = 1
            else:
                current_col = 0

            if self._max_degree == 0:
                return XP

            # degree 1 term
            XP[:, current_col : current_col + n_features] = X
            index = list(range(current_col, current_col + n_features))
            current_col += n_features
            index.append(current_col)

            # loop over degree >= 2 terms
            for _ in range(2, self._max_degree + 1):
                new_index = []
                end = index[-1]
                for feature_idx in range(n_features):
                    start = index[feature_idx]
                    new_index.append(current_col)
                    if self.interaction_only:
                        start += index[feature_idx + 1] - index[feature_idx]
                    next_col = current_col + end - start
                    if next_col <= current_col:
                        break
                    # XP[:, start:end] are terms of degree d - 1
                    # that exclude feature #feature_idx.
                    np.multiply(
                        XP[:, start:end],
                        X[:, feature_idx : feature_idx + 1],
                        out=XP[:, current_col:next_col],
                        casting="no",
                    )
                    current_col = next_col

                new_index.append(current_col)
                index = new_index

            if self._min_degree > 1:
                n_XP, n_Xout = self._n_out_full, self.n_output_features_
                if self.include_bias:
                    Xout = jnp.empty(
                        shape=(n_samples, n_Xout), dtype=XP.dtype, order=self.order
                    )
                    Xout[:, 0] = 1
                    Xout[:, 1:] = XP[:, n_XP - n_Xout + 1 :]
                else:
                    Xout = XP[:, n_XP - n_Xout :].copy()
                XP = Xout
        return XP

    def transform3(self, X):
        n_samples, n_features = X.shape
        XP = jnp.empty(
            shape=(n_samples, self._n_out_full), dtype=X.dtype
        )
        if self.include_bias:
            XP = XP.at[:,0].set(1)
            current_col = 1
        else:
            current_col = 0

        if self._max_degree == 0:
            return XP

        XP = XP.at[:, current_col : current_col + n_features].set(X)
        index = list(range(current_col, current_col + n_features))
        current_col += n_features
        index.append(current_col)

        for _ in range(2, self._max_degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(n_features):
                start = index[feature_idx]
                new_index.append(current_col)
                if self.interaction_only:
                    start += index[feature_idx + 1] - index[feature_idx]
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                XP = XP.at[:, current_col:next_col].set(jnp.multiply(
                    XP[:, start:end],
                    X[:, feature_idx : feature_idx + 1],
                ))
                current_col = next_col

            new_index.append(current_col)
            index = new_index
        if self._min_degree > 1:
            n_XP, n_Xout = self._n_out_full, self.n_output_features_
            if self.include_bias:
                Xout = jnp.empty(
                    shape=(n_samples, n_Xout), dtype=XP.dtype
                )
                Xout = Xout.at[:,0].set(1)
                Xout = Xout.at[:, 1:].set(XP[:, n_XP - n_Xout + 1 :])
            else:
                Xout = XP[:, n_XP - n_Xout :].copy()
            XP = Xout.copy()
        return XP

    def predict(
        self,
        m1,
        m2,
        s1z,
        s2z,
        inclination=0,
        phi=0,
        distance=1.,
        deltaT=DEFAULT_WAVEFORM_PARAMETERS["deltaT"],
        **kwargs
    ):
        """
        Predict a gravitational waveform from a pre-computed model.
        
        Parameters
        ----------
        distance
            Distance of the source in Mpc
        inclination
            inclination of the binary
        phi
            initial phase
        deltaT
            sampling rate in Hz
        time
            time axis on wich return the waveform
        
        Returns
        -------
        waveform
            dict with hp, hc and time axis (in seconds)
        """
        
        if m2>m1:
            m1, m2 = m2, m1
            s1z, s2z = s2z, s1z
        
        total_mass = m1 + m2

        X = list(
            CompactBinary(
                m1=m1 * self.total_mass / total_mass,
                m2=m2 * self.total_mass / total_mass,
                s1z=s1z,
                s2z=s2z,
            )
            .get_parameters(self.features)
            .values()
        )

        pred = self.regressor.predict([X])[0]

        if kwargs.get('jax') == True:
            # Your code here
            poly = PolynomialFeatures(2)
            X = {key: pred[i] for i, key in enumerate(self.regressor_keys)}
            poly.fit_transform(X)
            poly = PolynomialFeatures(interaction_only=True)
            poly.fit_transform(X)
            y_pred = transform3(poly,X)
        else:
            XQ = {key: pred[i] for i, key in enumerate(self.regressor_keys)}
            poly = PolynomialFeatures(2)
            poly.fit_transform(XQ)
            poly = PolynomialFeatures(interaction_only=True)
            poly.fit_transform(XQ)
            y_pred = transform2(poly,XQ)

        attr_pred = {}
        for key in self.attr_keys:
            # Inverse pca
            attr_pred[key] = self.pca[key].inverse_transform(
                [y_pred[key] for key in self.attr_pca_keys[key]]
            )
            attr_pred[key] += y_pred["offset_" + key]
        
        if 'time' not in list(kwargs.keys()):
            time = np.linspace(
                self.nominal_time[0],
                self.nominal_time[-1],
                int(
                    total_mass
                    * (self.nominal_time[-1] - self.nominal_time[0])
                    / deltaT
                ),
            )
            # nsamples = int((self.nominal_time[-1] - self.nominal_time[0]) \
            #                 * self.total_mass/deltaT)
            # time = self.nominal_time[0] + np.arange(nsamples) * deltaT/total_mass
        else:
            time = kwargs['time']/total_mass

        self.resampler_nominal2regular.new_time = time

        # Resample
        attr_pred = self.resampler_nominal2regular.transform(**attr_pred)
        
        # Transform dataset to have bivariate signals
        h_pred = self.transformer.transform(**attr_pred)

        # Post processing
        h_pred["hp"], h_pred["hc"] = post_process_22(
            h_pred["hp"],
            h_pred["hc"],
            new_params={
                "total_mass": total_mass,
                "distance": distance,
                "inclination": inclination,
                "phi": phi,
            },
            old_params={
                "total_mass": self.total_mass,
                "distance": self.distance,
                "inclination": self.inclination,
                "phi": self.phi,
            },
        )
        h_pred["time"] = time*total_mass
        return h_pred
