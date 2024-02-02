import numpy as _np

# code from https://github.com/TNTLFreiburg/brainfeatures/blob/master/brainfeatures/feature_generation/frequency_feature_generator.py

def bounded_variation(coefficients, axis):
    """
    Compute the bounded variation of the coefficients along the given axis.
    """
    diffs = _np.diff(coefficients, axis=axis)
    abs_sums = _np.sum(_np.abs(diffs), axis=axis)
    max_c = _np.max(coefficients, axis=axis)
    min_c = _np.min(coefficients, axis=axis)
    return _np.divide(abs_sums, max_c - min_c)


def maximum(coefficients, axis):
    """ 
    Compute the maximum of the coefficients along the given axis.
    """
    return _np.max(coefficients, axis=axis)


def mean(coefficients, axis):
    """ 
    Compute the mean of the coefficients along the given axis.
    """
    return _np.mean(coefficients, axis=axis)


def minimum(coefficients, axis):
    """
    Compute the minimum of the coefficients along the given axis.
    """
    return _np.min(coefficients, axis=axis)


def power(coefficients, axis):
    """
    Compute the power of the coefficients along the given axis.
    """
    return _np.sum(coefficients*coefficients, axis=axis)


def power_ratio(powers, axis=-2):
    """
    Compute the power ratio of the coefficients along the given axis.
    """
    ratios = powers / _np.sum(powers, axis=axis, keepdims=True)
    return ratios


def spectral_entropy(ratios, axis=None):
    """
    Compute the spectral entropy of the coefficients along the given axis.
    """
    return -1 * ratios * _np.log(ratios)


def variance(coefficients, axis):
    """
    Compute the variance of the coefficients along the given axis.
    """
    return _np.var(coefficients, axis=axis)