# Inspired by implementation of Manuel Blum and pyeeg library
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew
import numpy as _np

# code from https://github.com/TNTLFreiburg/brainfeatures/blob/master/brainfeatures/feature_generation


def _embed_seq(X, Tau, D):
    # taken from pyeeg
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------

    X
        list

        a time series

    Tau
        integer

        the lag or delay when building embedding sequence

    D
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-D list

        embedding matrix built

    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.],
           [ 3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.],
           [ 5.,  6.,  7.,  8.]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[ 0.,  2.,  4.],
           [ 1.,  3.,  5.],
           [ 2.,  4.,  6.],
           [ 3.,  5.,  7.],
           [ 4.,  6.,  8.]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[ 0.],
           [ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.],
           [ 6.],
           [ 7.],
           [ 8.]])

    """
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return _np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def energy(epochs, axis, **kwargs):
    return _np.mean(epochs*epochs, axis=axis)


def fisher_information(epochs, axis, **kwargs):
    def fisher_info_1d(a, tau, de):
        # taken from pyeeg improvements
        r"""
        Compute the Fisher information of a signal with embedding dimension "de" and delay "tau" [PYEEG]_.
        Vectorised (i.e. faster) version of the eponymous PyEEG function.
        :param a: a one dimensional floating-point array representing a time series.
        :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
        :param tau: the delay
        :type tau: int
        :param de: the embedding dimension
        :type de: int
        :return: the Fisher information, a scalar
        :rtype: float
        """

        mat = _embed_seq(a, tau, de)
        W = _np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)  # normalize singular values
        FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]
        return _np.sum(FI_v)

    tau = kwargs["Tau"]
    de = kwargs["DE"]
    return _np.apply_along_axis(fisher_info_1d, axis, epochs, tau, de)


def hjorth_activity(epochs, axis, **kwargs):
    return _np.var(epochs, axis=axis)


def hjorth_complexity(epochs, axis, **kwargs):
    diff1 = _np.diff(epochs, axis=axis)
    diff2 = _np.diff(diff1, axis=axis)
    sigma1 = _np.std(diff1, axis=axis)
    sigma2 = _np.std(diff2, axis=axis)
    return _np.divide(_np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))


def hjorth_mobility(epochs, axis, **kwargs):
    diff = _np.diff(epochs, axis=axis)
    sigma0 = _np.std(epochs, axis=axis)
    sigma1 = _np.std(diff, axis=axis)
    return _np.divide(sigma1, sigma0)


def _hjorth_parameters(epochs, axis, **kwargs):
    activity = _np.var(epochs, axis=axis)
    diff1 = _np.diff(epochs, axis=axis)
    diff2 = _np.diff(diff1, axis=axis)
    sigma0 = _np.std(epochs, axis=axis)
    sigma1 = _np.std(diff1, axis=axis)
    sigma2 = _np.std(diff2, axis=axis)
    mobility = _np.divide(sigma1, sigma0)
    complexity = _np.divide(_np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))
    return activity, complexity, mobility


def kurtosis(epochs, axis, **kwargs):
    return _kurt(epochs, axis=axis, bias=False)


def line_length(epochs, axis, **kwargs):
    return _np.sum(_np.abs(_np.diff(epochs)), axis=axis)


def maximum(epochs, axis, **kwargs):
    return _np.max(epochs, axis=axis)


def mean(epochs, axis, **kwargs):
    return _np.mean(epochs, axis=axis)


def median(epochs, axis, **kwargs):
    return _np.median(epochs, axis=axis)


def minimum(epochs, axis, **kwargs):
    return _np.min(epochs, axis=axis)


def non_linear_energy(epochs, axis, **kwargs):
    return _np.apply_along_axis(lambda epoch: _np.mean((_np.square(epoch[1:-1]) - epoch[2:] * epoch[:-2])), axis, epochs)

def skewness(epochs, axis, **kwargs):
    return _skew(epochs, axis=axis, bias=False)


def svd_entropy(epochs, axis, **kwargs):
    def svd_entropy_1d(X, Tau, DE, W):
        # taken from pyeeg
        """Compute SVD Entropy from either two cases below:
        1. a time series X, with lag tau and embedding dimension dE (default)
        2. a list, W, of normalized singular values of a matrix (if W is provided,
        recommend to speed up.)

        If W is None, the function will do as follows to prepare singular spectrum:

            First, computer an embedding matrix from X, Tau and DE using pyeeg
            function embed_seq():
                        M = embed_seq(X, Tau, DE)

            Second, use scipy.linalg function svd to decompose the embedding matrix
            M and obtain a list of singular values:
                        W = svd(M, compute_uv=0)

            At last, normalize W:
                        W /= sum(W)

        Notes
        -------------

        To speed up, it is recommended to compute W before calling this function
        because W may also be used by other functions whereas computing it here
        again will slow down.
        """

        if W is None:
            Y = _embed_seq(X, Tau, DE)
            W = _np.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values

        return -1 * sum(W * _np.log(W))
    Tau = kwargs["Tau"]
    DE = kwargs["DE"]
    W = kwargs["W"]
    return _np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE, W)


def zero_crossing(epochs, axis, **kwargs):
    e = 0.01
    norm = epochs - epochs.mean()
    return _np.apply_along_axis(lambda epoch: _np.sum((epoch[:-5] <= e) & (epoch[5:] > e)), axis, norm)


def zero_crossing_derivative(epochs, axis, **kwargs):
    e = 0.01
    diff = _np.diff(epochs)
    norm = diff-diff.mean()
    return _np.apply_along_axis(lambda epoch: _np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), axis, norm)