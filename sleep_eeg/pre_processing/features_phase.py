from scipy.signal import hilbert
import numpy as np

# code from https://github.com/TNTLFreiburg/brainfeatures/blob/master/brainfeatures/feature_generation


def instantaneous_phases(band_signals, axis):
    """ 
    Compute the instantaneous phases of the given band signals along the given axis.
    """
    analytical_signal = hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)


def phase_locking_values(inst_phases):
    """ 
    Compute the phase locking values of the given instantaneous phases.
    """
    (n_windows, n_bands, n_signals, n_samples) = inst_phases.shape
    plvs = []
    for electrode_id1 in range(n_signals):
        # only compute upper triangle of the synchronicity matrix and fill
        # lower triangle with identical values
        # +1 since diagonal is always 1
        for electrode_id2 in range(electrode_id1+1, n_signals):
            for band_id in range(n_bands):
                plv = phase_locking_value2(
                    theta1=inst_phases[:, band_id, electrode_id1],
                    theta2=inst_phases[:, band_id, electrode_id2]
                )
                plvs.append(plv)

    # n_window x n_bands * (n_signals*(n_signals-1))/2
    plvs = np.array(plvs).T
    return plvs

def phase_locking_value2(theta1, theta2):
    """ 
    Compute the phase locking value of the given instantaneous phases.
    """
    # NOTE: band loop, cos, sin, manual/builtin norm won the timing challenge
    # however, this might be different for varying lengths of signals...
    delta = np.subtract(theta1, theta2)
    xs_mean = np.mean(np.cos(delta), axis=-1)
    ys_mean = np.mean(np.sin(delta), axis=-1)
    plv = np.linalg.norm([xs_mean, ys_mean], axis=0)
    return plv