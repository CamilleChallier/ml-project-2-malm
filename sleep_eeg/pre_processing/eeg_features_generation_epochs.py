import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from sleep_eeg.utils import *
from sleep_eeg.pre_processing.eeg_preprocessing import *
import pywt
from pywt import wavedec, dwt_max_level, wavelist
from pywt._extensions._pywt import (DiscreteContinuousWavelet,
    ContinuousWavelet, Wavelet, _check_dtype)
from pywt._functions import (integrate_wavelet, scale2frequency,
    central_frequency)
from sleep_eeg.pre_processing import features_frequency, features_wavelets, features_time
from scipy.signal import find_peaks

""" 
Generate features from raw data per epochs
same code as eeg_features_generation.py but for epochs, see eeg_features_generation.py for docstring and more details
"""

class FeatureGenerator_epochs(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_features(self, raw):
        pass
    
    def get_data(self, raw):
        features_name = raw.ch_names
        data = raw.get_data()
        return data, features_name
    
    def generate_column_names(self, channel_names, freq_feats, bands, name_method):
        column_names = pd.Series()
        for feats_names in freq_feats :
            for i in range(len(bands)):
                n = pd.Series(channel_names) + "_" + name_method + "_" + feats_names + "_band_" + str(i)
                column_names = pd.concat([column_names, n]).reset_index(drop=True)
        return column_names
    
    def select_channels_fft_wt(self, raw, channels_to_keep):
    
        list_to_removed = raw.info["ch_names"]
        
        if channels_to_keep == "EEG" or channels_to_keep == "EEG + EMG" :
            list_to_removed = [ele for ele in list_to_removed if not ele.startswith("EEG")]
                
        if channels_to_keep =="EMG" or channels_to_keep == "EEG + EMG" :
            channel_emg = ["Chin1", "Chin2", "TibR", "TibL"]
            list_to_removed = [ele for ele in list_to_removed if ele not in channel_emg] 
            
        print("Channels removed : ", list_to_removed)   
            
        #raw.drop_channels(list_to_removed)   
        return list_to_removed
        
class Fourrier_epochs(FeatureGenerator_epochs) :
    # code from https://github.com/TNTLFreiburg/brainfeatures/blob/master/brainfeatures/feature_generation/frequency_feature_generator.py
    
    def __init__(self, bands, sfreq):
        
        self.freq_feats = sorted([
            feat_func
            for feat_func in dir(features_frequency)
            if not feat_func.startswith('_')])[0:2]
        self.bands = bands
        self.sfreq = sfreq
        
    def convert_with_fft(self, weighted_epochs):
        epochs_amplitudes = np.abs(np.fft.rfft(weighted_epochs, axis=2))
        epochs_amplitudes /= weighted_epochs.shape[-1]
        return epochs_amplitudes
        
    def generate_features(self, raw):
        
        # channel_to_removed = self.select_channels_fft_wt(raw, channels_to_keep= "EEG + EMG")
        # raw_selected = raw.copy().drop_channels(channel_to_removed)  
        # weighted_epochs, channel_names = self.get_data(raw_selected)
        weighted_epochs = raw

        (n_epochs, n_elecs, n_samples_in_epoch) = weighted_epochs.shape
        epochs_psds = self.convert_with_fft(weighted_epochs)
        freq_bin_size = self.sfreq / n_samples_in_epoch
        freqs = np.fft.fftfreq(int(n_samples_in_epoch), 1. / self.sfreq)

        # extract frequency bands and generate features
        # n_epochs x n_elecs x n_bands x n_feats
        freq_feats = np.ndarray(shape=(n_epochs, len(self.freq_feats),
                                       len(self.bands), n_elecs))
        for freq_feat_id, freq_feat_name in enumerate(self.freq_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if freq_feat_name == "power_ratio":
                powers = freq_feats[:, self.freq_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                func = getattr(features_frequency, freq_feat_name)
                ratio = func(powers, axis=-2)
                freq_feats[:, freq_feat_id, :, :] = ratio
            # assumes that "ratio" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            elif freq_feat_name == "spectral_entropy":
                func = getattr(features_frequency, freq_feat_name)
                ratios = freq_feats[:, self.freq_feats.index("power_ratio"),:,:]
                spec_entropy = func(ratios)
                freq_feats[:, freq_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_frequency, freq_feat_name)
                # amplitudes shape: epochs x electrodes x frequencies
                band_psd_features = np.ndarray(shape=(n_epochs, len(self.bands),
                                                      n_elecs))
                for band_id, (lower, upper) in enumerate(self.bands):
                    lower_bin, upper_bin = (int(lower / freq_bin_size),
                                            int(upper / freq_bin_size))
                    # if upper_bin corresponds to nyquist frequency or higher,
                    # take last available frequency
                    if upper_bin >= len(freqs):
                        upper_bin = len(freqs) - 1
                    band_psds = np.take(epochs_psds,
                                        range(lower_bin, upper_bin), axis=-1)
                    band_psd_features[:, band_id, :] = func(band_psds, axis=-1)

                freq_feats[:, freq_feat_id, :, :] = band_psd_features
                
        freq_feats = freq_feats.reshape(n_epochs, -1, n_elecs)

        return freq_feats      
             

class WaveletFeatureGenerator_epochs(FeatureGenerator_epochs):
    """ computes features in the time-frequency domain implemented in
    features_wavelets using wavelet transforms """
    
    
    def __init__(self, sfreq, wavelet, band_limits, domain):
        
        self.wt_feats = sorted([
            feat_func
            for feat_func in dir(features_wavelets)
            if not feat_func.startswith('_')])[0:2]
        assert wavelet in wavelist(), "unknown wavelet {}".format(wavelet)
        if wavelet in wavelist(kind="discrete"):
            self.wavelet = Wavelet(wavelet)
        else:
            self.wavelet = ContinuousWavelet(wavelet)
        self.sfreq = sfreq
        self.band_limits = band_limits
        self.levels = None
        self.domain = domain
        

    def freq_to_scale(self, freq, wavelet, sfreq):
        """ compute cwt scale to given frequency
        see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
        central_freq = central_frequency(wavelet)
        assert freq > 0, "freq smaller or equal to zero!"
        scale = central_freq / freq
        return scale * sfreq
    
    def generate_level_names(self, n_levels):
        """ name the levels of the wavelet transformation, i.e. a1, d1, d0 """
        level_names = ["a" + str(n_levels-1)]
        for i in range(n_levels)[:-1][::-1]:
            level_names.append("d" + str(i))
        return level_names
    
    def pywt_cwt(self, data, scales, wavelet):
        # accept array_like input; make a copy to ensure a contiguous array
        dt = _check_dtype(data)
        data = np.array(data, dtype=dt)
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
        if np.isscalar(scales):
            scales = np.array([scales])
        if data.ndim == 1:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
            for i in np.arange(np.size(scales)):
                precision = 10
                int_psi, x = integrate_wavelet(wavelet, precision=precision)
                step = x[1] - x[0]
                j = np.floor(np.arange(
                    scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
                if np.max(j) >= np.size(int_psi):
                    j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
                coef = - np.sqrt(scales[i]) * np.diff(
                    np.convolve(data, int_psi[j.astype(np.int)][::-1]))
                d = (coef.size - data.size) / 2.
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            return out
        else:
            raise ValueError("Only dim == 1 supportet")

    def generate_cwt_features(self, raw):
        
        # weighted_windows, channel_names = self.get_data(raw)
        weighted_windows = raw
        
        (n_windows, n_elecs, n_samples_in_window) = weighted_windows.shape
        scales = self.freqs_to_scale(self.band_limits, self.wavelet,
                                     self.sfreq)
        if self.levels is None:
            self.levels = self.generate_level_names(len(scales))
        coefficients = np.apply_along_axis(
            func1d=self.pywt_cwt, axis=2, arr=weighted_windows, scales=scales,
            wavelet=self.wavelet)
        # n_windows x n_elecs x n_levels x n_coefficients
        coefficients = np.swapaxes(coefficients, 1, 2)
        coefficients = np.abs(coefficients) / weighted_windows.shape[-1]

        cwt_feats = np.ndarray(shape=(n_windows, len(self.wt_feats),
                                      len(scales), n_elecs))
        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = cwt_feats[:, self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                feats = func(powers, axis=-1)
            elif wt_feat_name == "spectral_entropy":
                ratios = cwt_feats[:, self.wt_feats.index("power_ratio"), :, :]
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(ratios)
            else:
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(coefficients=coefficients, axis=-1)
            cwt_feats[:, wt_feat_id, :, :] = feats
            
        cwt_feats = cwt_feats.reshape(n_windows, -1, n_elecs)

        return cwt_feats

        # column_names = self.generate_column_names(channel_names, self.wt_feats, self.band_limits, "cwt")
        
        # df_cwt = pd.DataFrame([cwt_feats], columns=column_names)
        # del cwt_feats, coefficients, scales, data
        
        # # predictor = {"cwt" : column_names}
        
        # return df_cwt#, predictor

    def generate_dwt_features(self, raw):
        
        # weighted_windows, channel_names = self.get_data(raw)
        weighted_windows = raw
        (n_windows, n_elecs, n_samples_in_window) = weighted_windows.shape
        if self.levels is None:
            max_level = dwt_max_level(n_samples_in_window, self.wavelet)
            pseudo_freqs = [self.sfreq/2**i for i in range(1, max_level)]
            pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs
                            if pseudo_freq >= 2]
            self.levels = self.generate_level_names(len(pseudo_freqs))
        n_levels = len(self.levels)
        dwt_feats = np.ndarray(
            shape=(n_windows, len(self.wt_feats), n_levels, n_elecs)
        )
        # list of length n_bands of ndarray: x n_epochs x n_channels x
        # n_band_coeffs
        multi_level_coeffs = wavedec(data=weighted_windows,
                                     wavelet=self.wavelet, level=n_levels-1,
                                     axis=2)
        multi_level_coeffs = [np.abs(d) for d in multi_level_coeffs]
        multi_level_coeffs = [d/weighted_windows.shape[-1] for d in
                              multi_level_coeffs]

        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = dwt_feats[:, self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                ratios = func(powers)
                dwt_feats[:, wt_feat_id, :, :] = ratios
            elif wt_feat_name == "spectral_entropy":
                func = getattr(features_wavelets, wt_feat_name)
                ratios = dwt_feats[:, self.wt_feats.index("power"), :, :]
                spec_entropy = func(ratios)
                dwt_feats[:, wt_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_wavelets, wt_feat_name)
                # use apply_along_axis here?
                for level_id, level_coeffs in enumerate(multi_level_coeffs):
                    level_coeffs = np.abs(level_coeffs)
                    level_feats = func(coefficients=level_coeffs, axis=2)
                    dwt_feats[:, wt_feat_id, level_id, :] = level_feats
        
        dwt_feats = dwt_feats.reshape(n_windows, -1, n_elecs)
        
        # column_names = pd.Series()
        # for feats_names in self.wt_feats :
        #     for i in range(len(self.bands)):
        #         n = pd.Series(channel_names) + "_fft_" + feats_names + "_band_" + str(i)
        #         column_names = pd.concat([column_names, n]).reset_index(drop=True)
        return dwt_feats
        # column_names = self.generate_column_names(channel_names, self.wt_feats, self.levels, "dwt")
        
        # # predictor = {"dwt" : column_names}
        
        # df_dwt = pd.DataFrame([dwt_feats], columns=column_names)
        # del dwt_feats, data
        
        # # predictor = {"cwt" : column_names}
        
        # return df_dwt#, predictor

    def generate_features(self, raw):
        """ generate either cwt or dwt features using pywavelets """
        
        # channel_to_removed = self.select_channels_fft_wt(raw, "EEG + EMG")
        # raw_selected = raw.copy().drop_channels(channel_to_removed)   
        raw_selected = raw
        
        if self.domain == "cwt":
            features= self.generate_cwt_features(raw_selected)
        else:
            assert self.domain == "dwt"
            features = self.generate_dwt_features(raw_selected)
        return features
    
    
class Time_features_epochs(FeatureGenerator_epochs):
    def __init__(self):
        self.time_feats = sorted([
            feat_func
            for feat_func in dir(features_time)
            if not feat_func.startswith('_')])

        # for computation of pyeeg features
        self.Kmax = 3
        self.n = 4
        self.T = 1
        self.Tau = 4
        self.DE = 10
        self.W = None
        
    def generate_features(self, raw):
        
        # channel_to_removed = self.select_channels_fft_wt(raw, "EEG")
        # eeg_raw = raw.copy().drop_channels(channel_to_removed) 
        # windows, channel_names = self.get_data(eeg_raw)
        windows = raw
        self.n_windows, self.channels_nb, self.samples_nb = windows.shape
        
        self.sfreq = 100 #eeg_raw.info["sfreq"]
        
        time_feats = np.ndarray(
            shape=(self.n_windows, len(self.time_feats), self.channels_nb))
        for time_feat_id, time_feat_name in enumerate(self.time_feats):
            func = getattr(features_time, time_feat_name)
            time_feats[:, time_feat_id, :] = func(
                windows, -1, Kmax=self.Kmax, n=self.n, T=self.T, Tau=self.Tau,
                DE=self.DE, W=self.W, fs=self.sfreq)
        
        # time_feats = time_feats.reshape(self.n_windows, -1)
        return time_feats
    
    
class Pipeline_features_generator_epochs(FeatureGenerator_epochs):
    """
    Implementation of a pipeline of PreProcessors
    """

    def __init__(self, steps: list) -> None:
        self.functions = []
        for step, arg in steps:
            # initialize
            if arg != None:
                self.functions.append(step(**arg))
            else:
                self.functions.append(step())

    def generate_features(
        self, raw):
        """
        Fit and transform the data with all steps of the pipeline
        """
        data = []
        #predictors = {}
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            features = function.generate_features(raw)
            print(features.shape)
            # predictors  = predictors | predictor
            data.append(features)
            del features
            
        data = np.concatenate(data, axis=1)
            
        return data#, predictor