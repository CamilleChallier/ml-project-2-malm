from typing import Any
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
from mne_features.univariate import get_univariate_funcs
from mne_features.bivariate import get_bivariate_funcs
import time

""" 
This file contains the classes used to generate the features from the raw data, features by night
code inspired from https://github.com/TNTLFreiburg/brainfeatures/blob/master/brainfeatures/feature_generation/frequency_feature_generator.py
"""

class FeatureGenerator(ABC):
    """ 
    Abstract class for FeatureGenerator
    """
    def __init__(self):
        pass

    @abstractmethod
    def generate_features(self, raw):
        pass
    
    def get_data(self, raw):
        """ 
        Get the data from the raw object.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        np.ndarray
            The data of the raw object.
        list[str]
            The names of the channels of the raw object.
            """
        features_name = raw.ch_names
        data = raw.get_data()
        return data, features_name
    
    def generate_column_names(self, channel_names, freq_feats, bands, name_method):
        """ 
        Generate the column names of the features.
        
        Parameters
        ----------
        channel_names : list[str]
            The names of the channels of the raw object.
        freq_feats : list[str]
            The names of the features.
        bands : list[list[int]]
            The frequency bands.
        name_method : str
            The name of the method used to generate the features.
        
        Returns
        -------
        pd.Series
            The names of the features
        """
        column_names = pd.Series()
        for feats_names in freq_feats :
            for i in range(len(bands)):
                n = pd.Series(channel_names) + "_" + name_method + "_" + feats_names + "_band_" + str(i)
                column_names = pd.concat([column_names, n]).reset_index(drop=True)
        return column_names
    
    def select_channels_fft_wt(self, raw, channels_to_keep):
        """ 
        Select the channels to keep for the FFT and WT methods.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        channels_to_keep : str
            The channels to keep.
        
        Returns
        -------
        list[str]
            The names of the channels to remove.
        """
    
        list_to_removed = raw.info["ch_names"]
        
        if channels_to_keep == "EEG" or channels_to_keep == "EEG + EMG" :
            list_to_removed = [ele for ele in list_to_removed if not ele.startswith("EEG")]
                
        if channels_to_keep =="EMG" or channels_to_keep == "EEG + EMG" :
            channel_emg = ["Chin1", "Chin2", "TibR", "TibL"]
            list_to_removed = [ele for ele in list_to_removed if ele not in channel_emg] 
            
        print("Channels removed : ", list_to_removed)   
            
        #raw.drop_channels(list_to_removed)   
        return list_to_removed
        
        

class Fourrier(FeatureGenerator) :
    
    
    def __init__(self, bands, sfreq):
        """ 
        Initialize the Fourrier class.
        
        Parameters
        ----------
        bands : list[list[int]]
            The frequency bands.
        sfreq : int
            The sampling frequency.
        
        Returns
        -------
        None
        """
        
        self.freq_feats = sorted([
            feat_func
            for feat_func in dir(features_frequency)
            if not feat_func.startswith('_')])[0:2]
        self.bands = bands
        self.sfreq = sfreq
        
    def convert_with_fft(self, data):
        """ 
        Compute the FFT of the data.
        
        Parameters
        ----------
        data : np.ndarray
            The data to be transformed.
        
        Returns
        -------
        np.ndarray
            The FFT of the data.
        """
        amplitudes = np.abs(np.fft.rfft(data, axis=1))
        amplitudes /= data.shape[-1]
        return amplitudes
        
    def generate_features(self, raw):  
        """ 
        Generate the features.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        pd.DataFrame
            The features."""    
        
        channel_to_removed = self.select_channels_fft_wt(raw, channels_to_keep= "EEG + EMG")
        raw_selected = raw.copy().drop_channels(channel_to_removed)  
        data, channel_names = self.get_data(raw_selected)
        
        self.channels_nb, self.samples_nb = data.shape
        epochs_psds = self.convert_with_fft(data)
        freq_bin_size = self.sfreq / self.samples_nb
        freqs = np.fft.fftfreq(int(self.samples_nb), 1. / self.sfreq)
        
        freq_feats = np.ndarray(shape=(len(self.freq_feats),
                                       len(self.bands), self.channels_nb))
        
        for freq_feat_id, freq_feat_name in enumerate(self.freq_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if freq_feat_name == "power_ratio":
                powers = freq_feats[ self.freq_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                func = getattr(features_frequency, freq_feat_name)
                ratio = func(powers, axis=-2)
                freq_feats[freq_feat_id, :, :] = ratio
            # assumes that "ratio" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            elif freq_feat_name == "spectral_entropy":
                func = getattr(features_frequency, freq_feat_name)
                ratios = freq_feats[:, self.freq_feats.index("power_ratio"),:,:]
                spec_entropy = func(ratios)
                freq_feats[freq_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_frequency, freq_feat_name)
                # amplitudes shape: epochs x electrodes x frequencies
                band_psd_features = np.ndarray(shape=( len(self.bands), self.channels_nb))
                for band_id, (lower, upper) in enumerate(self.bands):
                    lower_bin, upper_bin = (int(lower / freq_bin_size),
                                            int(upper / freq_bin_size))
                    # if upper_bin corresponds to nyquist frequency or higher,
                    # take last available frequency
                    if upper_bin >= len(freqs):
                        upper_bin = len(freqs) - 1
                    band_psds = np.take(epochs_psds,
                                        range(lower_bin, upper_bin), axis=-1)
                    band_psd_features[band_id, :] = func(band_psds, axis=-1)

                freq_feats[freq_feat_id, :, :] = band_psd_features

        freq_feats = freq_feats.reshape(-1)
        
        # column_names = pd.Series()
        # for feats_names in self.freq_feats :
        #     for i in range(len(self.bands)):
        #         n = pd.Series(channel_names) + "_fft_" + feats_names + "_band_" + str(i)
        #         column_names = pd.concat([column_names, n]).reset_index(drop=True)
                
        column_names = self.generate_column_names(channel_names, self.freq_feats, self.bands, "fft")
        
        # predictor = {"Fourrier" : column_names}
        
        return pd.DataFrame([freq_feats], columns=column_names)#, predictor
        
        
        

class WaveletFeatureGenerator(FeatureGenerator):
    """ computes features in the time-frequency domain implemented in
    features_wavelets using wavelet transforms """
    
    
    def __init__(self, sfreq, wavelet, band_limits, domain):
        
        """ 
        Initialize the WaveletFeatureGenerator class.
        
        Parameters
        ----------
        sfreq : int
            The sampling frequency.
        wavelet : str
            The wavelet to use.
        band_limits : list[list[int]]
            The frequency bands.
        domain : str
            The domain to use.
            
        """
        
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
        """
        Continuous wavelet transform using pywt.
        Parameters
        
        ----------
        data : array_like
            Input signal
        scales : array_like
            Scale(s) to use
        wavelet : Wavelet object or name string, or tuple of wavelets
        """
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
                    np.convolve(data, int_psi[j.astype(int)][::-1]))
                d = (coef.size - data.size) / 2.
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            return out
        else:
            raise ValueError("Only dim == 1 supportet")

    def generate_cwt_features(self, raw):
        """ 
        Generate the features.
        
        Parameters
        ----------
        
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        pd.DataFrame
            The generated continuous wavelet features.
        """
        
        data, channel_names = self.get_data(raw)
        self.channels_nb, self.samples_nb = data.shape
        
        scales =[self.freq_to_scale(freq[1], self.wavelet, self.sfreq) for freq in self.band_limits]                                 
        if self.levels is None:
            self.levels = self.generate_level_names(len(scales))
        
        coefficients = np.apply_along_axis(
            func1d=self.pywt_cwt, axis=1, arr=data, scales=scales,
            wavelet=self.wavelet)
        
        coefficients = np.swapaxes(coefficients, 0, 1)
        coefficients = np.abs(coefficients) / self.samples_nb

        cwt_feats = np.ndarray(shape=(len(self.wt_feats),
                                      len(scales), self.channels_nb))
        
        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = cwt_feats[self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                feats = func(powers, axis=-1)
            elif wt_feat_name == "spectral_entropy":
                ratios = cwt_feats[self.wt_feats.index("power_ratio"), :, :]
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(ratios)
            else:
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(coefficients=coefficients, axis=-1)
            cwt_feats[wt_feat_id, :, :] = feats

        cwt_feats = cwt_feats.reshape(-1)
        
        # column_names = pd.Series()
        # for feats_names in self.wt_feats :
        #     for i in range(len(self.bands)):
        #         n = pd.Series(channel_names) + "_fft_" + feats_names + "_band_" + str(i)
        #         column_names = pd.concat([column_names, n]).reset_index(drop=True)
        
        column_names = self.generate_column_names(channel_names, self.wt_feats, self.band_limits, "cwt")
        
        df_cwt = pd.DataFrame([cwt_feats], columns=column_names)
        del cwt_feats, coefficients, scales, data
        
        # predictor = {"cwt" : column_names}
        
        return df_cwt#, predictor

    def generate_dwt_features(self, raw):
        """ 
        Generate the discret wavelet features.
        
        Parameters
        ----------
        
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        pd.DataFrame
            The generated discret wavelet features.
        """
        
        data, channel_names = self.get_data(raw)
        self.channels_nb, self.samples_nb = data.shape
        
        if self.levels is None:
            max_level = dwt_max_level(self.samples_nb,"db4")
            pseudo_freqs = [self.sfreq/2**i for i in range(1, max_level)]
            pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs
                            if pseudo_freq >= 2]
            self.levels = self.generate_level_names(len(pseudo_freqs))
        n_levels = len(self.levels)
        dwt_feats = np.ndarray(
            shape=(len(self.wt_feats), n_levels, self.channels_nb)
        )
        # list of length n_bands of ndarray: x n_epochs x n_channels x
        # n_band_coeffs
        multi_level_coeffs = wavedec(data=data,
                                     wavelet="db4", level=n_levels-1,
                                     axis=1)
        multi_level_coeffs = [np.abs(d) for d in multi_level_coeffs]
        multi_level_coeffs = [d/ self.samples_nb for d in
                              multi_level_coeffs]

        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = dwt_feats[self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                ratios = func(powers)
                dwt_feats[wt_feat_id, :, :] = ratios
            elif wt_feat_name == "spectral_entropy":
                func = getattr(features_wavelets, wt_feat_name)
                ratios = dwt_feats[self.wt_feats.index("power"), :, :]
                spec_entropy = func(ratios)
                dwt_feats[wt_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_wavelets, wt_feat_name)
                # use apply_along_axis here?
                for level_id, level_coeffs in enumerate(multi_level_coeffs):
                    level_coeffs = np.abs(level_coeffs)
                    level_feats = func(coefficients=level_coeffs, axis=1)
                    dwt_feats[wt_feat_id, level_id, :] = level_feats

        dwt_feats = dwt_feats.reshape(-1)
        
        
        # column_names = pd.Series()
        # for feats_names in self.wt_feats :
        #     for i in range(len(self.bands)):
        #         n = pd.Series(channel_names) + "_fft_" + feats_names + "_band_" + str(i)
        #         column_names = pd.concat([column_names, n]).reset_index(drop=True)
        column_names = self.generate_column_names(channel_names, self.wt_feats, self.levels, "dwt")
        
        # predictor = {"dwt" : column_names}
        
        df_dwt = pd.DataFrame([dwt_feats], columns=column_names)
        del dwt_feats, data
        
        # predictor = {"cwt" : column_names}
        
        return df_dwt#, predictor

    def generate_features(self, raw):
        """ generate either cwt or dwt features using pywavelets """
        
        channel_to_removed = self.select_channels_fft_wt(raw, "EEG + EMG")
        raw_selected = raw.copy().drop_channels(channel_to_removed)   
        
        if self.domain == "cwt":
            features= self.generate_cwt_features(raw_selected)
        else:
            assert self.domain == "dwt"
            features = self.generate_dwt_features(raw_selected)
        return features
    
class Sex_feature(FeatureGenerator): 
    """ 
    Add gender as a feature
    """  
    
    def __init__(self):
        pass
    
    def generate_features(self, raw):
        return pd.DataFrame([raw.info["subject_info"]["sex"]], columns=["sex"]) #, {"sex":"sex"}
     
class ECG_features(FeatureGenerator):   
    """ 
    Compute the ECG features
    """
    
    def __init__(self):
        pass

    def generate_features(self, raw) : 
        
        self.sfreq =raw.info["sfreq"]
        
        ecg_signal = raw.copy().pick(["ECG ECG"]).get_data()[0]

        ecg_signal = - (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
      
        threshold =np.sort(ecg_signal)[-int(len(ecg_signal) /  self.sfreq / 60)] *0.7
        p = np.where(ecg_signal > threshold)[0]
        peaks = p[find_peaks(ecg_signal[p])[0]]    

        
        # plt.plot(ecg_signal)
        # plt.plot(peaks, ecg_signal[peaks], "x")
        # plt.plot(np.zeros_like(ecg_signal), "--", color="gray")
        # plt.show()
        
        rr = np.diff(peaks)
        
        results = {}

        hr = 60/(rr/self.sfreq)
        
        # HRV metrics
        if len(hr) > 0:
            results['ECG_Mean_RR_(ms)'] = np.mean(rr)
            results['ECG_STD_RR/SDNN_(ms)'] = np.std(rr)
            results['ECG_Mean_HR_Kubios_beats/min'] = 60000/np.mean(rr)
            results['ECG_Mean_HR_(beats/min)'] = np.mean(hr)
            results['ECG_STD_HR_(beats/min)'] = np.std(hr)
            results['ECG_Min_HR_(beats/min)'] = np.min(hr)
            results['ECG_Max_HR_(beats/min)'] = np.max(hr)
            results['ECG_RMSSD_(ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
            results['ECG_NN50'] = np.sum(np.abs(np.diff(rr)) > 50)*1
            results['ECG_pNN50_(%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / len(rr)
        else : 
            results['ECG_Mean_RR_(ms)'] = np.NaN
            results['ECG_STD_RR/SDNN_(ms)'] = np.NaN
            results['ECG_Mean_HR_Kubios_beats/min'] = np.NaN
            results['ECG_Mean_HR_(beats/min)'] = np.NaN
            results['ECG_STD_HR_(beats/min)'] = np.NaN
            results['ECG_Min_HR_(beats/min)'] = np.NaN
            results['ECG_Max_HR_(beats/min)'] = np.NaN
            results['ECG_RMSSD_(ms)'] = np.NaN
            results['ECG_NN50'] = np.NaN
            results['ECG_pNN50_(%)'] = np.NaN
            
        
        # predictor = {"ecg" : results.keys()}
        
        return pd.DataFrame.from_dict(results, orient='index').T#, predictor
    
    
class Night_lenght(FeatureGenerator):   
    """ 
    Compute the night lenght
    """
    
    def __init__(self):
        pass
    
    def generate_features(self, raw):        
        
        data = raw.get_data()
        lenght = data.shape[1]

        return pd.DataFrame([lenght], columns=["Night_lenght"])#, {"Night_lenght":"Night_lenght"}
        
class TiB_features(FeatureGenerator):   
    """ 
    Compute the EMG TiB features"""
    
    def __init__(self):
        pass
    
    def generate_features(self, raw, ratio =4):
        muscle_signal = raw.copy().get_data(["TibL", "TibR"])
        muscle_signal =  (muscle_signal - muscle_signal.mean()) / muscle_signal.std()
        max = np.max(muscle_signal, axis =1)/ratio
        tot = [np.sum(muscle_signal[i] > max[i]) for i in range(2)]
        
        return pd.DataFrame([tot], columns = ["TibL", "TibR"]) #, {"Tib": ["TibL", "TibR"]}
   
   
def assemble_overlapping_band_limits(non_overlapping_bands: list) -> np.ndarray:
    """ create 50% overlapping frequency bands from non-overlapping bands """
    overlapping_bands = []
    for i in range(len(non_overlapping_bands) - 1):
        band_i = non_overlapping_bands[i]
        overlapping_bands.append(band_i)
        band_j = non_overlapping_bands[i + 1]
        overlapping_bands.append([int((band_i[0] + band_j[0]) / 2),
                                  int((band_i[1] + band_j[1]) / 2)])
    overlapping_bands.append(non_overlapping_bands[-1])
    return np.array(overlapping_bands)

 
class Mne_features (FeatureGenerator):
    
    """ 
    Compute the MNE features
    # finally not used here
    """
    def __init__(self):
        
        self.BAND_LIMITS = np.array(
        [[0, 2], [2, 4],  [4, 8], [8, 13],
         [13, 18],  [18, 24], [24, 30], [30, 49.9]])
        
        self.default_mne_feature_generation_params = {
        'energy_freq_bands__deriv_filt': True,
        'spect_edge_freq__ref_freq': None,
        'spect_edge_freq__edge': None,
        'wavelet_coef_energy__wavelet_name': 'db4',
        'teager_kaiser_energy__wavelet_name': 'db4',
        # bivariate
        'max_cross_corr__include_diag': False,
        'phase_lock_val__include_diag': False,
        'nonlin_interdep__tau': 2,
        'nonlin_interdep__emb': 10,
        'nonlin_interdep__nn': 5,
        'nonlin_interdep__include_diag': False,
        'time_corr__with_eigenvalues': True,
        'time_corr__include_diag': False,
        'spect_corr__with_eigenvalues': True,
        'spect_corr__include_diag': False,
    }

    def generate_features(self, raw):
        
        from mne_features.feature_extraction import extract_features
        
        self.sfreq = raw.info["sfreq"]
        selected_funcs = get_univariate_funcs(self.sfreq)
        selected_funcs.update(get_bivariate_funcs(self.sfreq))
        func_params = self.default_mne_feature_generation_params

        channel_to_removed = self.select_channels_fft_wt(raw, "EEG")
        eeg_raw = raw.copy().drop_channels(channel_to_removed) 
        

        features = extract_features(
        eeg_raw.get_data()[:,np.newaxis, :], self.sfreq, selected_funcs, funcs_params=func_params)
        
        name = []
        for feat in list(self.default_mne_feature_generation_params.keys()):
            if feat == 'pow_freq_bands__freq_bands' or feat == 'energy_freq_bands__freq_bands':
                for i in range(len(self.BAND_LIMITS)):
                    for ch in eeg_raw.info['ch_names']:
                        name.append("mne_" + feat + "_band_" + str(i)+ "_ch_" + ch)
            for ch in eeg_raw.info['ch_names']:
                name.append("mne_" + feat + "_ch_" + ch)
        
        return pd.DataFrame([features.reshape(-1)], columns = name)

class Time_features(FeatureGenerator):
    """ 
    Compute the time features
    """
    def __init__(self):
        """ 
        Initialize the Time_features class.
        """
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
        
        channel_to_removed = self.select_channels_fft_wt(raw, "EEG")
        eeg_raw = raw.copy().drop_channels(channel_to_removed) 
        data, channel_names = self.get_data(eeg_raw)
        self.channels_nb, self.samples_nb = data.shape
        
        self.sfreq = eeg_raw.info["sfreq"]
        
        time_feats = np.ndarray(
            shape=(len(self.time_feats), self.channels_nb))
        for time_feat_id, time_feat_name in enumerate(self.time_feats):
            func = getattr(features_time, time_feat_name)
            time_feats[time_feat_id, :] = func(
                data, -1, Kmax=self.Kmax, n=self.n, T=self.T, Tau=self.Tau,
                DE=self.DE, W=self.W, fs=self.sfreq)

        time_feats = time_feats.reshape(-1)
        
        name = []
        for feat in list(self.time_feats):
            for ch in eeg_raw.info['ch_names']:
                name.append("time_" + feat + "_ch_" + ch)
        return pd.DataFrame([time_feats], columns = name)
    
class Pipeline_features_generator(FeatureGenerator):
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
        df = pd.DataFrame()
        #predictors = {}
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            features = function.generate_features(raw)
            # predictors  = predictors | predictor
            df = pd.concat([df,features], axis=1) 
            del features
            
        return df#, predictor
