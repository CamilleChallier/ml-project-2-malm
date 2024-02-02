import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Union
import mne


class PreProcessor(ABC):
    """
    Abstract class for PreProcessors
    """

    def __init__(self) -> None:
        pass
    
    def transform(
        self, raw) :
        pass

    def fit_transform(
        self, raw):
        return self.transform(raw)
    
class Renamer(PreProcessor):
    def __init__(self) -> None:
        pass
    
    def transform(
        self, raw) :
        """ 
        Rename the channels of the raw object to be compatible with all patient and standart MNE names.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        mne.io.Raw
            The preprocessed raw object.
        """
        raw.rename_channels(self.new_name)
        return raw
        
    def fit_transform(self, raw):
        """ 
        Find and Rename the channels of the raw object.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        mne.io.Raw
            The preprocessed raw object.
        """
        self.new_name = {}

        for channel in raw.info["ch_names"] :
            if channel.startswith("M1") or channel.startswith("M2") or channel.startswith("E1") or channel.startswith("E2") or channel.startswith("A1") or channel.startswith("A2") or channel.startswith("Fp1") or channel.startswith("Fp2") or channel.startswith("F3") or channel.startswith("F4") or channel.startswith("C3") or channel.startswith("C4") or channel.startswith("P3") or channel.startswith("P4") or channel.startswith("O1") or channel.startswith("O2") or channel.startswith("F7") or channel.startswith("F8") or channel.startswith("T3") or channel.startswith("T4") or channel.startswith("T5") or channel.startswith("T6") or channel.startswith("Fz") or channel.startswith("Pz") or channel.startswith("Cz") or channel.startswith("Oz") :
                self.new_name[channel] = "EEG " + channel
            if channel == "ECG" :
                self.new_name[channel] = "ECG ECG" 
        
        return self.transform(raw)
        
    
class Eeg_signal_extractor(PreProcessor):
    def __init__(
        self) -> None:
        pass

    def transform(
        self, raw ):
        """
        Extract the EEG signals from the raw object.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        mne.io.Raw
            The preprocessed raw object.
        """
    
        raw.drop_channels(self.list_to_removed)        
        raw.rename_channels(self.new_names)
        return raw
    
    def fit_transform(
        self, raw):
        """ 
        Find and Extract the EEG signals from the raw object.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        
        Returns
        -------
        mne.io.Raw
            The preprocessed raw object.
        """
        
        self.list_to_removed = []
        self.new_names = {}
        for channel in raw.info["ch_names"] :
            if not channel.startswith("EEG") or channel == "EEG E1": 
                self.list_to_removed.append(channel)
            else :   
                channel2 = channel.replace("EEG ", "")
                self.new_names[channel] = channel2
    
        return self.transform(raw)
    
class Resampler(PreProcessor):
    def __init__(
        self, sfreq = 250) -> None:
        """            
        sfreq : int
            The desired sampling frequency.
        """
        self.sfreq=sfreq
        

    def fit_transform(
        self, raw ):
        """
        Resample the raw object to the desired sampling frequency.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        """
        
        raw.resample(self.sfreq, npad="auto")
        return raw


class Montage_setter(PreProcessor):
    def __init__(
        self, montage_name="standard_1020") -> None:
        """
        Set the montage of the raw object to the desired montage.
        
        Parameters
        ----------
        montage_name : str
        """
        self.montage_name = montage_name
        

    def transform(
        self, raw ):
        raw.set_montage(self.montage)
        return raw
        

    def fit_transform(
        self, raw):
        
        self.montage = mne.channels.make_standard_montage(self.montage_name)
        return self.transform(raw)
    
class Cropper(PreProcessor):
    def __init__(
        self, tmin = 10, tmax = None) -> None:
        """ 
        Crop the raw object to the desired time interval.
        """
        self.tmin = tmin
        self.tmax = tmax

    def fit_transform(
        self, raw):
        return raw.crop (tmin = self.tmin,tmax=self.tmax)
    
class Pipeline_pre_processor(PreProcessor):
    """
    Implementation of a pipeline of PreProcessors
    """

    def __init__(self, steps: list) -> None:
        """ 
        Initialize the pipeline with a list of PreProcessors.
        """
        self.functions = []
        for step, arg in steps:
            # initialize
            if arg != None:
                self.functions.append(step(**arg))
            else:
                self.functions.append(step())

    def fit_transform(
        self, raw):
        """
        Fit and transform the data with all steps of the pipeline
        """
        for function in self.functions:
            raw = function.fit_transform(raw)
        return raw
    
class Filter(PreProcessor):
    """ 
    Filter the raw object to the desired frequency interval.
    """
    
    def __init__(self, low_cut, hi_cut) -> None:
        """
        Initialize the filter with the desired frequency interval.
        
        Parameters
        ----------
        low_cut : int
            The lower bound of the frequency interval.
        hi_cut : int
            The upper bound of the frequency interval.
        """
        self.low_cut = low_cut
        self.hi_cut = hi_cut
    
    def fit_transform(
        self, raw):
        """ 
        Filter the raw object to the desired frequency interval.
        """

        return raw.copy().filter(self.low_cut, self.hi_cut)
    
class Epochs_creator(PreProcessor):
    
    def __init__(self, epoch_lenght, overlap = 0) -> None:
        """ 
        Create epochs of the desired lenght and overlap.
        
        Parameters
        ----------
        epoch_lenght : int
            The lenght of the epochs.
        overlap : int, optional
            The overlap between epochs, by default 0.
        """
        self.epoch_lenght = epoch_lenght
        self.overlap = overlap
    
    def fit_transform(
        self, raw):
        """ 
        Create epochs of the desired lenght and overlap.
        
        Parameters
        ----------
        raw : mne.io.Raw
            The raw object to be preprocessed.
        """
        
        return mne.make_fixed_length_epochs(raw, duration=self.epoch_lenght, preload=False, overlap = self.overlap)
    
class Mean_remover(PreProcessor):
    """ 
    Remove the mean of the each channels.
    """
    def __init__(
        self) -> None:
        pass

    def fit_transform(
        self, raw):
        
        raw = raw.copy().load_data().apply_function(lambda x: (x - np.mean(x)))
        
        return raw
    
class Standard_scaler(PreProcessor):
    """ 
    Standardize the data of each given channels.
    """
    def __init__(
        self, channels) -> None:
        """ 
        Initialize the desired channels to standardize.
        """
        self.channels = channels

    def fit_transform(
        self, raw):
        """ 
        Standardize the data of each given channels.
        """
        
        raw = raw.copy().load_data().apply_function(lambda x: (x - np.mean(x))/np.std(x), picks = self.channels)
        return raw
    
class Common_average_reference(PreProcessor):
    def __init__(
        self) -> None:
        pass

    def fit_transform(
        self, raw):
        """ 
        Apply the common average reference to the raw object.
        """
        
        raw =raw.set_eeg_reference(ref_channels='average')
        
        return raw