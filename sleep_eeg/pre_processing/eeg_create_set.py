import numpy as np 
import pandas as pd
import os
import mne

def resample_sleep_eeg(array1D: np.ndarray, target_length: int) -> np.ndarray:
    """Resample the time series sleep stages of a patient.

    Args:
        array1D (np.ndarray): the old sleep stages vector
        target_length (int): the target length of the new sleep stages vector

    Returns:
        np.ndarray: the new sleep stages vector with shape (target_length,)
    """
    old_x = np.linspace(0, 1, array1D.shape[1])
    new_lenght = np.linspace(0, 1, target_length)
    new_x = []
    for i in range(array1D.shape[0]):
        new_x.append(np.interp(new_lenght, old_x, array1D[i] ))
    return np.array(new_x)

def find_smallest_sleep_time(
    sleep_stages: list[np.ndarray], threshold: int = 11000000 
) -> int:
    """Find the length of the shortest sleep duration, above a certain threshold.
    Here 1 duration is equivalent to 1/512 sec.

    Args:
        sleep_stages (list[np.ndarray]): A list of numpy arrays containing the eeg of each patient.
        threshold (int, optional): The minimum threshold time for a full night sleep

    Returns:
        int: The smallest sleep duration.
    """
    smallest_array_length = sleep_stages[0][0].shape[0]
    for sleep_stages_per_patient in sleep_stages:
        if (
            sleep_stages_per_patient[0].shape[0] < smallest_array_length
            and sleep_stages_per_patient[0].shape[0] > threshold
        ):
            smallest_array_length = sleep_stages_per_patient[0].shape[0]
    return smallest_array_length


def resample_all_nights(
    night_identifications: list[str],
    sleep_stages: list[np.ndarray],
    full_night_threshold: int = 11000000,
) -> dict:
    """Resample all nights to the same length.

    Args:
        night_identifications (list[str]): The identifications of the night.
        sleep_stages (list[np.ndarray]): A list of each sleep_stages for each night_id.
        full_night_threshold (int, optional): The minimum threshold time for a full night sleep.

    Returns:
        dict: A dictionnary of structure `night_id` : { `sleep_duration` : int , `resampled_sleep_stages` : np.ndarray }
    """
    resampling_length = find_smallest_sleep_time(
        sleep_stages=sleep_stages, threshold=full_night_threshold
    )
    return {
        night_id: {
            "sleep_duration": sleep_stages_per_patient[0].shape[0],
            "resampled_sleep_eeg": resample_sleep_eeg(
                sleep_stages_per_patient, resampling_length
            ),
        }
        for night_id, sleep_stages_per_patient in zip(
            night_identifications, sleep_stages
        )
    }