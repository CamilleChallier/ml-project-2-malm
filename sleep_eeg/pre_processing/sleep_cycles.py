import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

from sleep_eeg.pre_processing.mq import make_classification
from sleep_eeg.utils import (
    get_night_identifications,
    get_sleep_stages,
    load_age_bmi_file,
    load_mq_file,
)


def get_sleep_stage_continuity(
    sleep_stages: list[np.ndarray], night_ids: list
) -> pd.DataFrame:
    """Get the number of time each sleep stage append.

    Args:
        sleep_stages (list(np.ndarray)): sleep stage time series for each patient
        night_ids (list): A list of all the night ids.

    Returns:
        pd.DataFrame: Data with number of time in a specific stage
    """
    compute_sleep_stages_changes = lambda x: x[
        [0] + list(np.where(np.roll(x, 1) != x)[0])
    ]
    compute_time_spend_stage = lambda x: list(
        np.unique(compute_sleep_stages_changes(x), return_counts=True)[1]
    )
    return pd.DataFrame(
        map(compute_time_spend_stage, sleep_stages),
        columns=[
            "stage0_continuity",
            "stage1_continuity",
            "stage2_continuity",
            "stage3_continuity",
            "stage5_continuity",
        ],
        index=night_ids,
    )


def count_number_combinations(arr):
    counts = {}

    for i in range(1, len(arr)):
        key = "stage_{}_to_{}".format(int(arr[i]), int(arr[i - 1]))
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1

    return counts


def get_sleep_stage_changement(
    sleep_stages: list[np.ndarray], night_ids: list
) -> pd.DataFrame:
    """Get the number of time each sleep stage combinaison switch.

    Args:
        sleep_stages (list(np.ndarray)): sleep stage time series for each patient
        night_ids (list): A list of all the night ids.

    Returns:
        pd.DataFrame: Record each combinaison of stage switch
    """
    compute_sleep_stages_changes = lambda x: x[
        [0] + list(np.where(np.roll(x, 1) != x)[0])
    ]

    sleep_stages_order = list(map(compute_sleep_stages_changes, sleep_stages))
    df = pd.DataFrame(
        map(count_number_combinations, sleep_stages_order), index=night_ids
    ).fillna(0)
    to_drop = [f"stage_{i}_to_{i}" for i in [0, 1, 2, 3, 5]]
    df[to_drop] = 0  # TODO : Camille could you check if this is correct ?
    return df


def extract_sleep_times_lantencies_wake_up_info(
    sleep_stage_cycles: list[np.ndarray], night_ids: list[str]
) -> pd.DataFrame:
    """Extract the sleep times and latencies of each sleep stage as well as some wake up information from the sleep stage cycles.

    Args:
        sleep_stage_cycles (list[np.ndarray]): A list of sleep stage cycles.
        night_ids (list[str]): The night identifications for each sleep stage cycle.

    Returns:
        pd.DataFrame: Sleep stages times and latencies, plus wake up and total sleep infos, for all nights.
    """
    out = {}
    for sleep_stage_cycle, night_id in zip(sleep_stage_cycles, night_ids):
        out[night_id] = {}
        # Sleep latency Feinberg and Floyd criteria, first three non consecutive epoch zeros
        # Create a boolean mask for non-zero elements
        non_zero_mask = (sleep_stage_cycle != 0).astype(int)
        # Use convolution to find consecutive non-zero elements
        convolution_result = np.convolve(non_zero_mask, [1, 1, 1], mode="valid")
        sleep_start = np.argwhere(convolution_result >= 3)[0, 0]
        out[night_id]["sleep_latency"] = sleep_start
        # Total sleep time is equal to all the non zero epochs
        total_sleep_time = np.argwhere(sleep_stage_cycle[sleep_start:] != 0).shape[0]
        out[night_id]["total_sleep_time"] = total_sleep_time
        out[night_id]["lay_awake_time"] = sleep_stage_cycle.shape[0] - total_sleep_time
        # Wake up info
        out[night_id]["woken_from_stage"] = sleep_stage_cycle[
            np.argwhere(sleep_stage_cycle != 0)
        ][-1, 0]
        out[night_id]["woken_by_research_team"] = 1 if sleep_stage_cycle[-1] != 0 else 0
        for i in [1, 2, 3, 5]:
            stage_i_indices = np.argwhere(sleep_stage_cycle[sleep_start:] == i)

            # Handle edge case for night `A16_N4` which never reaches a stage 5 sleep.
            if night_id == "A16_N4" and i == 5:
                stage_i_time = 0.0
                sleep_i_latency = np.nan
            else:
                stage_i_time = stage_i_indices.shape[0]
                sleep_i_latency = stage_i_indices[0, 0]

            out[night_id][f"stage{i}_time"] = stage_i_time
            # TODO: check with Laura if criteria is met when patient starts by a sleep stage 2, 3 or 5
            out[night_id][f"stage{i}_latency"] = sleep_i_latency

    return pd.DataFrame(out).T


def find_smallest_sleep_time(
    sleep_stages: list[np.ndarray], threshold: int = 800
) -> int:
    """Find the length of the shortest sleep duration, above a certain threshold.
    Here 1 duration is equivalent to 30s.

    Args:
        sleep_stages (list[np.ndarray]): A list of numpy arrays containing the sleep stages of each patient.
        threshold (int, optional): The minimum threshold time for a full night sleep. Defaults to 800.

    Returns:
        int: The smallest sleep duration.
    """
    smallest_array_length = sleep_stages[0].shape[0]
    for sleep_stages_per_patient in sleep_stages:
        if (
            sleep_stages_per_patient.shape[0] < smallest_array_length
            and sleep_stages_per_patient.shape[0] > threshold
        ):
            smallest_array_length = sleep_stages_per_patient.shape[0]

    return smallest_array_length


def resample_sleep_stages(array1D: np.ndarray, target_length: int) -> np.ndarray:
    """Resample the time series sleep stages of a patient.

    Args:
        array1D (np.ndarray): the old sleep stages vector
        target_length (int): the target length of the new sleep stages vector

    Returns:
        np.ndarray: the new sleep stages vector with shape (target_length,)
    """
    old_x = np.linspace(0, 1, array1D.shape[0])
    f = interp1d(old_x, array1D, kind="nearest")
    new_x = np.linspace(0, 1, target_length)
    return f(new_x).astype(int)


def resample_all_nights(
    night_identifications: list[str],
    sleep_stages: list[np.ndarray],
    full_night_threshold: int = 800,
) -> dict:
    """Resample all nights to the same length.

    Args:
        night_identifications (list[str]): The identifications of the night.
        sleep_stages (list[np.ndarray]): A list of each sleep_stages for each night_id.
        full_night_threshold (int, optional): The minimum threshold time for a full night sleep. Defaults to 800.

    Returns:
        dict: A dictionnary of structure `night_id` : { `sleep_duration` : int , `resampled_sleep_stages` : np.ndarray }
    """
    resampling_length = find_smallest_sleep_time(
        sleep_stages=sleep_stages, threshold=full_night_threshold
    )
    return {
        night_id: {
            "sleep_duration": sleep_stages_per_patient.shape[0],
            "resampled_sleep_stages": resample_sleep_stages(
                sleep_stages_per_patient, resampling_length
            ),
        }
        for night_id, sleep_stages_per_patient in zip(
            night_identifications, sleep_stages
        )
    }


def create_resampled_time_series_df(
    sleep_stages: list[np.ndarray], night_ids: list, full_night_threshold: int = 800
) -> pd.DataFrame:
    """Create a DataFrame with each night being a row, and the time series info about the sleep stages on the columns.

    Args:
        sleep_stages (list[np.ndarray]): A list of numpy arrays containing the sleep stages of each patient.
        night_ids (list): A list of all the night ids.
        full_night_threshold (int, optional): The minimum threshold time for a full night sleep (for resample_all_nights). Defaults to 800.
    Returns:
        pd.DataFrame: Number of columns equal to length of resampled nights + 1 for `sleep_duration`
    """
    resampled_nights_df = pd.DataFrame(
        resample_all_nights(
            night_ids, sleep_stages, full_night_threshold=full_night_threshold
        )
    ).T
    sleep_stages_df = pd.DataFrame(
        resampled_nights_df.iloc[:, 1].to_list()  # TODO: add relevant column names
    )
    # sleep_stages_df.insert(loc=0, column='sleep_duration', value=resampled_nights_df["sleep_duration"])
    sleep_stages_df["sleep_duration"] = resampled_nights_df["sleep_duration"].to_list()
    sleep_stages_df.index = night_ids
    sleep_stages_df.columns = [
        f"resampled_t{i}" for i in range(sleep_stages_df.shape[1])
    ]
    return sleep_stages_df


def get_sleep_stage_duration_df(
    sleep_stages_df: pd.DataFrame, add_total_sleep_time: bool = True
) -> pd.DataFrame:
    """Get the amount of time slots attributed to each sleep stages.
     REDUNDANT WITH extract_sleep_times_lantencies_wake_up_info

    Args:
        sleep_stages_df (pd.DataFrame): See `create_resampled_time_series_df` for more info.
        add_total_sleep_time (bool): If True add back the total sleep time info. Defaults to True.

    Returns:
        pd.DataFrame: Data with individual durations of each
    """
    sleep_stages_individual_durations = sleep_stages_df.iloc[:, :-1]
    sleep_stages_individual_durations = sleep_stages_individual_durations.apply(
        pd.Series.value_counts, axis=1
    )
    if add_total_sleep_time:
        sleep_stages_individual_durations["total_sleep_duration"] = sleep_stages_df[
            "sleep_duration"
        ]
    return sleep_stages_individual_durations


def get_fourier_elements_df(
    sleep_stages: list[np.ndarray],
    night_ids: list[str],
    nbr_elements: int,
    fs: int = 1 / 30,
    sleep_duration: bool = False,
):
    """Returns a dataframe with 2*nbr_elements+1 columns with the biggest fourier transform amplitudes of the sleep stages vector and their corresponding frequencies.
    The last column is the duration of the night.

    Args:
        sleep_stages (list[np.ndarray]): A list of numpy arrays containing the sleep stages of each patient.
        night_ids (list[str]): A list of all the night ids.
        nbr_elements (int): The number of fourier elements (ie Amplitude and f) to extract.
        fs (int, optional): The sampling frequency. for our data this is 1/30[Hz]
        sleep_duration (bool, optional): Whether to include the sleep duration in the dataframe or not.

    Returns:
        fourier_elements [pd.DataFrame]: (2*nbr_elements+1, number of nights) dataframe with the fourier elements of each night.
    """
    # TODO : multichannel fourier transform
    fourier_elements = pd.DataFrame()

    for stages in sleep_stages:
        fft_wave = np.fft.fft(stages)
        fft_freq = np.fft.fftfreq(fft_wave.shape[0], d=1 / fs)
        args = np.argsort(fft_wave)[-nbr_elements:]
        amplitudes = fft_wave[args]
        freqs = fft_freq[args]
        fourier_elements = pd.concat(
            [
                fourier_elements,
                pd.DataFrame(
                    [
                        amplitudes.real.tolist()
                        + amplitudes.imag.tolist()
                        + freqs.tolist()
                        + ([stages.shape[0]] if sleep_duration else [])
                    ]
                ),
            ],
            ignore_index=True,
        )

    fourier_elements.columns = (
        ["A.real" + str(i) for i in range(1, nbr_elements + 1)]
        + ["A.imag" + str(i) for i in range(1, nbr_elements + 1)]
        + ["f" + str(i) for i in range(1, nbr_elements + 1)]
        + (["sleep_duration"] if sleep_duration else [])
    )

    fourier_elements.index = night_ids

    return fourier_elements


def get_individual_sleep_stage_PSD(
    sleep_stage_cycles: list[np.ndarray], night_ids: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a DataFrame of power spectral densities for each sleep stage of each night

    Args:
        sleep_stage_cycles (list[np.ndarray]): All the sleep stage cycles
        night_ids (list[str]): Each id corresponding to each night in sleep_stage_cycles

    Returns:
        pd.DataFrame: PSD data for each night (rows), each stage and each frequencies (columns)
        np.ndarray: Frequencies corresponding to each `fi` column
    """
    freq, psd = welch(sleep_stage_cycles[0], fs=1 / 30)
    step = freq.shape[0]
    df = pd.DataFrame(
        index=night_ids,
        columns=[
            f"stage_{i}_psd_f{j}" for i in [0, 1, 2, 3, 5] for j, _ in enumerate(freq)
        ],
    )
    for i, sleep_stage_cycle in enumerate(sleep_stage_cycles):
        for j, stage_i in enumerate([0, 1, 2, 3, 5]):
            freq_stage_i, psd_stage_i = welch(
                (sleep_stage_cycle == stage_i).astype(int), fs=1 / 30
            )
            df.iloc[i, j * step : (j + 1) * step] = psd_stage_i
    return df.astype(float), freq


def assemble_data(
    nbr_fourier_elements: int = 5,
    add_bias: bool = True,
    drop_mq_grp_par_nig_dru: bool = True,
    full_night_threshold: int = 800,
    mq_class_percentiles: list[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """Assembles the DataFrame including the multiple possible responses, and all the other columns predictors.

    Args:
        nbr_fourier_elements (int, optional): The number of fourier elements (ie Amplitude and f) to extract. Defaults to 5.
        add_bias (bool, optional): If True a column of ones will be added on the left of the DataFrame. Defaults to True.
        drop_mq_grp_par_nig_dru (bool, optional): Drop the `group`, `participant`, `night` and `drug` columns in mq. Defaults to True.
        full_night_threshold (int, optional): The full night threshold used to determine resmpling. Defaults to 800.
        mq_class_percentiles (list[float], optional): The percentiles used to classifiy the morning questionnaire answers. If None, no classification is done. Defaults to None.

    Returns:
        pd.DataFrame: The assembled DataFrame.
        dict: The different groups of variables in the DataFrame

    """
    mq = load_mq_file()
    if mq_class_percentiles is not None:
        mq = make_classification(mq, percentiles=mq_class_percentiles)

    night_ids = get_night_identifications(mq)
    sleep_stages = get_sleep_stages(night_ids)
    sleep_stages_changements_df = get_sleep_stage_changement(sleep_stages, night_ids)
    sleep_durations_latencies_df = extract_sleep_times_lantencies_wake_up_info(
        sleep_stages, night_ids
    )
    sleep_stages_continuity = get_sleep_stage_continuity(sleep_stages, night_ids)
    fourier_elements_df = get_fourier_elements_df(
        sleep_stages, night_ids, nbr_fourier_elements, sleep_duration=False
    )
    time_series_df = create_resampled_time_series_df(
        sleep_stages, night_ids, full_night_threshold=full_night_threshold
    )
    psd_df, _ = get_individual_sleep_stage_PSD(sleep_stages, night_ids)

    mq_columns = mq.columns.tolist()
    mq_index = mq.index
    age_bmi = load_age_bmi_file()
    mq = pd.merge(mq, age_bmi)
    mq.index = mq_index

    if drop_mq_grp_par_nig_dru:
        mq.drop(["Group", "Participant", "Night", "Drug"], inplace=True, axis=1)

    groups = {
        "durations_latencies": sleep_durations_latencies_df.columns.tolist(),
        "changements": sleep_stages_changements_df.columns.tolist(),
        "continuity": sleep_stages_continuity.columns.tolist(),
        "fourrier": fourier_elements_df.columns.tolist(),
        "resampled_times_series": time_series_df.columns.tolist(),
        "psd": psd_df.columns.tolist(),
        "age_bmi": ["Age (y)", "Sex (1=m, 2=f)", "bmi (kg/m^2)"],
        "mq": mq_columns,
    }

    avengers = pd.concat(
        [
            sleep_stages_changements_df,
            sleep_durations_latencies_df,
            sleep_stages_continuity,
            fourier_elements_df,
            time_series_df,
            psd_df,
            mq,
        ],
        axis=1,
    )

    if add_bias:
        avengers.insert(loc=0, column="bias", value=1.0)
    return avengers, groups
