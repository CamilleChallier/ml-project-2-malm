import os

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error


def load_mq_file(path: str = None) -> pd.DataFrame:
    """Load the MQ.xlsx file into a pandas DataFrame, from its file path"""
    if path is None:
        path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "MQ.xlsx",
        )
    mq = pd.read_excel(path)
    mq.drop("Unnamed: 14", axis=1, inplace=True)
    mq.replace("-", np.nan, inplace=True)
    mq.dropna(inplace=True)
    mq = mq.iloc[:-1]  # Remove non-existing B25_N3.xlsx file info.
    mq.index = get_night_identifications(mq)
    return mq


def load_age_bmi_file(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "Book1.xlsx",
        )
    return pd.read_excel(path).dropna()


def get_night_identifications(mq: pd.DataFrame) -> list[str]:
    """Get a list of the night file identifications, for example ['A01_N3', 'A01_N4', 'A01_N5', ...]"""
    return [
        f"{group}0{int(participant)}_{night}"
        if participant < 10
        else f"{group}{int(participant)}_{night}"
        for group, participant, night in zip(
            mq["Group"], mq["Participant"], mq["Night"]
        )
    ]


def load_eeg(path: str) -> pd.DataFrame:
    data = mne.io.read_raw_edf(path)
    return data


def load_data_eeg(directory: str = "../data/eeg") -> pd.DataFrame:
    dict_eeg = {}

    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("N3.edf"):
                eeg = load_eeg(filepath)
                data = eeg.get_data()[:-2]
                dict_eeg[file] = data

    return dict_eeg


def get_sleep_stages(
    night_ids: list[str], path_to_scorings: str = None
) -> list[np.ndarray]:
    """Get the sleep stages from every night identification in `night_ids`, from the `Scorings` folder"""
    if path_to_scorings is None:
        path_to_scorings = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "Scorings",
        )
    out = [
        pd.read_excel(os.path.join(path_to_scorings, f"{night_id}.xlsx"))[
            "V7"
        ].to_numpy()
        for night_id in night_ids
    ]
    # Fix `255` sleep stage anomaly for night `B08_N3`
    B08_N3_index = night_ids.index("B08_N3")
    out[B08_N3_index] = np.where(out[B08_N3_index] == 255, 0, out[B08_N3_index])
    return out


def get_custom_rmse():
    """Get the custom RMSE for sklearn scorers"""
    return make_scorer(
        lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
        greater_is_better=False,
    )


def rename_nights(old_night_ids: list[str]):
    night_ids = []
    for night_fname in old_night_ids:
        night_id = night_fname.split(".")[0]
        # print(night_id[1:3])
        if "_" in night_id[1:3]:
            new_night_id = night_id[0] + "0" + night_id[1:]
        else:
            new_night_id = night_id
        night_ids.append(new_night_id)
    return night_ids
