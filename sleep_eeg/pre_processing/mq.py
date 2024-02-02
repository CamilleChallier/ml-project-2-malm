import numpy as np
import pandas as pd


def make_classification(
    mq: pd.DataFrame,
    percentiles: list[float] = [0.25, 0.5, 0.75],
    classifiable_mqs: list[str] = [
        "mq_4_calm_sleep",
        "mq_5_superficial_sleep",
        "mq_6_recovery",
        "mq_7_bad_mood",
        "mq_8_energy",
        "mq_9_tense",
        "mq_10_unconcentrated",
    ],
) -> pd.DataFrame:
    """Classify the mq metric responses depending on the quantile informations.

    Args:
        mq (pd.DataFrame): The mq dataframe.
        percentiles (list[float], optional): The percentiles to calculate the value. Defaults to [0.25, 0.5, 0.75].
        classifiable_mqs (list[str], optional): The columns that should be classified. Defaults to [ "mq_4_calm_sleep", "mq_5_superficial_sleep", "mq_6_recovery", "mq_7_bad_mood", "mq_8_energy", "mq_9_tense", "mq_10_unconcentrated", ].

    Returns:
        pd.DataFrame: The classified mq data
    """
    mq_classified = mq.copy(deep=True)
    quantile_df = mq.describe(percentiles=percentiles)
    percentiles = [
        f"{percentage}%"
        for percentage in (np.round(percentiles, decimals=2) * 100).astype(int).tolist()
    ]
    for mq_metric in classifiable_mqs:
        mq_metric_quantiles = quantile_df[mq_metric]

        def classify(x: float) -> int:
            if 0.0 <= x <= mq_metric_quantiles[percentiles[0]]:
                return 0
            for class_id, (low, high) in enumerate(
                zip(percentiles, percentiles[1:] + ["max"]), start=1
            ):
                if mq_metric_quantiles[low] < x <= mq_metric_quantiles[high]:
                    return class_id

        mq_classified[mq_metric] = mq[mq_metric].apply(classify)

    return mq_classified


def compute_class_weights(
    mq_classified: pd.DataFrame,
    classified_mq_cols: list[str] = [
        "mq_4_calm_sleep",
        "mq_5_superficial_sleep",
        "mq_6_recovery",
        "mq_7_bad_mood",
        "mq_8_energy",
        "mq_9_tense",
        "mq_10_unconcentrated",
    ],
) -> pd.DataFrame:
    """Compute 1-frequency_i_j where i is the mq metric and j is the corresponding class, to get class weights

    Args:
        mq_classified (pd.DataFrame): The classified morning questionnaire data
        classified_mq_cols (list[str], optional): The columns that were classified. Defaults to [ "mq_4_calm_sleep", "mq_5_superficial_sleep", "mq_6_recovery", "mq_7_bad_mood", "mq_8_energy", "mq_9_tense", "mq_10_unconcentrated", ].

    Returns:
        pd.DataFrame: The class weights for each class and each classified column
    """
    return (
        1
        - mq_classified[classified_mq_cols].astype(int).apply(pd.value_counts)
        / mq_classified.shape[0]
    )
