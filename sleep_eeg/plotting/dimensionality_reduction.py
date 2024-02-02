import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def plot_lower_dimensional_subspace(
    data: np.ndarray,
    mq: pd.DataFrame,
    dimensions: int = 2,
    loc: str = "lower right",
    classified_mqs: list[str] = [
        "Group",
        "Participant",
        "Group+Participant",
        "Night",
        "Drug",
        "mq_1_Sleeplatency",
        "mq_2_times_awake",
        "mq_3_time_lay_awake",
        "mq_4_calm_sleep",
        "mq_5_superficial_sleep",
        "mq_6_recovery",
        "mq_7_bad_mood",
        "mq_8_energy",
        "mq_9_tense",
        "mq_10_unconcentrated",
        "Age (y)",
        "Sex (1=m, 2=f)",
        "bmi (kg/m^2)",
    ],
    encode_name: list[str] = ["Group", "Night", "Group+Participant"],
    show: bool = True,
    figsize=(20, 30),
    ncols=4,
    nrows=5,
):
    if dimensions == 3:
        subplot_kw = dict(projection="3d")
    else:
        subplot_kw = None
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize, subplot_kw=subplot_kw
    )

    for i, (ax, mq_name) in enumerate(zip(axes.flatten(), classified_mqs)):
        if mq_name in encode_name:
            c = LabelEncoder().fit_transform(mq[mq_name])
        else:
            c = mq[mq_name]

        if dimensions == 2:
            scatter = ax.scatter(
                x=data[:, 0],
                y=data[:, 1],
                c=c,
                alpha=0.5,
                cmap="viridis",
            )
        elif dimensions == 3:
            scatter = ax.scatter(
                xs=data[:, 0],
                ys=data[:, 1],
                zs=data[:, 2],
                marker="o",
                c=c,
            )
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            *scatter.legend_elements(),
            # title=mq_name,
            loc=loc,
            # bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            # shadow=True,
            ncol=4,
        )
        ax.add_artist(legend1)
        ax.set_title(mq_name)
    plt.tight_layout()
    if show:
        plt.show()
    return axes
