import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes


def plot_mq_histograms(
    mq: pd.DataFrame,
    layout: tuple = (2, 5),
    figsize: tuple = (15, 5),
    percentiles: list[float] = [0.25, 0.5, 0.75],
    mq_classified_index: list[int] = [3, 4, 5, 6, 7, 8, 9],
    mq_names: list[str] = [
        "Time to fall asleep [min]",
        "Wake ups during the night",
        "Time awake during the night [min]",
        "Calm sleep",
        "Sleep superficiality",
        "Recovery",
        "Mood today",
        "Energy today",
        "Feeling tense",
        "Concentration",
    ],
) -> tuple:
    """Plot morning questionnaire distributions as histograms with percentiles.
    Args:
        mq (pd.DataFrame): Morning questionnaire dataframe.
        layout (tuple, optional): Subplot layout. Defaults to (2,5). Alternative is (3,4).
        figsize (tuple, optional): Figure size. Defaults to (15,5). Alternative is (10,8).
        percentiles (list[float], optional): Percentiles to plot. Defaults to [0.25, 0.5, 0.75].
        mq_classified_index (list[int], optional): Index of the mqs that are classified. Defaults to [3,4,5,6,7,8,9].
        mq_names (list[str], optional): Names of the questions. Defaults to ["Time to fall asleep [min]","Wake ups during the night","Time awake during the night [min]","Calm sleep","Sleep superficiality","Recovery","Mood today","Energy today","Feeling tense","Concentration",].
    Returns:
        tuple: Figure and axes.
    """
    sns.set_theme(style="darkgrid")
    mqs = [m for m in mq.columns.tolist() if m.startswith("mq")]
    percentiles_name = [
        f"{percentage}%"
        for percentage in (np.round(percentiles, decimals=2) * 100).astype(int).tolist()
    ]
    quantile_df = mq.describe(percentiles=percentiles)

    fig, axes = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=figsize)
    axes = axes.flatten()

    for ax, metric in zip(axes, mqs):
        sns.histplot(
            mq,
            x=metric,
            kde=True,
            ax=ax,
            common_norm=True,
            # hue="Drug",
        )
        ax.set_xlabel(mq_names[mqs.index(metric)])

        if mqs.index(metric) in mq_classified_index:
            quantiles = [quantile_df[metric][p] for p in percentiles_name]
            # ax.set_xticks(quantiles)
            # ax.set_xticklabels(np.floor(quantiles).astype(int).tolist())#, rotation= 90)
            kde_line = sns.kdeplot(mq[metric], ax=ax, color="red")
            x_kde, y_kde = ax.lines[0].get_xydata().T
            ax.lines[1].remove()
            del kde_line
            for q in quantiles:
                idx = np.argmin(np.abs(x_kde - q))
                ax.axvline(
                    x=x_kde[idx],
                    ymin=0,
                    ymax=y_kde[idx] / ax.get_ylim()[1],
                    color="C0",
                    linestyle="--",
                )
    if layout == (3, 4):
        axes[10].axis("off")
        axes[11].axis("off")
    plt.tight_layout()
    return fig, axes


def plot_mq_boxplot(
    mq: pd.DataFrame,
    mq_names: list[str] = [
        "Time to fall asleep [min]",
        "Wake ups during the night",
        "Time awake during the night [min]",
        "Calm sleep",
        "Sleep superficiality",
        "Recovery",
        "Mood today",
        "Energy today",
        "Feeling tense",
        "Concentration",
    ],
) -> tuple:
    """Plot morning questionnaire distributions as boxplots
    Args:
        mq (pd.DataFrame): Morning questionnaire dataframe.
        mq_names (list[str], optional): Names of the questions. Defaults to ["Time to fall asleep [min]","Wake ups during the night","Time awake during the night [min]","Calm sleep","Sleep superficiality","Recovery","Mood today","Energy today","Feeling tense","Concentration",].
    Returns:
        tuple: Figure and axes.
    """
    sns.set_theme(style="darkgrid")
    qualitative_metrics = [m for m in mq.columns.tolist() if m.startswith("mq")]
    fig, axes = plt.subplots(ncols=len(qualitative_metrics), figsize=(30, 8))
    for ax, metric in zip(axes, qualitative_metrics):
        sns.boxplot(
            mq,
            y=metric,
            ax=ax,
            # hue="Drug",
        )
        ax.set_ylabel(mq_names[qualitative_metrics.index(metric)])
    plt.tight_layout()
    return fig, axes
