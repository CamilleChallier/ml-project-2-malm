import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d


def plot_sleep_cycle(
    ax,
    night_id: str,
    sleep_stage_cycle: np.ndarray,
    sleep_durations_latencies_df: pd.DataFrame,
    main_curve_color: str = "gray",
    main_curve_width: float = 1.0,
    sleep_onset_color: str = "red",
    sleep_stages_colors: list[str] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ],
    sleep_stages_colors_alpha: float = 0.6,
    ymin: float = -0.5,
    ymax: float = 5.9,
    major_xaxis_ticks: int = 50,
    minor_xaxis_ticks: int = 10,
    upsampling_ratio: int = 10,
    stage_labels: list = ["Awake", "N1", "N2", "N3", "REM"],
) -> None:
    """Plot the sleep stage cycle of a night, on a certain axis

    Args:
        ax (Axes): Matplotlib axis
        night_id (str): The night id
        sleep_stage_cycle (np.ndarray): The sleep stage cycle
        sleep_durations_latencies_df (pd.DataFrame): The sleep stages latencies and times, output of the `sleep_eeg.pre_processing.sleep_cycles.extract_sleep_times_lantencies_wake_up_info` function.
        main_curve_color (str, optional): Color of the main curve. Defaults to 'gray'.
        main_curve_width (float, optional): Width of the main curve. Defaults to 1.0.
        sleep_stages_colors (list[str], optional): Colors for each sleep stage. Defaults to [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"].
        sleep_onset_color (str, optional): Color of the sleep onset vertical line. Defaults to 'red'.
        sleep_stages_colors_alpha (float, optional): Alpha value for the sleep stages colors. Defaults to 0.6.
        ymin (float, optional): The lowest visible y value. Defaults to -0.5.
        ymax (float, optional): The highest visible y value. Defaults to 5.9.
        major_xaxis_ticks (int, optional): Every how much do you want a major tick. Defaults to 50.
        minor_xaxis_ticks (int, optional): Every how much do you want a minor tick. Defaults to 10.
        upsampling_ratio (int, optional): The upsampling ratio, usefull for coloring the area under curve. Defaults to 10.
        stage_labels (list, optional): The stage labels. Defaults to ["Awake", "N1", "N2", "N3", "REM"].
    """
    recording_length = sleep_stage_cycle.shape[0]

    ax.plot(
        sleep_stage_cycle,
        label=f"sleep cycle",
        drawstyle="steps-mid",
        color=main_curve_color,
        linewidth=main_curve_width,
    )

    # Resample for filling area under the curve more accuratly (no blank spots)
    x = np.arange(recording_length)
    x_interp = np.linspace(0, recording_length - 1, recording_length * upsampling_ratio)
    y_interp = interp1d(x, sleep_stage_cycle, kind="nearest")(x_interp)

    for latency, color, stage, stage_label in zip(
        [
            "sleep_latency",
            "stage1_latency",
            "stage2_latency",
            "stage3_latency",
            "stage5_latency",
        ],
        sleep_stages_colors,
        [0, 1, 2, 3, 5],
        stage_labels,
    ):
        if latency != "sleep_latency":
            ax.fill_between(
                x_interp,
                y1=ymin,
                y2=y_interp,
                step="mid",
                alpha=sleep_stages_colors_alpha,
                where=y_interp == stage,
                color=color,
                label=f"{stage_label} time = {int(sleep_durations_latencies_df.loc[night_id][f'stage{stage}_time'])} epochs",
            )
            x_index = sleep_durations_latencies_df.loc[night_id][latency]
            if x_index == 0:
                continue
            x_index += sleep_durations_latencies_df.loc[night_id]["sleep_latency"]

            ax.hlines(
                y=stage,
                xmin=sleep_durations_latencies_df.loc[night_id]["sleep_latency"],
                xmax=x_index,
                label=f"{latency.replace('_', ' ').replace(f'stage{stage}', stage_label)} = {int(sleep_durations_latencies_df.loc[night_id][latency])} epochs",
                colors=color,
                linestyles="--",
                alpha=sleep_stages_colors_alpha,
            )

        else:
            ax.vlines(
                x=sleep_durations_latencies_df.loc[night_id]["sleep_latency"],
                ymin=ymin,
                ymax=ymax,
                label=f"sleep onset = {int(sleep_durations_latencies_df.loc[night_id][latency])} epochs",
                colors=sleep_onset_color,
                linestyles="-",
            )

            ax.fill_between(
                x_interp,
                y1=ymin,
                y2=y_interp,
                step="mid",
                alpha=sleep_stages_colors_alpha,
                where=y_interp == 0,
                color=color,
                label=f"lay awake time = {int(sleep_durations_latencies_df.loc[night_id]['lay_awake_time'])} epochs",
            )

    ax.set_xlabel("epoch [30s]", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(major_xaxis_ticks))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_xaxis_ticks))
    ax.set_xlim([0, recording_length])
    ax.set_ylabel("sleep stage", fontsize=16)
    ax.set_yticks([0, 1, 2, 3, 5])
    ax.set_yticklabels(stage_labels, fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_ylim([ymin, ymax])
    # ax.set_title(f"Sleep cycle for night {night_id}")
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
        fontsize=16,
    )


def plot_sleep_stage_epochs(
    sleep_durations_latencies_df: pd.DataFrame,
    night_id: str,
    ax: Axes,
    colors: list = None,
    title: str = None,
    annot: bool = True,
    xtick_labels: list = ["Awake", "N1", "N2", "N3", "REM"],
    xlabel: str = "sleep stage",
) -> Axes:
    """Plot the number of epochs attribuated to each sleep stage for a single night.

    Args:
        sleep_durations_latencies_df (pd.DataFrame): The sleep stages latencies and times, output of the `sleep_eeg.pre_processing.sleep_cycles.extract_sleep_times_lantencies_wake_up_info` function.
        night_id (str): The id of the night to plot.
        ax (Axes): Matplotlib axis.
        colors (list, optional): Colors of the bars. If None set defaults. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        annot (bool, optional): Whether to annotate each bar with its count. Defaults to True.
        xtick_labels (list, optional): The xtick labels. Defaults to ["Awake", "N1", "N2", "N3", "REM"].
        xlabel (str, optional): The xlabel. Defaults to "sleep stage".

    Returns:
        Axes: matplotlib axis.
    """
    if colors is None:
        colors = sns.color_palette(n_colors=5)
    sns.barplot(
        sleep_durations_latencies_df.loc[night_id][
            [
                "lay_awake_time",
                "stage1_time",
                "stage2_time",
                "stage3_time",
                "stage5_time",
            ]
        ],
        palette=colors,
        ax=ax,
    )
    if annot:
        for i in ax.containers:
            ax.bar_label(
                i,
                fontsize=16,
            )

    ax.set_xticks(ax.get_xticks())  # To remove a matplotlib warning
    ax.set_xticklabels(xtick_labels, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)

    ax.tick_params(axis="y", labelsize=14)
    if title is None:
        title = f"Cumulative sleep stage epochs for night {night_id}"
    ax.set_title(title)
    ax.set_ylabel("epoch [30s]", fontsize=16)
    return ax


def plot_sleep_stage_percentage(
    sleep_stage_cycle: np.ndarray,
    colors: list = None,
    title: str = "Total sleep time percentage of each sleep stage for 1 night",
    xtick_labels: list = ["Awake", "N1", "N2", "N3", "REM"],
    xlabel: str = None,
    annot: bool = True,
) -> Axes:
    """Plot the percentage of the total sleep time, for each sleep stage.

    Args:
        sleep_stage_cycle (np.ndarray): A single sleep stage cycle
        colors (list, optional): Colors of the bars. If None set defaults. Defaults to None.
        title (str, optional): . Defaults to "Total sleep time percentage of each sleep stage for 1 night".
        xtick_labels (list, optional): Change the xtick labels. Defaults to ["Awake", "N1", "N2", "N3", "REM"].
        xlabel (str, optional): Change the xlabel. Defaults to None.
        annot (bool, optional): Whether to annotate each bar with its percentage. Defaults to True.

    Returns:
        Axes: matplotlib axis.
    """
    pct_df = pd.DataFrame(
        columns=["sleep stage", "temp"], index=list(range(sleep_stage_cycle.shape[0]))
    )
    pct_df["sleep stage"] = sleep_stage_cycle
    pct_df["temp"] = 1

    if colors is None:
        colors = sns.color_palette(n_colors=5)[1:]

    ax = sns.countplot(
        pct_df[pct_df["sleep stage"] != 0],
        palette=colors,
        stat="percent",
        x="sleep stage",
    )
    if annot:
        for i in ax.containers:
            ax.bar_label(
                i,
            )

    ax.set_title(title)
    ax.set_ylabel("[%]")

    if xtick_labels is not None:
        ax.set_xticks(ax.get_xticks())  # To remove a matplotlib warning
        ax.set_xticklabels(xtick_labels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return ax


def plot_sleep_continuity(
    sleep_stages_continuity: pd.DataFrame,
    night_id: str,
    colors: list = None,
    title: str = None,
    xtick_labels: list = ["Awake", "N1", "N2", "N3", "REM"],
    xlabel: str = "sleep stage",
    annot: bool = True,
) -> Axes:
    """_summary_

    Args:
        sleep_stages_continuity (pd.DataFrame): Data from `sleep_eeg.pre_processing.sleep_cycles.get_sleep_stage_continuity`
        night_id (str): ID of the night.
        colors (list, optional): Colors to use for each bar. Defaults to None.
        title (str, optional): Title. Defaults to None.
        xtick_labels (list, optional): xtick_labels. Defaults to ["Awake", "N1", "N2", "N3", "REM"].
        xlabel (str, optional): xlabel. Defaults to "sleep stage".
        annot (bool, optional): Anottate the bars or not. Defaults to True.

    Returns:
        Axes: matplotlib.pyplot axis
    """

    if colors is None:
        colors = sns.color_palette(n_colors=5)
    ax = sns.barplot(sleep_stages_continuity.loc[night_id], palette=colors)
    if annot:
        for i in ax.containers:
            ax.bar_label(
                i,
            )

    ax.set_xticks(ax.get_xticks())  # To remove a matplotlib warning
    ax.set_xticklabels(labels=xtick_labels)
    ax.set_xlabel(xlabel)

    if title is None:
        title = f"Sleep stage continuity for night {night_id}"
    ax.set_title(title)
    ax.set_ylabel("number of changes")
    return ax


def plot_sleep_stage_change(
    sleep_stages_changements: pd.DataFrame,
    night_id: str,
    mask_diag: bool = False,
    cmap: str = "viridis",
    tick_labels: list = ["Awake", "N1", "N2", "N3", "REM"],
) -> Axes:
    """Plot the sleep stage changes as a heatmap.

    Args:
        sleep_stages_changements (pd.DataFrame): Data from `sleep_eeg.pre_processing.sleep_cycles.get_sleep_stage_changement`
        night_id (str): ID of the night to plot.
        mask_diag (bool, optional): Wether to remove the diagonal by filling it in white. Dafaults to False.
        cmap (str, optional): The colormap to use. Dafaults to "viridis".
        tick_labels (list, optional): The tick labels. Defaults to ["Awake", "N1", "N2", "N3", "REM"].

    Returns:
        Axes: axis
    """
    if mask_diag:
        mask = np.eye(5)
    else:
        mask = np.zeros((5, 5))
    night_sleep_stage_change = pd.DataFrame(
        index=[0, 1, 2, 3, 5], columns=[0, 1, 2, 3, 5]
    )
    for key, value in sleep_stages_changements.loc[night_id].to_dict().items():
        night_sleep_stage_change.loc[int(key[6]), int(key[-1])] = value
        night_sleep_stage_change = night_sleep_stage_change.fillna(0)
    ax = sns.heatmap(
        night_sleep_stage_change,
        annot=True,
        cmap=cmap,
        mask=mask,
        annot_kws={"size": 16},
    )
    ax.set_ylabel("from sleep stage", fontsize=16)
    ax.set_xlabel("to sleep stage", fontsize=16)
    # ax.set_title(f"Sleep stage changes for night {night_id}")
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_yticklabels(tick_labels, fontsize=16)
    return ax


def plot_average_sleep_stage_epochs(sleep_durations_latencies_df: pd.DataFrame) -> Axes:
    """Plot average sleep stage epochs across all nights.

    Args:
        sleep_durations_latencies_df (pd.DataFrame): Data from `sleep_eeg.pre_processing.sleep_cycles.extract_sleep_times_lantencies_wake_up_info`

    Returns:
        Axes: axis
    """
    ax = sns.barplot(
        sleep_durations_latencies_df[
            [
                "lay_awake_time",
                "stage1_time",
                "stage2_time",
                "stage3_time",
                "stage5_time",
            ]
        ]
    )
    ax.set_title(
        f"Average sleep stage epochs for all nights (n={sleep_durations_latencies_df.shape[0]})"
    )
    ax.set_ylabel("epoch [30s]")
    ax.set_xticks(ax.get_xticks())  # To remove a matplotlib warning
    ax.set_xticklabels(labels=[0, 1, 2, 3, 5])
    ax.set_xlabel("sleep stage")
    return ax
