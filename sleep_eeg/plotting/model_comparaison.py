import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def barplot_models_parameters_responses(
    models_CV_score: pd.DataFrame, score_column_name: str, palette: str = "rocket"
):
    """Create a bar plot of model parameters against response variables.

    Args:
        models_CV_score (pd.DataFrame): DataFrame containing cross-validation scores for each model and predictor combination.
        score_column_name (str): Name of the score column to be plotted.
        palette (str, optional): Color palette for the bar plot. Defaults to "rocket".

    Returns:
        sns.FacetGrid: FacetGrid object representing the created bar plot.
    """
    g = sns.FacetGrid(
        data=models_CV_score,
        col="y_name",
        row="model",
        sharey=True,
        sharex="col",
        margin_titles=True,
    )
    g.map_dataframe(
        sns.barplot,
        x=score_column_name,
        y="predictors",
        palette=palette,
        legend=False,
        hue="predictors",
    )
    g.add_legend()
    return g


def heatmap_scores_params_models_responses(
    models_CV_score: pd.DataFrame,
    score_name: str,
    nrows: int = 2,
    ncols: int = 5,
    figsize: tuple = (30, 10),
    cmap=sns.cm.rocket_r,
):
    """Create a heatmap of average scores for different model parameters and response variables.

    Args:
        models_CV_score (pd.DataFrame): DataFrame containing cross-validation scores for each model and predictor combination.
        score_name (str): Name of the score column to be plotted.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to 2.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 5.
        figsize (tuple, optional): Size of the figure. Defaults to (30, 10).
        cmap (sns.cm, optional): Colormap for the heatmap. Defaults to sns.cm.rocket_r.

    Returns:
        Tuple[plt.Figure, np.ndarray]: Tuple containing the created figure and axes.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    response_variables = np.unique(models_CV_score["y_name"])

    for ax, qual_metric in zip(axes.flatten(), response_variables):
        sub_df = models_CV_score[models_CV_score["y_name"] == qual_metric][
            ["model", "predictors", score_name]
        ].astype({score_name: float})
        average_score_df = (
            sub_df.groupby(["model", "predictors"])[score_name].mean().reset_index()
        )
        sns.heatmap(
            average_score_df.pivot(
                columns="model", index="predictors", values=score_name
            ),
            annot=True,
            fmt=".5g",
            ax=ax,
            cmap=cmap,
        )
        ax.set_title(qual_metric)
    plt.tight_layout()
    return fig, axes


def relplot_true_pred_identity_params_responses(
    models_CV_score: pd.DataFrame,
    model_name: str,
    row: str = "predictors",
    col: str = "y_name",
    hue: str = "predictors",
    style: str = None,
    sharex: bool = False,
    sharey: bool = False,
):
    """Create a scatter plot of true vs predicted values for a specific model.

    Args:
        models_CV_score (pd.DataFrame): DataFrame containing cross-validated predictions for each model and predictor combination.
        model_name (str): Name of the model to be plotted.
        row (str, optional): Variable to define rows in the subplot grid. Defaults to "predictors".
        col (str, optional): Variable to define columns in the subplot grid. Defaults to "y_name".
        hue (str, optional): Variable to define color in the plot. Defaults to "predictors".
        style (str, optional): Variable to define style in the plot. Defaults to None.
        sharex (bool, optional): Whether to share x-axis across subplots. Defaults to False.
        sharey (bool, optional): Whether to share y-axis across subplots. Defaults to False.

    Returns:
        sns.FacetGrid: FacetGrid object representing the created scatter plot.
    """
    rel = sns.relplot(
        data=models_CV_score[models_CV_score["model"] == model_name],
        x="y_pred",
        y="y_true",
        row=row,
        col=col,
        hue=hue,
        style=style,
        # row="predictors",
        # col="y_name",
        # col="model",
        # row="y_name",
        # hue="predictors",
        # style="predictors",
        kind="scatter",
        facet_kws={"sharey": sharey, "sharex": sharex, "margin_titles": True},
    )

    for ax in rel.fig.axes:
        ax.set_xlim(ax.get_ylim())
        ax.set_aspect("equal")
        ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)

    return rel
