from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


def optimise_models_get_CV_scores(
    models: list,
    search_spaces_per_model: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str,
    n_iter: int,
    y_name: str,  # GGGG
    preprocessing_pipe: Pipeline = None,
    class_weight: dict = None,
    model_naming: Callable = lambda model: model.__class__.__name__,
    n_jobs: int = 1,
    df_hyp_rmse: pd.DataFrame = None,  # GGGG
) -> tuple[pd.DataFrame, list]:
    # best_params = [] #GGGG
    # opt_results = []
    step = cv.get_n_splits(X)
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            "test_score",
            "train_score",
        ],
        index=range(len(models) * step),
    )
    for i, (model, search_spaces) in enumerate(zip(models, search_spaces_per_model)):
        if class_weight is not None and hasattr(model, "class_weight"):
            model = model.set_params(class_weight=class_weight)
        model_name = model_naming(model)
        print(f"Evaluating model: {model_name}")
        if preprocessing_pipe is not None:
            pipe = deepcopy(preprocessing_pipe)
            pipe.steps.append(("model", model))
            search_spaces = {f"model__{key}": val for key, val in search_spaces.items()}
        else:
            pipe = model

        opt = BayesSearchCV(
            estimator=pipe,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=0,
            return_train_score=True,
            refit=False,
            n_jobs=n_jobs,
        ).fit(X, y)

        # GGGG
        test_scores = [
            opt.cv_results_[f"split{j}_test_score"][opt.best_index_]
            for j in range(step)
        ]
        train_scores = [
            opt.cv_results_[f"split{j}_train_score"][opt.best_index_]
            for j in range(step)
        ]
        models_CV_scores["test_score"].iloc[i * step : (i + 1) * step] = test_scores
        models_CV_scores["train_score"].iloc[i * step : (i + 1) * step] = train_scores
        # GGGG
        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model_name
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)

        # best_params.append(opt.best_params_)#GGGG
        # opt_results.append(opt.optimizer_results_[0])
        df_hyp_rmse.at[i, "model"] = model_name
        df_hyp_rmse.at[i, "hyp_" + y_name] = [opt.best_params_]
        df_hyp_rmse.at[i, "train_score_" + y_name] = -np.array(train_scores).mean()
        df_hyp_rmse.at[i, "test_score_" + y_name] = -np.array(test_scores).mean()
        best_params = df_hyp_rmse
    return models_CV_scores, best_params


def fit_models_get_CV_scores(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str | dict,
    preprocessing_pipe: Pipeline = None,
    class_weight: dict = None,
    model_naming: Callable = lambda model: model.__class__.__name__,
) -> pd.DataFrame:
    """Fit models and get cross-validation scores."""
    step = cv.get_n_splits(X)
    if not isinstance(scoring, dict):
        scoring = {scoring: scoring}
    test_scorings_column_names = [
        f"test_{scoring_name}" for scoring_name in scoring.keys()
    ]
    train_scorings_column_names = [
        f"train_{scoring_name}" for scoring_name in scoring.keys()
    ]
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            "fit_time",
            *test_scorings_column_names,
            *train_scorings_column_names,
        ],
        index=range(len(models) * step),
    )
    for i, model in enumerate(models):
        if class_weight is not None and hasattr(model, "class_weight"):
            model = model.set_params(class_weight=class_weight)
        model_name = model_naming(model)
        print(f"Evaluating model: {model_name}")
        if preprocessing_pipe is not None:
            pipe = deepcopy(preprocessing_pipe)
            pipe.steps.append(("model_used", model))
        else:
            pipe = model
        model_CV_scores = cross_validate(
            estimator=pipe, X=X, y=y, cv=cv, scoring=scoring, return_train_score=True
        )
        for test_scoring_name in test_scorings_column_names:
            models_CV_scores[test_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[test_scoring_name]
        for train_scoring_name in train_scorings_column_names:
            models_CV_scores[train_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[train_scoring_name]
        models_CV_scores["fit_time"].iloc[i * step : (i + 1) * step] = model_CV_scores[
            "fit_time"
        ]

        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model_name
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)

    return models_CV_scores


def fit_models_get_CV_scores_across_mqs(
    models: list,
    X: pd.DataFrame,
    ys: pd.DataFrame,
    cv,
    scoring: str | dict,
    preprocessing_pipe: Pipeline = None,
    class_weights: dict = None,
    model_naming: Callable = lambda model: model.__class__.__name__,
) -> pd.DataFrame:
    models_CV_scores_across_mqs = pd.DataFrame()
    for y_name in ys.columns:
        if class_weights is not None:
            class_weight = class_weights[y_name]
        else:
            class_weight = None
        result_df = fit_models_get_CV_scores(
            models=models,
            X=X,
            y=ys[y_name],
            cv=cv,
            scoring=scoring,
            preprocessing_pipe=preprocessing_pipe,
            class_weight=class_weight,
            model_naming=model_naming,
        )
        result_df["y_name"] = y_name
        models_CV_scores_across_mqs = pd.concat(
            [models_CV_scores_across_mqs, result_df], ignore_index=True, axis=0
        )
    return models_CV_scores_across_mqs


def optimise_models_get_CV_scores_across_mqs(
    models: list,
    search_spaces_per_model: list[dict],
    X: pd.DataFrame,
    ys: pd.DataFrame,
    cv,
    scoring: str,
    n_iter: int,
    preprocessing_pipe: Pipeline = None,
    class_weights: dict = None,
    model_naming: Callable = lambda model: model.__class__.__name__,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict]:
    models_CV_scores_across_mqs = pd.DataFrame()
    # models_best_params_across_mqs = {} #GGGG
    best_y_params_per_model = pd.DataFrame(
        columns=["model"]
        + [
            resp
            for resp_var in ys.columns
            for resp in (
                "hyp_" + resp_var,
                "train_score_" + resp_var,
                "test_score_" + resp_var,
            )
        ],
        index=np.arange(len(models)),
    )  # Dataframe that will progressively be filled with the best hyperparameters and the corresponding CV rmse
    for y_name in ys.columns:
        if class_weights is not None:
            class_weight = class_weights[y_name]
        else:
            class_weight = None
        result_df, best_y_params_per_model = optimise_models_get_CV_scores(
            models=models,
            search_spaces_per_model=search_spaces_per_model,
            X=X,
            y=ys[y_name],
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            preprocessing_pipe=preprocessing_pipe,
            class_weight=class_weight,
            model_naming=model_naming,
            n_jobs=n_jobs,
            df_hyp_rmse=best_y_params_per_model,  # GGGG
            y_name=y_name,  # GGGG
        )
        result_df["y_name"] = y_name
        models_CV_scores_across_mqs = pd.concat(
            [models_CV_scores_across_mqs, result_df], ignore_index=True, axis=0
        )
        # models_best_params_across_mqs[y_name] = best_y_params_per_model
    return models_CV_scores_across_mqs, best_y_params_per_model
