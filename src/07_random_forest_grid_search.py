
# %% [markdown]
# # 07 — Random Forest Grid Search
#
# Train and tune a Random Forest regression model using:
#
# - An untouched final test set
# - 5-fold cross-validation
# - GridSearchCV
# - Multiple Random Forest hyperparameters
#
# Workflow:
#
#     Complete Dataset
#            ↓
#     Train/Test Split
#            ↓
#     Training Data
#            ↓
#     Random Forest Hyperparameter Grid
#            ↓
#     5-Fold Cross-Validation
#            ↓
#     Select Best Hyperparameters
#            ↓
#     Refit Best Forest on All Training Data
#            ↓
#     Evaluate Once on Untouched Test Data
#
# A Random Forest combines many Decision Trees.
#
# Each tree is intentionally made somewhat different by:
#
# - Training on a bootstrap sample of the observations
# - Considering only a subset of features at each split
#
# For regression, the forest averages the predictions from all trees.


# %%
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# %% [markdown]
# ## Project paths


# %%
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "king_county_house_sales.parquet"
)


# %% [markdown]
# ## Modeling configuration


# %%
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

TARGET_COLUMN = "price"

TEST_SIZE = 0.20
RANDOM_STATE = 42
NUMBER_OF_FOLDS = 5


# %% [markdown]
# ## Random Forest hyperparameter grid
#
# `n_estimators`
#
#     Number of Decision Trees in the forest.
#
# `max_depth`
#
#     Maximum depth allowed for each individual tree.
#
# `min_samples_split`
#
#     Minimum number of observations a node must contain before
#     that node may attempt another split.
#
# `min_samples_leaf`
#
#     Minimum number of observations allowed in a resulting leaf.
#
# `max_features`
#
#     Number or proportion of features each tree may consider when
#     searching for a split.
#
# Because the estimator is inside a Pipeline, each name follows:
#
#     pipeline_step__parameter_name


# %%
PARAMETER_GRID: dict[str, list[Any]] = {
    "random_forest__n_estimators": [
        200,
        400,
    ],
    "random_forest__max_depth": [
        10,
        20,
        None,
    ],
    "random_forest__min_samples_split": [
        2,
        25,
    ],
    "random_forest__min_samples_leaf": [
        1,
        5,
    ],
    "random_forest__max_features": [
        "sqrt",
        0.7,
    ],
}


# %% [markdown]
# ## Load the processed dataset


# %%
def load_dataset(data_path: Path) -> pl.DataFrame:
    """Load the processed King County housing dataset.

    Parameters
    ----------
    data_path
        Path to the processed Parquet file.

    Returns
    -------
    pl.DataFrame
        Loaded housing dataset.

    Raises
    ------
    FileNotFoundError
        If the processed dataset does not exist.
    ValueError
        If the dataset contains no rows.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{data_path}\n\n"
            "Run the data-setup script before training the model."
        )

    dataframe = pl.read_parquet(data_path)

    if dataframe.is_empty():
        raise ValueError("The processed dataset contains no rows.")

    print("Dataset loaded")
    print("-" * 60)
    print(
        f"Shape: {dataframe.height:,} rows × "
        f"{dataframe.width:,} columns"
    )

    return dataframe


# %% [markdown]
# ## Validate required columns


# %%
def validate_columns(dataframe: pl.DataFrame) -> None:
    """Confirm that all required features and the target are present."""
    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    available_columns = set(dataframe.columns)

    missing_columns = required_columns.difference(
        available_columns
    )

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )


# %% [markdown]
# ## Inspect missing values


# %%
def inspect_null_values(dataframe: pl.DataFrame) -> None:
    """Display and validate null counts in the model data."""
    feature_null_counts = dataframe.select(
        pl.col(FEATURE_COLUMNS).null_count()
    )

    total_feature_nulls = sum(
        feature_null_counts.row(0)
    )

    target_null_count = (
        dataframe
        .get_column(TARGET_COLUMN)
        .null_count()
    )

    print("\nNull counts in selected features:")
    print(feature_null_counts)

    print(f"\nTotal feature nulls: {total_feature_nulls:,}")
    print(f"Target nulls:        {target_null_count:,}")

    if total_feature_nulls > 0:
        raise ValueError(
            "One or more selected feature columns contain null values."
        )

    if target_null_count > 0:
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' contains null values."
        )


# %% [markdown]
# ## Prepare NumPy arrays


# %%
def prepare_arrays(
    dataframe: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the feature matrix X and target vector y."""
    X = (
        dataframe
        .select(FEATURE_COLUMNS)
        .to_numpy()
    )

    y = (
        dataframe
        .get_column(TARGET_COLUMN)
        .to_numpy()
        .ravel()
    )

    print("\nNumPy array shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    return X, y


# %% [markdown]
# ## Create the train/test split
#
# GridSearchCV receives only the training data.
#
# The final test set remains untouched during:
#
# - Hyperparameter tuning
# - Cross-validation
# - Model selection


# %%
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split the complete dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/test split:")
    print(f"Training samples:   {X_train.shape[0]:,}")
    print(f"Testing samples:    {X_test.shape[0]:,}")
    print(f"Number of features: {X_train.shape[1]:,}")

    return X_train, X_test, y_train, y_test


# %% [markdown]
# ## Build the Random Forest pipeline
#
# GridSearchCV will clone this pipeline for every:
#
# - Hyperparameter candidate
# - Cross-validation fold
#
# `n_jobs=1` is deliberate.
#
# GridSearchCV itself will parallelize the candidate fits using
# `n_jobs=-1`. Keeping the Random Forest at `n_jobs=1` avoids nested
# parallelism, where both levels compete for all processor cores.


# %%
def build_pipeline() -> Pipeline:
    """Create an untrained Random Forest regression pipeline."""
    random_forest = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            (
                "random_forest",
                random_forest,
            ),
        ]
    )


# %% [markdown]
# ## Create the cross-validation strategy


# %%
def create_cross_validation_strategy() -> KFold:
    """Create a reproducible five-fold cross-validation strategy."""
    return KFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# %% [markdown]
# ## Create GridSearchCV
#
# `scoring="r2"`
#
#     Select the candidate with the highest mean validation R².
#
# `refit=True`
#
#     After selecting the winner, fit that winning pipeline again
#     using the complete training dataset.
#
# `return_train_score=True`
#
#     Store both training-fold and validation-fold scores.
#
# `n_jobs=-1`
#
#     Use all available processor cores to evaluate candidates.


# %%
def create_grid_search(
    pipeline: Pipeline,
    cross_validation: KFold,
) -> GridSearchCV:
    """Create the Random Forest hyperparameter search."""
    return GridSearchCV(
        estimator=pipeline,
        param_grid=PARAMETER_GRID,
        scoring="r2",
        cv=cross_validation,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )


# %% [markdown]
# ## Describe the search size
#
# This grid contains:
#
#     2 n_estimators values
#     ×
#     3 max_depth values
#     ×
#     2 min_samples_split values
#     ×
#     2 min_samples_leaf values
#     ×
#     2 max_features values
#     =
#     48 combinations
#
# With five-fold cross-validation:
#
#     48 × 5 = 240 Random Forest fits
#
# Random Forest fitting is more computationally expensive than fitting
# a single Decision Tree, so this script may take several minutes.


# %%
def print_search_plan() -> None:
    """Display the number of candidates and model fits."""
    number_of_candidates = len(
        list(ParameterGrid(PARAMETER_GRID))
    )

    number_of_cv_fits = (
        number_of_candidates * NUMBER_OF_FOLDS
    )

    print("\nGrid search plan")
    print("-" * 60)

    print(
        f"Hyperparameter combinations: "
        f"{number_of_candidates:,}"
    )

    print(
        f"Cross-validation folds:       "
        f"{NUMBER_OF_FOLDS:,}"
    )

    print(
        f"Total cross-validation fits:  "
        f"{number_of_cv_fits:,}"
    )


# %% [markdown]
# ## Display the best search result


# %%
def print_best_search_result(
    grid_search: GridSearchCV,
) -> None:
    """Print the winning hyperparameters and CV score."""
    best_forest = grid_search.best_estimator_.named_steps[
        "random_forest"
    ]

    print("\nBest cross-validation result")
    print("-" * 60)

    print("Best hyperparameters:")

    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        readable_name = parameter_name.replace(
            "random_forest__",
            "",
        )

        print(
            f"  {readable_name}: "
            f"{parameter_value}"
        )

    print(
        f"\nBest mean validation R²: "
        f"{grid_search.best_score_:.4f}"
    )

    print(
        f"Number of fitted trees:    "
        f"{len(best_forest.estimators_):,}"
    )


# %% [markdown]
# ## Create a table of the highest-scoring candidates
#
# Mixed hyperparameter types are converted to strings so that Polars
# can represent values such as:
#
# - 10
# - None
# - sqrt
# - 0.7
#
# inside the same columns.


# %%
def create_top_results_table(
    grid_search: GridSearchCV,
    number_to_display: int = 10,
) -> pl.DataFrame:
    """Create a table of the highest-ranked candidates."""
    results = grid_search.cv_results_

    ranked_indices = np.argsort(
        results["rank_test_score"]
    )[:number_to_display]

    rows: list[dict[str, Any]] = []

    for index in ranked_indices:
        parameters = results["params"][index]

        rows.append(
            {
                "rank": int(
                    results["rank_test_score"][index]
                ),
                "n_estimators": int(
                    parameters[
                        "random_forest__n_estimators"
                    ]
                ),
                "max_depth": str(
                    parameters[
                        "random_forest__max_depth"
                    ]
                ),
                "min_samples_split": int(
                    parameters[
                        "random_forest__min_samples_split"
                    ]
                ),
                "min_samples_leaf": int(
                    parameters[
                        "random_forest__min_samples_leaf"
                    ]
                ),
                "max_features": str(
                    parameters[
                        "random_forest__max_features"
                    ]
                ),
                "mean_train_r2": float(
                    results["mean_train_score"][index]
                ),
                "mean_validation_r2": float(
                    results["mean_test_score"][index]
                ),
                "validation_std": float(
                    results["std_test_score"][index]
                ),
            }
        )

    return (
        pl.DataFrame(rows)
        .sort("rank")
    )


# %% [markdown]
# ## Calculate regression metrics


# %%
def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and R²."""
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    mse = mean_squared_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(mse)

    r_squared = r2_score(
        actual,
        predicted,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
    }


# %% [markdown]
# ## Display final model results


# %%
def print_final_model_results(
    best_pipeline: Pipeline,
    best_cv_score: float,
    training_metrics: dict[str, float],
    testing_metrics: dict[str, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print CV, training, and final test performance."""
    forest = best_pipeline.named_steps[
        "random_forest"
    ]

    pipeline_score = best_pipeline.score(
        X_test,
        y_test,
    )

    training_cv_gap = (
        training_metrics["r_squared"]
        - best_cv_score
    )

    cv_test_gap = (
        best_cv_score
        - testing_metrics["r_squared"]
    )

    print("\nFinal tuned Random Forest results")
    print("-" * 60)

    print(
        f"Number of trees:         "
        f"{len(forest.estimators_):,}"
    )

    print("\nTraining-set performance:")

    print(
        f"Mean Absolute Error:      "
        f"${training_metrics['mae']:,.2f}"
    )

    print(
        f"Root Mean Squared Error:  "
        f"${training_metrics['rmse']:,.2f}"
    )

    print(
        f"R² score:                 "
        f"{training_metrics['r_squared']:>10.4f}"
    )

    print("\nCross-validation performance:")

    print(
        f"Best mean validation R²:  "
        f"{best_cv_score:>10.4f}"
    )

    print("\nUntouched test-set performance:")

    print(
        f"Mean Absolute Error:      "
        f"${testing_metrics['mae']:,.2f}"
    )

    print(
        f"Root Mean Squared Error:  "
        f"${testing_metrics['rmse']:,.2f}"
    )

    print(
        f"R² score:                 "
        f"{testing_metrics['r_squared']:>10.4f}"
    )

    print(
        f"Best pipeline .score():   "
        f"{pipeline_score:>10.4f}"
    )

    print("\nGeneralization summary")
    print("-" * 60)

    print(
        f"Training − CV R² gap:     "
        f"{training_cv_gap:>10.4f}"
    )

    print(
        f"CV − Test R² gap:         "
        f"{cv_test_gap:>10.4f}"
    )


# %% [markdown]
# ## Display sample predictions


# %%
def print_sample_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    number_to_display: int = 10,
) -> None:
    """Print several final test-set predictions."""
    print("\nFirst 10 test-set predictions:")
    print("-" * 60)

    for position, (actual_value, predicted_value) in enumerate(
        zip(
            actual[:number_to_display],
            predicted[:number_to_display],
        ),
        start=1,
    ):
        absolute_error = abs(
            actual_value - predicted_value
        )

        print(
            f"{position:>2}. "
            f"Actual: ${actual_value:>12,.2f} | "
            f"Predicted: ${predicted_value:>12,.2f} | "
            f"Absolute error: ${absolute_error:>12,.2f}"
        )


# %% [markdown]
# ## Create the feature-importance table
#
# Random Forest feature importance averages information across all
# trees in the forest.
#
# A larger value means the feature contributed more to reducing
# prediction error across the forest.
#
# Importance does not tell us whether the feature increases or
# decreases the predicted price.


# %%
def create_feature_importance_table(
    best_pipeline: Pipeline,
) -> pl.DataFrame:
    """Create feature importances from the fitted Random Forest."""
    forest = best_pipeline.named_steps[
        "random_forest"
    ]

    return (
        pl.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": forest.feature_importances_,
            }
        )
        .sort(
            "importance",
            descending=True,
        )
    )


# %% [markdown]
# ## Main program


# %%
def main() -> None:
    """Run the Random Forest GridSearchCV workflow."""

    # ------------------------------------------------------------------
    # 1. Load the processed King County housing dataset.
    # ------------------------------------------------------------------
    dataframe = load_dataset(DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Confirm that every selected feature and the target are present.
    # ------------------------------------------------------------------
    validate_columns(dataframe)

    # ------------------------------------------------------------------
    # 3. Confirm that the model inputs contain no missing values.
    # ------------------------------------------------------------------
    inspect_null_values(dataframe)

    # ------------------------------------------------------------------
    # 4. Create the feature matrix X and target vector y.
    #
    # X contains the 16 predictor features.
    # y contains the corresponding house sale prices.
    # ------------------------------------------------------------------
    X, y = prepare_arrays(dataframe)

    # ------------------------------------------------------------------
    # 5. Reserve 20% of the observations as the untouched test set.
    #
    # GridSearchCV receives only X_train and y_train.
    # The test data is not used during hyperparameter selection.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(
        X=X,
        y=y,
    )

    # ------------------------------------------------------------------
    # 6. Build an unfitted Random Forest pipeline.
    #
    # GridSearchCV will clone the pipeline for each candidate and fold.
    # ------------------------------------------------------------------
    pipeline = build_pipeline()

    # ------------------------------------------------------------------
    # 7. Define the five reproducible cross-validation folds.
    # ------------------------------------------------------------------
    cross_validation = create_cross_validation_strategy()

    # ------------------------------------------------------------------
    # 8. Create GridSearchCV.
    #
    # This object combines:
    #
    # - The Random Forest pipeline
    # - The hyperparameter grid
    # - The cross-validation strategy
    # - The R² scoring rule
    # ------------------------------------------------------------------
    grid_search = create_grid_search(
        pipeline=pipeline,
        cross_validation=cross_validation,
    )

    # ------------------------------------------------------------------
    # 9. Display the size of the exhaustive search.
    # ------------------------------------------------------------------
    print_search_plan()

    # ------------------------------------------------------------------
    # 10. Run the complete hyperparameter search.
    #
    # Each candidate is evaluated using the same five folds.
    #
    # Because refit=True, GridSearchCV also fits the winning pipeline
    # one final time using all of X_train and y_train.
    # ------------------------------------------------------------------
    grid_search.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 11. Display the winning hyperparameters and mean CV score.
    # ------------------------------------------------------------------
    print_best_search_result(grid_search)

    # ------------------------------------------------------------------
    # 12. Compare the ten highest-ranked candidates.
    # ------------------------------------------------------------------
    top_results = create_top_results_table(
        grid_search=grid_search,
        number_to_display=10,
    )

    print("\nTop ten GridSearchCV candidates")
    print("-" * 60)
    print(top_results)

    # ------------------------------------------------------------------
    # 13. Retrieve the winning pipeline.
    #
    # GridSearchCV has already refitted this pipeline using all of the
    # available training observations.
    # ------------------------------------------------------------------
    best_pipeline = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 14. Generate predictions for the training and test datasets.
    # ------------------------------------------------------------------
    training_predictions = best_pipeline.predict(
        X_train
    )

    testing_predictions = best_pipeline.predict(
        X_test
    )

    # ------------------------------------------------------------------
    # 15. Calculate training-set performance.
    #
    # Comparing training performance with cross-validation performance
    # helps us assess whether the forest may still be overfitting.
    # ------------------------------------------------------------------
    training_metrics = calculate_metrics(
        actual=y_train,
        predicted=training_predictions,
    )

    # ------------------------------------------------------------------
    # 16. Evaluate the winning forest on the untouched final test set.
    #
    # This is the first time the final test observations are used for
    # model evaluation.
    # ------------------------------------------------------------------
    testing_metrics = calculate_metrics(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 17. Display training, cross-validation, and final test results.
    # ------------------------------------------------------------------
    print_final_model_results(
        best_pipeline=best_pipeline,
        best_cv_score=float(grid_search.best_score_),
        training_metrics=training_metrics,
        testing_metrics=testing_metrics,
        X_test=X_test,
        y_test=y_test,
    )

    # ------------------------------------------------------------------
    # 18. Inspect several individual predictions.
    # ------------------------------------------------------------------
    print_sample_predictions(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 19. Examine feature importance averaged across the winning forest.
    # ------------------------------------------------------------------
    feature_importance_table = (
        create_feature_importance_table(best_pipeline)
    )

    print("\nBest Random Forest feature importance")
    print("-" * 60)
    print(feature_importance_table)


# %%
if __name__ == "__main__":
    main()

# %%
