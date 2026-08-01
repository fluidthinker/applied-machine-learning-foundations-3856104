# %% [markdown]
# # 06 — Decision Tree Grid Search
#
# Use GridSearchCV to find a strong combination of Decision Tree
# hyperparameters.
#
# Workflow:
#
#     Complete Dataset
#            ↓
#     Train/Test Split
#            ↓
#     Training Data
#            ↓
#     Hyperparameter Grid
#            ↓
#     5-Fold Cross-Validation
#            ↓
#     Select Best Hyperparameters
#            ↓
#     Refit Best Model on All Training Data
#            ↓
#     Evaluate Once on Untouched Test Data
#
# GridSearchCV automates the manual experiment we previously performed
# with:
#
#     max_depth = 5
#     max_depth = 10
#     max_depth = None
#
# Instead of changing one value by hand, GridSearchCV evaluates every
# specified hyperparameter combination using cross-validation.


# %%
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
from sklearn.tree import DecisionTreeRegressor


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
# ## Hyperparameter grid
#
# A hyperparameter is a model setting chosen before training.
#
# This grid searches three Decision Tree hyperparameters:
#
# `max_depth`
#
#     Maximum number of decision levels allowed in the tree.
#
# `min_samples_split`
#
#     Minimum number of observations required before a node may split.
#
# `min_samples_leaf`
#
#     Minimum number of observations that must remain in each leaf.
#
# Because the Decision Tree is inside a Pipeline, each parameter name uses:
#
#     pipeline_step__parameter_name
#
# For example:
#
#     decision_tree__max_depth
#
# means:
#
#     Set max_depth on the Pipeline step named decision_tree.


# %%
PARAMETER_GRID: dict[str, list[Any]] = {
    "decision_tree__max_depth": [
        5,
        8,
        10,
        12,
        15,
        None,
    ],
    "decision_tree__min_samples_split": [
        2,
        10,
        25,
    ],
    "decision_tree__min_samples_leaf": [
        1,
        5,
        10,
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
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
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
# The test set remains untouched during:
#
# - hyperparameter search,
# - cross-validation,
# - model selection.
#
# It is used once at the end for final evaluation.


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
# ## Build the Decision Tree pipeline
#
# The Decision Tree begins with its default complexity settings.
#
# GridSearchCV will replace the selected settings with each combination
# from PARAMETER_GRID.


# %%
def build_pipeline() -> Pipeline:
    """Create an untrained Decision Tree regression pipeline."""
    return Pipeline(
        steps=[
            (
                "decision_tree",
                DecisionTreeRegressor(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# %% [markdown]
# ## Create the cross-validation strategy
#
# The same five folds are used to evaluate every hyperparameter
# combination.
#
# This produces a fair comparison because each candidate is evaluated
# using the same training and validation observations.


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
# Name breakdown:
#
#     Grid
#         The collection of hyperparameter combinations.
#
#     Search
#         Try every combination.
#
#     CV
#         Evaluate each combination using cross-validation.
#
# `scoring="r2"`
#
#     Select the combination with the highest mean validation R².
#
# `refit=True`
#
#     After finding the best combination, train that best model again
#     using the complete training dataset.
#
# `return_train_score=True`
#
#     Store training-fold scores as well as validation-fold scores.
#     This helps us inspect the overfitting/underfitting tradeoff.
#
# `n_jobs=-1`
#
#     Use all available processor cores to run fits in parallel.


# %%
def create_grid_search(
    pipeline: Pipeline,
    cross_validation: KFold,
) -> GridSearchCV:
    """Create the Decision Tree hyperparameter search."""
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
# ## Describe the size of the search
#
# ParameterGrid expands the dictionary into every unique combination.
#
# For this script:
#
#     6 max_depth values
#     ×
#     3 min_samples_split values
#     ×
#     3 min_samples_leaf values
#     =
#     54 combinations
#
# With five-fold cross-validation:
#
#     54 combinations × 5 folds
#     =
#     270 model fits
#
# GridSearchCV then performs one additional refit using the best
# combination and all available training data.


# %%
def print_search_plan() -> None:
    """Display the number of candidates and cross-validation fits."""
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
#
# Important fitted attributes:
#
# `best_params_`
#
#     Hyperparameter combination with the highest mean validation score.
#
# `best_score_`
#
#     Mean cross-validation R² for that combination.
#
# `best_estimator_`
#
#     Pipeline refitted on all training data using the best combination.


# %%
def print_best_search_result(
    grid_search: GridSearchCV,
) -> None:
    """Print the best hyperparameters and validation score."""
    best_tree = grid_search.best_estimator_.named_steps[
        "decision_tree"
    ]

    print("\nBest cross-validation result")
    print("-" * 60)

    print("Best hyperparameters:")

    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        readable_name = parameter_name.replace(
            "decision_tree__",
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
        f"Refitted tree depth:      "
        f"{best_tree.get_depth():,}"
    )

    print(
        f"Refitted tree leaves:     "
        f"{best_tree.get_n_leaves():,}"
    )


# %% [markdown]
# ## Display the highest-scoring candidates
#
# GridSearchCV stores the result of every candidate in `cv_results_`.
#
# We sort candidates by `rank_test_score` and display the five best.
#
# The "test" scores inside cv_results_ are validation-fold scores.
# They are not scores from our untouched final test set.


# %%
def create_top_results_table(
    grid_search: GridSearchCV,
    number_to_display: int = 5,
) -> pl.DataFrame:
    """Create a table containing the highest-ranked candidates."""
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
                "max_depth": parameters[
                    "decision_tree__max_depth"
                ],
                "min_samples_split": parameters[
                    "decision_tree__min_samples_split"
                ],
                "min_samples_leaf": parameters[
                    "decision_tree__min_samples_leaf"
                ],
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

    return pl.DataFrame(rows).sort("rank")


# %% [markdown]
# ## Calculate final regression metrics


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
# ## Print final model performance


# %%
def print_final_model_results(
    grid_search: GridSearchCV,
    training_metrics: dict[str, float],
    testing_metrics: dict[str, float],
) -> None:
    """Print training and final test-set performance."""
    print("\nFinal tuned Decision Tree results")
    print("-" * 60)

    print("Training-set performance:")
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
        f"GridSearchCV .score():    "
        f"{grid_search.best_estimator_.score(
            X_test_global,
            y_test_global,
        ):>10.4f}"
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


# %%
def create_feature_importance_table(
    grid_search: GridSearchCV,
) -> pl.DataFrame:
    """Create feature importances from the best fitted tree."""
    best_tree = grid_search.best_estimator_.named_steps[
        "decision_tree"
    ]

    return (
        pl.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": best_tree.feature_importances_,
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
    """Run the Decision Tree GridSearchCV workflow."""

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
    # ------------------------------------------------------------------
    X, y = prepare_arrays(dataframe)

    # ------------------------------------------------------------------
    # 5. Reserve 20% of the observations as an untouched final test set.
    #
    # GridSearchCV receives only X_train and y_train.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(
        X=X,
        y=y,
    )

    # ------------------------------------------------------------------
    # 6. Build an unfitted Decision Tree pipeline.
    #
    # GridSearchCV will clone this pipeline for every candidate and fold.
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
    # - the Decision Tree pipeline,
    # - the hyperparameter grid,
    # - the cross-validation strategy,
    # - the R² scoring rule.
    # ------------------------------------------------------------------
    grid_search = create_grid_search(
        pipeline=pipeline,
        cross_validation=cross_validation,
    )

    # ------------------------------------------------------------------
    # 9. Display how much work the exhaustive search will perform.
    # ------------------------------------------------------------------
    print_search_plan()

    # ------------------------------------------------------------------
    # 10. Run the complete hyperparameter search using training data only.
    #
    # For every candidate, GridSearchCV:
    #
    # - trains on four folds,
    # - validates on one fold,
    # - rotates through all five folds,
    # - averages the validation R² scores.
    #
    # Because refit=True, it then trains the winning pipeline one final
    # time using all of X_train and y_train.
    # ------------------------------------------------------------------
    grid_search.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 11. Inspect the winning hyperparameters and cross-validation score.
    # ------------------------------------------------------------------
    print_best_search_result(grid_search)

    # ------------------------------------------------------------------
    # 12. Compare the five highest-ranked hyperparameter combinations.
    # ------------------------------------------------------------------
    top_results = create_top_results_table(
        grid_search=grid_search,
        number_to_display=5,
    )

    print("\nTop five GridSearchCV candidates")
    print("-" * 60)
    print(top_results)

    # ------------------------------------------------------------------
    # 13. Retrieve the best fitted pipeline.
    #
    # GridSearchCV already refitted this pipeline using the complete
    # training dataset.
    # ------------------------------------------------------------------
    best_pipeline = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 14. Generate predictions for the training and final test datasets.
    # ------------------------------------------------------------------
    training_predictions = best_pipeline.predict(X_train)
    testing_predictions = best_pipeline.predict(X_test)

    # ------------------------------------------------------------------
    # 15. Calculate training performance.
    #
    # Compare this with mean cross-validation performance to assess
    # whether the selected model may still be overfitting.
    # ------------------------------------------------------------------
    training_metrics = calculate_metrics(
        actual=y_train,
        predicted=training_predictions,
    )

    # ------------------------------------------------------------------
    # 16. Evaluate the selected model on the untouched test data.
    #
    # This is the first time the final test observations participate in
    # model evaluation.
    # ------------------------------------------------------------------
    testing_metrics = calculate_metrics(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 17. Display the final training and test metrics.
    # ------------------------------------------------------------------
    print("\nFinal tuned Decision Tree results")
    print("-" * 60)

    print("Training-set performance:")
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
        f"{best_pipeline.score(X_test, y_test):>10.4f}"
    )

    # ------------------------------------------------------------------
    # 18. Inspect several individual test-set predictions.
    # ------------------------------------------------------------------
    print_sample_predictions(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 19. Examine feature importance from the selected Decision Tree.
    # ------------------------------------------------------------------
    feature_importance_table = (
        create_feature_importance_table(grid_search)
    )

    print("\nBest Decision Tree feature importance")
    print("-" * 60)
    print(feature_importance_table)


# %%
if __name__ == "__main__":
    main()
# %%
