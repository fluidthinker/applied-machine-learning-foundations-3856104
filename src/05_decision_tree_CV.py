# %% [markdown]
# # 05 — Decision Tree Cross-Validation
#
# Train and evaluate a Decision Tree regression model using:
#
# - A train/test split
# - 5-fold cross-validation on the training data
# - A final evaluation on the untouched test data
#
# Workflow:
#
#     Complete dataset
#           ↓
#     Train/Test Split
#           ↓
#     Training Data
#           ↓
#     5-Fold Cross-Validation
#           ↓
#     Train Final Model on All Training Data
#           ↓
#     Evaluate Once on Test Data
#
# Cross-validation gives us a more stable estimate of model performance
# than relying on one validation split.
#
# The test data is not used during cross-validation.

# %%
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
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

# Use the depth that performed best in our manual experiments.
MAX_DEPTH = 10

# Divide the training data into five folds.
NUMBER_OF_FOLDS = 5


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
    """Create the feature matrix X and target vector y.

    Parameters
    ----------
    dataframe
        Housing dataset containing the selected columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two-dimensional feature matrix and one-dimensional target vector.
    """
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
# Cross-validation will occur only inside the training data.
#
# The test data remains untouched until the final evaluation.

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
# The model is constrained to a maximum depth of 10.
#
# This was the best-performing value among our initial manual experiments:
#
# - max_depth=5
# - max_depth=10
# - max_depth=None

# %%
def build_pipeline() -> Pipeline:
    """Create an untrained Decision Tree regression pipeline."""
    pipeline = Pipeline(
        steps=[
            (
                "decision_tree",
                DecisionTreeRegressor(
                    max_depth=MAX_DEPTH,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


# %% [markdown]
# ## Create the cross-validation strategy
#
# KFold divides the training observations into five folds.
#
# During each round:
#
#     Four folds → train the model
#     One fold   → validate the model
#
# Every fold serves as the validation fold once.
#
# `shuffle=True` mixes the observations before creating the folds.
#
# `random_state` ensures that the same folds are created each time
# the script runs.

# %%
def create_cross_validation_strategy() -> KFold:
    """Create a reproducible five-fold cross-validation strategy."""
    return KFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# %% [markdown]
# ## Run cross-validation
#
# `cross_val_score()` creates and trains a fresh copy of the pipeline
# during every fold.
#
# We pass only the training data:
#
#     X_train
#     y_train
#
# The untouched test set is not involved.
#
# `scoring="r2"` means each fold is evaluated using R².

# %%
def run_cross_validation(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cross_validation: KFold,
) -> np.ndarray:
    """Evaluate the pipeline using five-fold cross-validation.

    Parameters
    ----------
    pipeline
        Untrained Decision Tree pipeline.
    X_train
        Training feature matrix.
    y_train
        Training target vector.
    cross_validation
        Rules defining how the folds are created.

    Returns
    -------
    np.ndarray
        One R² score for each validation fold.
    """
    fold_scores = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cross_validation,
        scoring="r2",
    )

    return fold_scores


# %% [markdown]
# ## Display cross-validation results
#
# The mean summarizes the model's typical validation performance.
#
# The standard deviation describes how much performance changes
# across the folds.
#
# A small standard deviation means the model performs consistently.
#
# A large standard deviation means performance depends more heavily
# on which observations appear in a particular fold.

# %%
def print_cross_validation_results(
    fold_scores: np.ndarray,
) -> None:
    """Print individual and summarized cross-validation scores."""
    print("\n5-Fold Cross-Validation results")
    print("-" * 60)

    for fold_number, score in enumerate(
        fold_scores,
        start=1,
    ):
        print(
            f"Fold {fold_number} R²: "
            f"{score:.4f}"
        )

    print("-" * 60)
    print(
        f"Mean validation R²:       "
        f"{fold_scores.mean():.4f}"
    )
    print(
        f"Validation R² std. dev.:  "
        f"{fold_scores.std():.4f}"
    )
    print(
        f"Lowest fold R²:           "
        f"{fold_scores.min():.4f}"
    )
    print(
        f"Highest fold R²:          "
        f"{fold_scores.max():.4f}"
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
# ## Print the fitted model results

# %%
def print_model_results(
    pipeline: Pipeline,
    training_metrics: dict[str, float],
    testing_metrics: dict[str, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print model structure and train/test performance."""
    decision_tree = pipeline.named_steps["decision_tree"]

    pipeline_score = pipeline.score(
        X_test,
        y_test,
    )

    print("\nFinal Decision Tree results")
    print("-" * 60)

    print(f"Configured max depth: {MAX_DEPTH}")
    print(
        f"Actual tree depth:    "
        f"{decision_tree.get_depth():,}"
    )
    print(
        f"Number of leaves:     "
        f"{decision_tree.get_n_leaves():,}"
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

    print("\nTest-set performance:")
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
        f"Pipeline .score():        "
        f"{pipeline_score:>10.4f}"
    )


# %% [markdown]
# ## Display sample predictions

# %%
def print_sample_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    number_to_display: int = 10,
) -> None:
    """Print several test-set predictions and their errors."""
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
    pipeline: Pipeline,
) -> pl.DataFrame:
    """Create a table of learned feature-importance values."""
    decision_tree = pipeline.named_steps["decision_tree"]

    feature_importance_table = (
        pl.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": (
                    decision_tree.feature_importances_
                ),
            }
        )
        .sort(
            "importance",
            descending=True,
        )
    )

    return feature_importance_table


# %% [markdown]
# ## Main program

# %%
def main() -> None:
    """Run the Decision Tree cross-validation workflow."""

    # ------------------------------------------------------------------
    # 1. Load the prepared King County housing dataset.
    # ------------------------------------------------------------------
    dataframe = load_dataset(DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Confirm that every required feature and the target are present.
    # ------------------------------------------------------------------
    validate_columns(dataframe)

    # ------------------------------------------------------------------
    # 3. Confirm that the model inputs contain no missing values.
    # ------------------------------------------------------------------
    inspect_null_values(dataframe)

    # ------------------------------------------------------------------
    # 4. Create the feature matrix X and target vector y.
    #
    # X contains 16 predictor columns.
    # y contains the corresponding house prices.
    # ------------------------------------------------------------------
    X, y = prepare_arrays(dataframe)

    # ------------------------------------------------------------------
    # 5. Reserve 20% of the observations as the final test set.
    #
    # Cross-validation will use only X_train and y_train.
    # X_test and y_test remain untouched until final evaluation.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(
        X=X,
        y=y,
    )

    # ------------------------------------------------------------------
    # 6. Build the Decision Tree pipeline.
    #
    # The pipeline contains a DecisionTreeRegressor with max_depth=10.
    # ------------------------------------------------------------------
    pipeline = build_pipeline()

    # ------------------------------------------------------------------
    # 7. Define how the five cross-validation folds will be created.
    #
    # KFold determines which observations belong to the training and
    # validation portions during each cross-validation round.
    # ------------------------------------------------------------------
    cross_validation = create_cross_validation_strategy()

    # ------------------------------------------------------------------
    # 8. Evaluate the model five times using cross-validation.
    #
    # Each fold trains a new copy of the pipeline on four folds and
    # evaluates it on the remaining fold.
    #
    # The model has still not seen the final test dataset.
    # ------------------------------------------------------------------
    fold_scores = run_cross_validation(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        cross_validation=cross_validation,
    )

    # ------------------------------------------------------------------
    # 9. Display each fold's R² score, followed by the mean and
    # standard deviation across all five validation folds.
    # ------------------------------------------------------------------
    print_cross_validation_results(fold_scores)

    # ------------------------------------------------------------------
    # 10. Fit the final Decision Tree using all available training data.
    #
    # cross_val_score() trained temporary model copies during validation.
    # It did not leave our original pipeline fitted.
    #
    # We therefore fit the pipeline once more using the complete
    # training dataset.
    # ------------------------------------------------------------------
    pipeline.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 11. Generate predictions for both the training and test datasets.
    #
    # Training predictions help us assess model fit.
    # Test predictions provide the final generalization estimate.
    # ------------------------------------------------------------------
    training_predictions = pipeline.predict(X_train)
    testing_predictions = pipeline.predict(X_test)

    # ------------------------------------------------------------------
    # 12. Calculate training metrics.
    #
    # Comparing training performance with cross-validation and test
    # performance helps reveal underfitting or overfitting.
    # ------------------------------------------------------------------
    training_metrics = calculate_metrics(
        actual=y_train,
        predicted=training_predictions,
    )

    # ------------------------------------------------------------------
    # 13. Calculate final test-set metrics.
    #
    # This is the first time the trained model is evaluated using the
    # untouched test observations.
    # ------------------------------------------------------------------
    testing_metrics = calculate_metrics(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 14. Display the final model structure and train/test metrics.
    # ------------------------------------------------------------------
    print_model_results(
        pipeline=pipeline,
        training_metrics=training_metrics,
        testing_metrics=testing_metrics,
        X_test=X_test,
        y_test=y_test,
    )

    # ------------------------------------------------------------------
    # 15. Inspect several individual test-set predictions.
    # ------------------------------------------------------------------
    print_sample_predictions(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 16. Examine which features contributed most to the tree's splits.
    # ------------------------------------------------------------------
    feature_importance_table = (
        create_feature_importance_table(pipeline)
    )

    print("\nDecision Tree feature importance:")
    print("-" * 60)
    print(feature_importance_table)


# %%
if __name__ == "__main__":
    main()
# %%
