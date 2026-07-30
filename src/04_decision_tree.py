# %% [markdown]
# # 04 — Decision Tree Regression
#
# Train and evaluate a Decision Tree regression model using the processed
# King County House Sales dataset.
#
# This script demonstrates:
#
# - Reusing the same supervised-learning workflow
# - Training a nonlinear regression model
# - Comparing training and testing performance
# - Inspecting tree depth and number of leaves
# - Examining feature importance
#
# A Decision Tree repeatedly splits the data into smaller groups using rules
# such as:
#
#     sqft_living <= 2,450
#     grade <= 8
#     waterfront <= 0
#
# The final groups are called leaves. Each leaf produces a predicted price.

# %%
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


# %% [markdown]
# ## Project paths

# %%
# The script is stored inside the project's scripts directory.
# Moving up two directory levels gives us the project root.
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


# %% [markdown]
# ## Helper functions

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
        If the processed dataset cannot be found.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{data_path}\n\n"
            "Run the data-setup script before training the model."
        )

    dataframe = pl.read_parquet(data_path)

    print("Dataset loaded")
    print("-" * 60)
    print(
        f"Shape: {dataframe.height:,} rows × "
        f"{dataframe.width:,} columns"
    )

    return dataframe


# %%
def validate_columns(dataframe: pl.DataFrame) -> None:
    """Confirm that all required feature and target columns are present."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns:\n"
            f"{missing_columns}"
        )


# %%
def inspect_null_values(dataframe: pl.DataFrame) -> None:
    """Display null counts for the selected features and target."""
    feature_null_counts = dataframe.select(
        pl.col(FEATURE_COLUMNS).null_count()
    )

    total_feature_nulls = sum(
        feature_null_counts.row(0)
    )

    target_nulls = dataframe[TARGET_COLUMN].null_count()

    print("\nNull counts in selected features:")
    print(feature_null_counts)

    print(f"\nTotal feature nulls: {total_feature_nulls:,}")
    print(f"Target nulls:        {target_nulls:,}")

    if total_feature_nulls > 0 or target_nulls > 0:
        raise ValueError(
            "Missing values were found in the model data. "
            "Handle them before training the Decision Tree."
        )


# %%
def prepare_arrays(
    dataframe: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the selected features and target into NumPy arrays.

    Parameters
    ----------
    dataframe
        Housing dataset containing all required columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix X and target vector y.
    """
    X = dataframe.select(FEATURE_COLUMNS).to_numpy()
    y = dataframe[TARGET_COLUMN].to_numpy()

    print("\nNumPy array shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    return X, y


# %%
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the feature matrix and target into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/test split:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# %%
def build_pipeline() -> Pipeline:
    """Create a Decision Tree regression pipeline.

    Notes
    -----
    max_depth=None allows the tree to keep growing until its other stopping
    conditions are reached.

    This is useful for demonstrating how an unrestricted tree can fit its
    training data extremely well, but it may also overfit.

    random_state makes the model reproducible.
    """
    decision_tree = DecisionTreeRegressor(
        max_depth=5,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("decision_tree", decision_tree),
        ]
    )

    return pipeline


# %%
def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r_squared = r2_score(actual, predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r_squared": r_squared,
    }


# %%
def print_model_results(
    training_metrics: dict[str, float],
    testing_metrics: dict[str, float],
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print model structure, metrics, and sample predictions."""
    decision_tree = pipeline.named_steps["decision_tree"]

    pipeline_score = pipeline.score(X_test, y_test)

    print("\nDecision Tree Regression results")
    print("-" * 60)

    print(f"Tree depth:           {decision_tree.get_depth():,}")
    print(f"Number of leaves:     {decision_tree.get_n_leaves():,}")

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


# %%
def print_sample_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    number_to_display: int = 10,
) -> None:
    """Print a small sample of actual and predicted house prices."""
    print("\nFirst 10 test-set predictions:")
    print("-" * 60)

    for position, (actual_value, predicted_value) in enumerate(
        zip(
            actual[:number_to_display],
            predicted[:number_to_display],
        ),
        start=1,
    ):
        absolute_error = abs(actual_value - predicted_value)

        print(
            f"{position:>2}. "
            f"Actual: ${actual_value:>12,.2f} | "
            f"Predicted: ${predicted_value:>12,.2f} | "
            f"Absolute error: ${absolute_error:>12,.2f}"
        )


# %%
def create_feature_importance_table(
    pipeline: Pipeline,
) -> pl.DataFrame:
    """Create a table of Decision Tree feature importance values."""
    decision_tree = pipeline.named_steps["decision_tree"]

    feature_importance_table = (
        pl.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": decision_tree.feature_importances_,
            }
        )
        .sort(
            "importance",
            descending=True,
        )
    )

    return feature_importance_table


# %%
def print_feature_importance(
    feature_importance_table: pl.DataFrame,
) -> None:
    """Display the Decision Tree feature importance table."""
    print("\nDecision Tree feature importance:")
    print("-" * 60)
    print(feature_importance_table)


# %% [markdown]
# ## Main program

# %%
def main() -> None:
    """Train and evaluate the Decision Tree regression model."""

    # ------------------------------------------------------------------
    # 1. Load the processed King County housing dataset.
    #
    # At this point, the raw data should already have been downloaded,
    # cleaned, and saved as a Parquet file by the data-setup script.
    # ------------------------------------------------------------------
    dataframe = load_dataset(DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Confirm that the dataset contains every feature and target column
    # required by this model.
    # ------------------------------------------------------------------
    validate_columns(dataframe)

    # ------------------------------------------------------------------
    # 3. Check for missing values.
    #
    # Scikit-learn's DecisionTreeRegressor expects the feature matrix and
    # target vector used here to contain valid numerical values.
    # ------------------------------------------------------------------
    inspect_null_values(dataframe)

    # ------------------------------------------------------------------
    # 4. Separate the model inputs from the value we want to predict.
    #
    # X contains the 16 house characteristics used as predictor features.
    # y contains the corresponding house sale prices.
    # ------------------------------------------------------------------
    X, y = prepare_arrays(dataframe)

    # ------------------------------------------------------------------
    # 5. Divide the observations into training and testing datasets.
    #
    # The model learns patterns from the training data.
    # The untouched test data acts as the model's final exam.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # ------------------------------------------------------------------
    # 6. Build the machine learning pipeline.
    #
    # The pipeline currently contains one step: DecisionTreeRegressor.
    # Keeping the estimator inside a Pipeline gives every model script the
    # same fit-and-predict interface and makes the workflow easy to extend.
    # ------------------------------------------------------------------
    pipeline = build_pipeline()

    # ------------------------------------------------------------------
    # 7. Fit the Decision Tree using only the training observations.
    #
    # During fit(), the tree searches for feature thresholds that divide
    # houses into increasingly similar groups based on sale price.
    # ------------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 8. Generate predictions for both datasets.
    #
    # Training predictions reveal how closely the tree fits data it has
    # already seen.
    #
    # Test predictions reveal how well the learned rules generalize to new,
    # unseen houses.
    # ------------------------------------------------------------------
    training_predictions = pipeline.predict(X_train)
    testing_predictions = pipeline.predict(X_test)

    # ------------------------------------------------------------------
    # 9. Calculate separate training and testing metrics.
    #
    # A very large gap between training and testing performance is an
    # important warning sign that the Decision Tree may be overfitting.
    # ------------------------------------------------------------------
    training_metrics = calculate_metrics(
        actual=y_train,
        predicted=training_predictions,
    )

    testing_metrics = calculate_metrics(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 10. Display the tree's structure and overall predictive performance.
    # ------------------------------------------------------------------
    print_model_results(
        training_metrics=training_metrics,
        testing_metrics=testing_metrics,
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
    )

    # ------------------------------------------------------------------
    # 11. Inspect several individual test-set predictions.
    #
    # Aggregate metrics summarize thousands of predictions. Looking at a
    # few individual examples helps connect those metrics to real houses.
    # ------------------------------------------------------------------
    print_sample_predictions(
        actual=y_test,
        predicted=testing_predictions,
    )

    # ------------------------------------------------------------------
    # 12. Examine which features the tree relied on most heavily when
    # creating its splitting rules.
    #
    # Feature importance values sum to 1.0. A larger value means the feature
    # contributed more to reducing prediction error across the tree.
    # ------------------------------------------------------------------
    feature_importance_table = create_feature_importance_table(
        pipeline
    )

    print_feature_importance(feature_importance_table)


# %%
if __name__ == "__main__":
    main()