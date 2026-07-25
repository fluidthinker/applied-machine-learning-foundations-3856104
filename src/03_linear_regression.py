# %% [markdown]
# # 03 — Linear Regression
#
# Train and evaluate a Linear Regression model using the processed
# King County House Sales dataset.
#
# Workflow:
#
#     Processed housing data
#               ↓
#     Select features and target
#               ↓
#     Train/test split
#               ↓
#     Linear Regression pipeline
#               ↓
#     Fit on training data
#               ↓
#     Predict test-set prices
#               ↓
#     Evaluate with MAE, RMSE, and R²
#
# This script intentionally follows the same general workflow as
# 02_baseline.py.
#
# The important change is the model:
#
#     DummyRegressor
#           ↓
#     LinearRegression
#
# The DummyRegressor ignored the features and predicted the mean
# training-set price for every house.
#
# Linear Regression examines the features and learns a coefficient
# for each one.


# %%
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# %% [markdown]
# ## Project and data paths
#
# This script is expected to live at:
#
#     project_root/
#     └── src/
#         └── ml_foundations/
#             └── 03_linear_regression.py
#
# Starting from this file:
#
#     .parent              → src 
#     .parent.parent       → project root
#  


# %%
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "king_county_house_sales.parquet"
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data path:    {DATA_PATH}")


# %% [markdown]
# ## Model configuration
#
# These should be the same feature columns used in 02_baseline.py.
#
# We exclude:
#
# - `id`: an identifier rather than a meaningful predictor
# - `price`: the target
# - `zipcode`: categorical and would require encoding
# - `date`: would require explicit date feature engineering
# - `yr_renovated`: contains null values
#
# We can add those variables later through an explicit preprocessing
# and feature-engineering strategy.


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
# ## Load the prepared dataset


# %%
def load_processed_data(path: Path) -> pl.DataFrame:
    """Load the processed King County housing dataset.

    Args:
        path: Path to the processed Parquet file.

    Returns:
        The processed housing data as a Polars DataFrame.

    Raises:
        FileNotFoundError: If the processed dataset does not exist.
        ValueError: If the dataset contains no rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            "Processed housing dataset not found.\n"
            f"Expected location: {path}\n\n"
            "Run 01_data_setup.py before running this script."
        )

    dataframe = pl.read_parquet(path)

    if dataframe.is_empty():
        raise ValueError("The processed housing dataset contains no rows.")

    return dataframe


# %% [markdown]
# ## Validate the selected columns
#
# `set()` creates a Python set: a collection of unique values.
#
# We use set difference to ask:
#
#     Which required columns are not present in the dataset?
#
# For example:
#
#     required = {"price", "bedrooms", "bathrooms"}
#     available = {"price", "bedrooms"}
#
#     required.difference(available)
#
# returns:
#
#     {"bathrooms"}


# %%
def validate_required_columns(dataframe: pl.DataFrame) -> None:
    """Confirm that all selected features and the target exist.

    Args:
        dataframe: Processed housing dataset.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    available_columns = set(dataframe.columns)

    missing_columns = required_columns.difference(available_columns)

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )


# %% [markdown]
# ## Select the features and target
#
# Machine learning commonly uses:
#
#     X = features
#     y = target
#
# `X` is two-dimensional:
#
#     rows × features
#
# `y` is one-dimensional:
#
#     one target value per row


# %%
def select_features_and_target(
    dataframe: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.Series]:
    """Separate the feature matrix from the target vector.

    Args:
        dataframe: Processed housing dataset.

    Returns:
        A tuple containing:
        - X: selected feature columns
        - y: house-price target
    """
    X = dataframe.select(FEATURE_COLUMNS)
    y = dataframe.get_column(TARGET_COLUMN)

    return X, y


# %% [markdown]
# ## Validate missing values
#
# LinearRegression cannot train directly on null values.
#
# This script intentionally stops if nulls are present rather than
# silently inventing an imputation strategy.
#
# Missing-value handling should always be an explicit modeling choice.


# %%
def validate_null_values(
    X: pl.DataFrame,
    y: pl.Series,
) -> None:
    """Confirm that the selected features and target contain no nulls.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Raises:
        ValueError: If the features or target contain null values.
    """
    # Polars returns a one-row DataFrame containing the null count
    # for every selected feature.
    feature_null_counts = X.null_count()

    print("\nNull counts in selected features:")
    print(feature_null_counts)

    # Extract that single row as a Python tuple.
    #
    # Example:
    #
    #     (0, 0, 0, 0)
    #
    feature_null_count_values = feature_null_counts.row(0)

    # Add the individual column counts together.
    total_feature_nulls = sum(feature_null_count_values)

    # A Polars Series returns one integer for its null count.
    target_null_count = y.null_count()

    print(f"\nTotal feature nulls: {total_feature_nulls:,}")
    print(f"Target nulls:        {target_null_count:,}")

    if total_feature_nulls > 0:
        raise ValueError(
            "One or more selected feature columns contain null values.\n"
            "Choose features without null values or add an explicit "
            "missing-value strategy."
        )

    if target_null_count > 0:
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' contains null values."
        )


# %% [markdown]
# ## Convert Polars objects to NumPy arrays
#
# Scikit-learn accepts several tabular data types, including NumPy arrays.
#
# `X.to_numpy()` produces a two-dimensional array:
#
#     (number of rows, number of features)
#
# `y.to_numpy()` is already one-dimensional because `y` is a Polars Series.
#
# `.ravel()` ensures that the target has the shape expected by
# scikit-learn:
#
#     (number of observations,)
#
# rather than:
#
#     (number of observations, 1)


# %%
def convert_to_numpy(
    X: pl.DataFrame,
    y: pl.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Polars feature and target objects to NumPy arrays.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        A tuple containing the NumPy feature matrix and target vector.
    """
    X_array = X.to_numpy()
    y_array = y.to_numpy().ravel()

    print("\nNumPy array shapes:")
    print(f"X: {X_array.shape}")
    print(f"y: {y_array.shape}")

    return X_array, y_array


# %% [markdown]
# ## Split the data
#
# The training set teaches the model.
#
# The testing set evaluates the trained model on observations it did
# not see during training.
#
# `random_state` makes the split reproducible. Running the script again
# produces the same training and testing observations.


# %%
def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Divide the features and target into training and testing sets.

    Args:
        X: Complete feature matrix.
        y: Complete target vector.

    Returns:
        X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/test split:")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Testing samples:  {X_test.shape[0]:,}")
    print(f"Number of features: {X_train.shape[1]:,}")

    return X_train, X_test, y_train, y_test


# %% [markdown]
# ## Create the Linear Regression pipeline
#
# The transcript's most important implementation change is this:
#
#     DummyRegressor(...)
#
# becomes:
#
#     LinearRegression()
#
# We still place the model inside a Pipeline.
#
# At the moment, the pipeline contains only the model. This may seem
# unnecessary, but it gives us a consistent structure that can later
# include preprocessing steps.
#
# For example:
#
#     preprocessing
#          ↓
#     feature transformation
#          ↓
#     model
#
# Linear Regression does not require feature scaling merely to function,
# so we are not adding StandardScaler here.


# %%
def create_linear_regression_pipeline() -> Pipeline:
    """Create a scikit-learn pipeline containing Linear Regression.

    Returns:
        An untrained Linear Regression pipeline.
    """
    pipeline = Pipeline(
        steps=[
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )

    return pipeline


# %% [markdown]
# ## Train the model
#
# Calling `.fit()` is the learning step.
#
# Linear Regression examines:
#
#     X_train → feature values
#     y_train → known house prices
#
# It then estimates:
#
# - one intercept
# - one coefficient for every feature
#
# These learned values define the trained model.


# %%
def train_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Fit the Linear Regression pipeline to the training data.

    Args:
        pipeline: Untrained model pipeline.
        X_train: Training features.
        y_train: Training targets.

    Returns:
        The fitted pipeline.
    """
    pipeline.fit(X_train, y_train)

    return pipeline


# %% [markdown]
# ## Generate predictions
#
# `.predict()` applies the learned equation to each test observation.
#
# Unlike the DummyRegressor, Linear Regression does not return the same
# value for every house.
#
# Each house receives a prediction based on its feature values.


# %%
def generate_predictions(
    pipeline: Pipeline,
    X_test: np.ndarray,
) -> np.ndarray:
    """Generate house-price predictions for the testing data.

    Args:
        pipeline: Fitted Linear Regression pipeline.
        X_test: Testing feature matrix.

    Returns:
        Predicted house prices.
    """
    predictions = pipeline.predict(X_test)

    return predictions


# %% [markdown]
# ## Evaluate the model
#
# We use the same regression metrics used for the baseline model.
#
# MAE:
#
#     Typical absolute prediction error in dollars.
#
# RMSE:
#
#     Prediction error in dollars, with larger mistakes penalized more.
#
# R²:
#
#     Improvement relative to predicting the mean target.
#
# Important correction to the course transcript:
#
# R² is not strictly limited to values between 0 and 1.
#
#     R² = 1
#         Perfect predictions.
#
#     R² = 0
#         No better than predicting the test-set mean.
#
#     R² < 0
#         Worse than that simple reference prediction.


# %%
def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for the test-set predictions.

    Args:
        pipeline: Fitted model pipeline.
        X_test: Testing features.
        y_test: Actual testing targets.
        predictions: Predicted testing targets.

    Returns:
        Dictionary containing MAE, RMSE, R², and pipeline score.
    """
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # For a regressor, pipeline.score(X, y) returns R².
    pipeline_score = pipeline.score(X_test, y_test)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pipeline_score": float(pipeline_score),
    }


# %% [markdown]
# ## Inspect the learned model
#
# Linear Regression learns:
#
#     intercept + one coefficient per feature
#
# In a pipeline, `named_steps` lets us retrieve a particular step by name.
#
# Here:
#
#     pipeline.named_steps["linear_regression"]
#
# returns the fitted LinearRegression object.
#
# Its important learned attributes are:
#
#     .intercept_
#     .coef_
#
# The trailing underscore is a scikit-learn convention indicating that
# the attribute was created during `.fit()`.


# %%
def create_coefficient_table(
    pipeline: Pipeline,
) -> pl.DataFrame:
    """Create a readable table of learned feature coefficients.

    Args:
        pipeline: Fitted Linear Regression pipeline.

    Returns:
        Polars DataFrame containing feature names and coefficients.
    """
    linear_regression = pipeline.named_steps["linear_regression"]

    coefficients = linear_regression.coef_

    coefficient_table = (
        pl.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "coefficient": coefficients,
            }
        )
        .with_columns(
            pl.col("coefficient")
            .abs()
            .alias("absolute_coefficient")
        )
        .sort(
            "absolute_coefficient",
            descending=True,
        )
    )

    return coefficient_table


# %% [markdown]
# ## Display model results


# %%
def print_results(
    pipeline: Pipeline,
    y_test: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, float],
    coefficient_table: pl.DataFrame,
) -> None:
    """Print the model results and a sample of predictions.

    Args:
        pipeline: Fitted Linear Regression pipeline.
        y_test: Actual test-set prices.
        predictions: Predicted test-set prices.
        metrics: Calculated evaluation metrics.
        coefficient_table: Learned feature coefficients.
    """
    linear_regression = pipeline.named_steps["linear_regression"]

    print("\nLinear Regression results")
    print("-" * 60)

    print(f"Intercept: ${linear_regression.intercept_:,.2f}")

    print()
    print(f"Mean Absolute Error:      ${metrics['mae']:,.2f}")
    print(f"Root Mean Squared Error:  ${metrics['rmse']:,.2f}")
    print(f"R² score:                   {metrics['r2']:.4f}")
    print(
        "Pipeline .score():           "
        f"{metrics['pipeline_score']:.4f}"
    )

    print("\nFirst 10 test-set predictions:")
    print("-" * 60)

    number_to_display = min(10, len(predictions))

    for index in range(number_to_display):
        actual_price = y_test[index]
        predicted_price = predictions[index]
        absolute_error = abs(actual_price - predicted_price)

        print(
            f"{index + 1:>2}. "
            f"Actual: ${actual_price:>12,.2f} | "
            f"Predicted: ${predicted_price:>12,.2f} | "
            f"Absolute error: ${absolute_error:>12,.2f}"
        )

    print("\nLearned feature coefficients:")
    print("-" * 60)
    print(coefficient_table)


# %% [markdown]
# ## Run the complete modeling workflow


# %%
def main() -> None:
    """Run the complete Linear Regression workflow."""
    housing = load_processed_data(DATA_PATH)

    print("\nDataset loaded")
    print("-" * 60)
    print(
        f"Shape: {housing.height:,} rows × "
        f"{housing.width:,} columns"
    )

    validate_required_columns(housing)

    X_polars, y_polars = select_features_and_target(housing)

    validate_null_values(
        X=X_polars,
        y=y_polars,
    )

    X, y = convert_to_numpy(
        X=X_polars,
        y=y_polars,
    )

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
    )

    linear_regression_pipeline = (
        create_linear_regression_pipeline()
    )

    linear_regression_pipeline = train_model(
        pipeline=linear_regression_pipeline,
        X_train=X_train,
        y_train=y_train,
    )

    predictions = generate_predictions(
        pipeline=linear_regression_pipeline,
        X_test=X_test,
    )

    metrics = evaluate_model(
        pipeline=linear_regression_pipeline,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )

    coefficient_table = create_coefficient_table(
        pipeline=linear_regression_pipeline,
    )

    print_results(
        pipeline=linear_regression_pipeline,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        coefficient_table=coefficient_table,
    )


# %%
if __name__ == "__main__":
    main()