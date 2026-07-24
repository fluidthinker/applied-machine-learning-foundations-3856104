# %% [markdown]
# # 02 — Baseline Regression Model
#
# In this script, we create the simplest possible model for predicting
# King County house prices.
#
# The model is called a "dummy" model because it does not learn relationships
# between the housing features and the target.
#
# Instead, it predicts the same value for every house:
#
#     the mean house price in the training data
#
# This gives us a baseline score.
#
# Later models—such as Linear Regression, Decision Trees, and Random Forests—
# should perform better than this baseline.
#
#
# Overall workflow:
#
#     Load prepared housing data
#                 ↓
#     Select a target and features
#                 ↓
#     Split data into training and test sets
#                 ↓
#     Fit a DummyRegressor
#                 ↓
#     Predict house prices
#                 ↓
#     Evaluate the predictions
#
#
# Main concepts introduced:
#
# - Features: Information used to make a prediction.
# - Target: The value we want to predict.
# - Training data: Data used to fit the model.
# - Test data: Unseen data used to evaluate the model.
# - Baseline model: A simple benchmark that future models should beat.
# - MAE: Average absolute prediction error.
# - RMSE: A prediction-error measure that penalizes large errors more strongly.
# - R²: Improvement relative to predicting the mean.


# %%
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# %% [markdown]
# ## Project configuration
#
# `__file__` refers to the current Python script.
#
# Suppose this file is located here:
#
#     project/
#     ├── data/
#     └── scripts/
#         └── 02_baseline.py
#
# Then:
#
#     Path(__file__).resolve().parent
#
# points to:
#
#     project/scripts
#
# Moving up one additional level with `.parent.parent` gives us:
#
#     project
#
# This lets the script locate the processed dataset regardless of the
# current working directory in the terminal.


# %%
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project root: {PROJECT_ROOT}")
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "king_county_house_sales.parquet"
)

# Setting a random state makes the train/test split reproducible.
#
# Each time this script runs, the same observations will be assigned to
# the training and test sets.
RANDOM_STATE = 42

# Twenty percent of the observations will be reserved for testing.
TEST_SIZE = 0.20


# %% [markdown]
# ## Load the processed housing dataset
#
# This is the Parquet file created by:
#
#     01_data_setup.py
#
# The dataset has already received a small amount of model-independent cleanup:
#
# - ZIP code was treated as a categorical value.
# - Separate year, month, and day values were combined into one date.
# - `yr_renovated = 0` was converted to null.
#
# We are not doing any advanced preprocessing in this script.


# %%
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "The processed housing dataset was not found.\n"
        f"Expected location: {DATA_PATH}\n\n"
        "Run 01_data_setup.py before running this script."
    )

housing = pl.read_parquet(DATA_PATH)

print("Housing data loaded successfully.")
print(f"Dataset location: {DATA_PATH}")
print(
    f"Dataset shape: {housing.height:,} rows × "
    f"{housing.width:,} columns"
)


# %% [markdown]
# ## Inspect the dataset
#
# Before modeling, we take a quick look at:
#
# - the first few rows,
# - the column names,
# - the data types,
# - and the target distribution.
#
# This is not a complete exploratory data analysis.
#
# It is only a basic check that the expected data was loaded.


# %%
print("\nFirst five records:")
print(housing.head())

print("\nColumns:")
print(housing.columns)

print("\nSchema:")
print(housing.schema)


# %% [markdown]
# ## Identify the target
#
# The target is the value we want the model to predict.
#
# In this problem, the target is:
#
#     price
#
# We assign the target column to `y`.
#
# The letter `y` is a common machine-learning convention for the output
# or answer that the model is trying to learn.


# %%
TARGET_COLUMN = "price"

if TARGET_COLUMN not in housing.columns:
    raise ValueError(
        f"The expected target column '{TARGET_COLUMN}' is missing."
    )

y = housing[TARGET_COLUMN]

print("\nTarget column:")
print(TARGET_COLUMN)

print("\nTarget summary:")
print(y.describe())


# %% [markdown]
# ## Select a small set of features
#
# Features are the input variables that could eventually be used to predict
# the target.
#
# For this baseline model, the features do not actually affect the prediction.
#
# `DummyRegressor(strategy="mean")` ignores the feature values and predicts
# the mean training-set price for every observation.
#
# However, we still create an `X` feature table because scikit-learn estimators
# expect the standard supervised-learning structure:
#
#     X = features
#     y = target
#
# We use a small set of intuitive numeric housing features:
#
# - bedrooms
# - bathrooms
# - sqft_living
# - sqft_lot
# - floors
# - waterfront
# - view
# - condition
# - grade
# - sqft_above
# - sqft_basement
# - yr_built
# - lat
# - long
#
# We intentionally leave out several columns for now:
#
# - id: an identifier is not a meaningful measured property of the house.
# - zipcode: categorical data requires additional preprocessing.
# - date: date values require decisions about how to represent them.
# - yr_renovated: contains null values.
# - sqft_living15 and sqft_lot15: useful, but not necessary for this first step.
#
# Later scripts can revisit these choices.


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
]

missing_features = set(FEATURE_COLUMNS).difference(housing.columns)

if missing_features:
    raise ValueError(
        "The dataset is missing expected feature columns: "
        f"{sorted(missing_features)}"
    )

X = housing.select(FEATURE_COLUMNS)

print("\nSelected features:")
for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")

print(
    f"\nFeature table shape: "
    f"{X.height:,} rows × {X.width:,} columns"
)


# %% [markdown]
# ## Check for null values
#
# Many scikit-learn models cannot work directly with missing values.
#
# The selected feature columns should not contain null values for this
# baseline exercise.
#
# We calculate the number of null values in each selected feature and in
# the target.
#
# If nulls are present, we stop rather than silently changing the data.
#
# Later, when we study preprocessing, we can deliberately decide how missing
# values should be handled.


# %%
feature_null_counts = X.null_count()

print("\nNull counts in selected features:")
print(feature_null_counts)

target_null_count = y.null_count()

print(f"\nNull values in target: {target_null_count:,}")

total_feature_nulls = feature_null_counts.row(0)
total_feature_nulls = sum(total_feature_nulls)

if total_feature_nulls > 0:
    raise ValueError(
        "One or more selected feature columns contain null values.\n"
        "Choose features without null values or add an explicit "
        "missing-value strategy."
    )

if target_null_count > 0:
    raise ValueError(
        "The target column contains null values."
    )


# %% [markdown]
# ## Convert Polars objects for scikit-learn
#
# Polars is being used for loading, cleaning, and inspecting the data.
#
# Scikit-learn works most commonly with:
#
# - NumPy arrays
# - pandas DataFrames
#
# Polars can convert its data directly into NumPy arrays.
#
# After conversion:
#
#     X_array
#
# is a two-dimensional array:
#
#     rows × features
#
# and:
#
#     y_array
#
# is a one-dimensional array containing house prices.


# %%
X_array = X.to_numpy()
y_array = y.to_numpy()

print("\nNumPy array shapes:")
print(f"X shape: {X_array.shape}")
print(f"y shape: {y_array.shape}")


# %% [markdown]
# ## Split the data into training and test sets
#
# We must evaluate the model on data that was not used to fit it.
#
# Therefore, we split the complete dataset into two parts:
#
# Training set:
#
#     Used by the model during `.fit()`.
#
# Test set:
#
#     Held back during training.
#     Used later to measure performance on unseen observations.
#
# With:
#
#     test_size = 0.20
#
# approximately:
#
# - 80% of the observations go into the training set.
# - 20% of the observations go into the test set.
#
# The function returns four objects:
#
#     X_train
#     X_test
#     y_train
#     y_test


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_array,
    y_array,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nTrain/test split complete.")
print(f"Training observations: {len(y_train):,}")
print(f"Test observations:     {len(y_test):,}")


# %% [markdown]
# ## Create the baseline model
#
# `DummyRegressor` provides deliberately simple regression strategies.
#
# We use:
#
#     strategy="mean"
#
# During `.fit()`, the model calculates the mean value of `y_train`.
#
# During `.predict()`, it returns that same mean value for every observation.
#
# The model does not learn that:
#
# - larger homes may cost more,
# - waterfront homes may cost more,
# - location may influence price,
# - condition may influence price.
#
# It ignores all those relationships.
#
# That is exactly why it is useful:
#
# it gives us a very simple benchmark.


# %%
baseline_model = DummyRegressor(
    strategy="mean",
)


# %% [markdown]
# ## Fit the baseline model
#
# In scikit-learn, `.fit()` means:
#
#     learn whatever must be learned from the training data
#
# For a complex model, this might involve learning:
#
# - coefficients,
# - decision rules,
# - tree splits,
# - or many other parameters.
#
# For this dummy model, fitting only means:
#
#     calculate the mean of y_train


# %%
baseline_model.fit(
    X_train,
    y_train,
)

print("\nBaseline model fitted successfully.")


# %% [markdown]
# ## Inspect what the dummy model learned
#
# A fitted mean-based DummyRegressor stores its prediction in:
#
#     constant_
#
# This should be equal to the mean of the training target values.


# %%
training_mean_price = float(np.mean(y_train))
model_constant = float(baseline_model.constant_.ravel()[0])

print("\nMean training-set house price:")
print(f"${training_mean_price:,.2f}")

print("\nValue stored by DummyRegressor:")
print(f"${model_constant:,.2f}")


# %% [markdown]
# ## Generate predictions
#
# `.predict(X_test)` asks the fitted model to generate one predicted house
# price for every observation in the test set.
#
# Because this is a mean baseline, every prediction should be identical.


# %%
y_pred = baseline_model.predict(X_test)

print("\nPredictions created.")
print(f"Number of predictions: {len(y_pred):,}")

print("\nFirst ten predictions:")
print(y_pred[:10])


# %% [markdown]
# ## Confirm that every prediction is the same
#
# `np.unique()` returns the distinct values in an array.
#
# For this mean baseline, there should be only one unique prediction.


# %%
unique_predictions = np.unique(y_pred)

print("\nNumber of unique predicted values:")
print(len(unique_predictions))

print("\nUnique prediction:")
print(unique_predictions)


# %% [markdown]
# ## Evaluate the baseline
#
# We calculate three common regression metrics:
#
# 1. Mean Absolute Error — MAE
#
# MAE measures the average absolute distance between the actual and predicted
# house prices.
#
# Conceptually:
#
#     actual price − predicted price
#
# The positive and negative signs are removed before averaging.
#
# MAE remains in the same units as the target:
#
#     dollars
#
#
# 2. Root Mean Squared Error — RMSE
#
# RMSE also measures prediction error in dollars.
#
# However, it squares the errors before averaging them.
#
# Large errors therefore receive more weight than they do with MAE.
#
#
# 3. R²
#
# R² compares the model with a simple mean prediction.
#
# Rough interpretation:
#
#     R² = 1
#         Perfect predictions.
#
#     R² = 0
#         No improvement over predicting the test-set mean.
#
#     R² < 0
#         Worse than the simple mean benchmark.
#
# Because this model itself predicts a training-set mean, its test-set R²
# should generally be very close to zero.


# %%
mae = mean_absolute_error(
    y_test,
    y_pred,
)

mse = mean_squared_error(
    y_test,
    y_pred,
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred,
)


# %% [markdown]
# ## Display the evaluation results
#
# These scores become the benchmark for later models.
#
# When we build Linear Regression, Decision Trees, and Random Forests,
# we can compare their test scores against these baseline results.


# %%
print("\nBaseline model results")
print("-" * 60)

print(f"Prediction strategy: Mean training-set price")
print(f"Constant prediction: ${model_constant:,.2f}")

print()
print(f"Mean Absolute Error:      ${mae:,.2f}")
print(f"Root Mean Squared Error:  ${rmse:,.2f}")
print(f"R² score:                  {r2:.4f}")


# %% [markdown]
# ## Compare a few actual and predicted values
#
# Looking at individual examples helps make the baseline concrete.
#
# The actual prices vary considerably.
#
# The baseline prediction remains unchanged for every house.


# %%
comparison = pl.DataFrame(
    {
        "actual_price": y_test[:10],
        "predicted_price": y_pred[:10],
        "absolute_error": np.abs(
            y_test[:10] - y_pred[:10]
        ),
    }
)

print("\nFirst ten actual-versus-predicted values:")
print(comparison)


# %% [markdown]
# ## Interpretation
#
# This baseline model is intentionally unsophisticated.
#
# It predicts:
#
#     the average training-set house price
#
# for every house.
#
# It gives us a minimum standard for future regression models.
#
# A useful model should normally:
#
# - produce a lower MAE,
# - produce a lower RMSE,
# - and produce a higher R².
#
# The baseline does not tell us whether Linear Regression or a Decision Tree
# will be good.
#
# It only establishes the performance that they must improve upon.


# %%
print("\nInterpretation")
print("-" * 60)

print(
    "The dummy model predicts the same training-set mean price "
    "for every test observation."
)

print(
    "Future regression models should reduce MAE and RMSE "
    "and increase R² relative to this baseline."
)


# %% [markdown]
# ## Key ideas to remember
#
# 1. A baseline is a benchmark, not a serious final model.
#
# 2. The test set must remain separate from the data used to fit the model.
#
# 3. A mean-based DummyRegressor predicts the training target mean for every
#    observation.
#
# 4. MAE describes the average absolute prediction error.
#
# 5. RMSE gives greater weight to large errors.
#
# 6. R² describes improvement relative to a mean prediction.
#
# 7. More complicated models are only useful if they perform better than a
#    meaningful baseline.