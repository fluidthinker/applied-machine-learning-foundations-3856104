# %% [markdown]
# # 01 — Data Setup
#
# Download and prepare the King County House Sales dataset used in the
# LinkedIn Learning Machine Learning Foundations course.
#
# Pipeline:
#
#     OpenML
#        ↓
#     Raw ARFF file
#        ↓
#     Initial cleanup
#        ↓
#     Processed Parquet file
#
# Design philosophy:
#
# - Preserve the original downloaded data.
# - Keep data preparation reproducible.
# - Perform only model-independent transformations here.
# - Leave imputation, scaling, encoding, and feature engineering for
#   later machine-learning pipelines.


# %%
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import polars as pl


# %% [markdown]
# ## Dataset configuration


# %%
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATA_URL = "https://www.openml.org/data/download/22044765/dataset"

RAW_DATA_PATH = RAW_DATA_DIR / "king_county_house_sales.arff"
PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "king_county_house_sales.parquet"
)

COLUMN_NAMES = [
    "id",
    "price",
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
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "date_year",
    "date_month",
    "date_day",
]

ARFF_METADATA_ROWS = 31


# %% [markdown]
# ## Create data directories


# %%
def create_data_directories() -> None:
    """Create the raw and processed data directories."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Download the raw dataset
#
# The source is an ARFF file:
#
#     Attribute-Relation File Format
#
# ARFF files contain:
#
#     Metadata describing the columns
#                 +
#     Comma-separated data rows
#
# Polars can read the data portion after we skip the ARFF metadata.


# %%
def download_raw_data(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the raw dataset when it is not already available.

    Args:
        url: Web address of the source dataset.
        destination: Local path where the raw file will be stored.
        overwrite: Whether to download the file again if it already exists.

    Returns:
        The path to the downloaded raw dataset.

    Raises:
        RuntimeError: If the dataset cannot be downloaded.
    """
    if destination.exists() and not overwrite:
        print(f"Using existing raw dataset: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from:\n{url}")

    try:
        urlretrieve(url, destination)
    except (HTTPError, URLError, OSError) as error:
        raise RuntimeError(
            "The King County dataset could not be downloaded from OpenML."
        ) from error

    print(f"Raw dataset saved to: {destination}")

    return destination


# %% [markdown]
# ## Load the raw ARFF data


# %%
def load_raw_data(path: Path) -> pl.DataFrame:
    """Load the data section of the King County ARFF file.

    Args:
        path: Path to the downloaded ARFF file.

    Returns:
        A Polars DataFrame containing the raw housing records.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the loaded dataset is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    dataframe = pl.read_csv(
        path,
        has_header=False,
        skip_rows=ARFF_METADATA_ROWS,
        new_columns=COLUMN_NAMES,
        infer_schema_length=10_000,
    )

    if dataframe.is_empty():
        raise ValueError("The raw dataset contains no records.")

    return dataframe


# %% [markdown]
# ## Initial cleanup
#
# This is based directly on the course notebook's `tweak_housing()`
# function.
#
# Transformations:
#
# 1. Treat ZIP code as a categorical value rather than a measured number.
# 2. Combine date_year, date_month, and date_day into one date column.
# 3. Replace yr_renovated = 0 with null because zero means that the house
#    was never renovated.
# 4. Remove the three separate date component columns.


# %%
def clean_housing_data(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Apply the course's initial housing-data transformations.

    Args:
        dataframe: Raw King County housing data.

    Returns:
        Cleaned housing data ready for exploratory analysis and later
        machine-learning preprocessing.
    """
    return (
        dataframe
        .with_columns(
            pl.col("zipcode")
            .cast(pl.String)
            .cast(pl.Categorical),
            pl.date(
                pl.col("date_year"),
                pl.col("date_month"),
                pl.col("date_day"),
            ).alias("date"),
            pl.col("yr_renovated")
            .replace(0, None)
            .alias("yr_renovated"),
        )
        .select(
            [
                "id",
                "price",
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
                "yr_renovated",
                "zipcode",
                "lat",
                "long",
                "sqft_living15",
                "sqft_lot15",
                "date",
            ]
        )
    )


# %% [markdown]
# ## Validate the prepared dataset


# %%
def validate_data(dataframe: pl.DataFrame) -> None:
    """Validate the basic structure of the prepared dataset.

    Args:
        dataframe: Prepared housing dataset.

    Raises:
        ValueError: If an expected structural condition is not met.
    """
    required_columns = {
        "id",
        "price",
        "zipcode",
        "date",
        "lat",
        "long",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The prepared dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.is_empty():
        raise ValueError("The prepared dataset contains no records.")

    if dataframe["price"].null_count() == dataframe.height:
        raise ValueError("The target column 'price' contains only null values.")

    if dataframe["date"].null_count() == dataframe.height:
        raise ValueError("The date transformation produced only null values.")


# %% [markdown]
# ## Save the prepared dataset


# %%
def save_processed_data(
    dataframe: pl.DataFrame,
    destination: Path,
) -> None:
    """Save the prepared housing data as a compressed Parquet file.

    Args:
        dataframe: Prepared housing dataset.
        destination: Output path for the Parquet file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataframe.write_parquet(
        destination,
        compression="zstd",
    )


# %% [markdown]
# ## Display a pipeline summary


# %%
def print_data_summary(
    raw_dataframe: pl.DataFrame,
    cleaned_dataframe: pl.DataFrame,
) -> None:
    """Print a concise summary of the data-setup pipeline.

    Args:
        raw_dataframe: Dataset before cleanup.
        cleaned_dataframe: Dataset after cleanup.
    """
    print("\nData setup complete")
    print("-" * 60)

    print(f"Source URL:     {DATA_URL}")
    print(f"Raw file:       {RAW_DATA_PATH}")
    print(f"Processed file: {PROCESSED_DATA_PATH}")

    print()
    print(
        "Raw shape:       "
        f"{raw_dataframe.height:,} rows × "
        f"{raw_dataframe.width:,} columns"
    )
    print(
        "Processed shape: "
        f"{cleaned_dataframe.height:,} rows × "
        f"{cleaned_dataframe.width:,} columns"
    )

    print()
    print("Processed schema:")
    print(cleaned_dataframe.schema)

    print()
    print("First five processed records:")
    print(cleaned_dataframe.head())


# %% [markdown]
# ## Run the data-setup pipeline


# %%
def main() -> None:
    """Run the complete data-download and preparation pipeline."""
    create_data_directories()

    raw_path = download_raw_data(
        url=DATA_URL,
        destination=RAW_DATA_PATH,
    )

    raw = load_raw_data(raw_path)
    cleaned = clean_housing_data(raw)

    validate_data(cleaned)
    save_processed_data(cleaned, PROCESSED_DATA_PATH)

    print_data_summary(
        raw_dataframe=raw,
        cleaned_dataframe=cleaned,
    )


# %%
if __name__ == "__main__":
    main()