import logging
import re
from pathlib import Path

import pandas as pd


def load_data(path):
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading data from %s", path)
    df = (
        pd.read_csv(path, parse_dates=["DateTime"])
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=convert_camel_case)
        .fillna("Unknown")
    )
    logger.info("Read %i rows", len(df))
    return df


def convert_camel_case(name:str)->str:
    """Convert camelCaseString to snake_case_string."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
