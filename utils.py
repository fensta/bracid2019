import math

import numpy as np
import pandas as pd


MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
TYPE_MAPPING = {""}
DATA_TYPES = []


def read_dataset(src, excluded=[], skip_rows=0, na_values=[]):
    """
    Reads in a dataset in csv format and stores it in a dataFrame.

    Parameters
    ----------
    src: string - path to input dataset
    excluded: list of int - 0-based indices of columns/features to be ignored
    skip_rows: int - number of rows to skip at the beginning of <src>
    na_values: list of str - values encoding missing values - these are represented by NaN

    Return
    ------
    pd.DataFrame - dataset

    """
    df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values)
    # TODO: normalize numeric features to the range (0-1)
    return df


def hvdm(xi, yi):
    """
    Computes the distance (Heterogenous Value Difference Metrics) between a rule/example and another example.

    Parameters
    ----------
    xi: pd.dataFrame - (m x n or a x b), where a<=m, b<=n rule or example
    yi: pd.dataFrame - (m x n) example

    Returns
    -------
    float - distance.

    """
    # Select same columns in both inputs
    # https://stackoverflow.com/questions/46228574/pandas-select-dataframe-columns-based-on-another-dataframes-columns
    long = yi[yi.columns.intersection(xi.columns)]
    short = xi[xi.columns.intersection(yi.columns)]
    print(xi.shape + ": " + xi.columns, yi.shape + ": " + yi.columns)
    dists = []
    # Compute distance for j-th feature
    for j, _ in enumerate(short):
        # Extract column from both dataframes into numpy array
        short_col = short.iloc[:, j].values
        long_col = long.iloc[:, j].values
        # Compute nominal/numeric distance
        if short.dtype[j] == "unicode" or short.dtype[j] == "object":
            dist = svdm()
        else:
            dist = di()
        dists.append(dist*dist)
    # Sum up distances
    dist = math.sqrt(sum(dists))


def svdm():
    """
    Computes the Value difference metric for nominal values.

    Parameters
    ----------

    Returns
    -------
    float - distance

    """
    return 0


def di():
    """
    Computes the Euclidean distance for numeric values.

    Parameters
    ----------

    Returns
    -------
    float - distance

    """
    return 0