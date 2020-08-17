"""
This module is for exporting Whittle format .mod and .par files
for a block model to do pit optimisation
"""

# import libraries
import pandas as pd
from pandas.core.base import PandasObject
from typing import Union, List, Tuple


def whittle_mod(blockmodel: pd.DataFrame):
    """
    create a Whittle 4D .MOD file from a block model

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    Whittle .MOD text file
    """
    raise Exception("MiningPy function {whittle_mod} hasn't been created yet")


def whittle_par():
    raise Exception("MiningPy function {whittle_par} hasn't been created yet")


def extend_pandas():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.whittle_mod = whittle_mod
