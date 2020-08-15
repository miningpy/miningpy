import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
import ezdxf
import pyvista as pv



def face_position_dxf():
    return


def extend_pandas_dxf():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.face_position_dxf = face_position_dxf
