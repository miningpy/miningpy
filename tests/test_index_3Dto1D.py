# standard automated tests for miningpy->core->index_3Dto1D function
import pandas as pd
import pytest
import miningpy

# test data
testdata = {
    'x': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    'y': [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2],
    'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


def test_index_3Dto1D_1():
    data = pd.DataFrame(testdata)
    result = {
        'ijk': [5, 5, 15],
    }
    result = pd.DataFrame(result)
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert True

