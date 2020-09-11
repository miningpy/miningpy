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
    data = data.index_3Dto1D(
        xyz_cols=('x', 'y', 'z'),
        origin=(-0.5, -0.5, -0.5),
        dims=(1, 1, 1)
    )
    result = {
        'ijk': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    }
    result = pd.DataFrame(result)
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['ijk']].equals(result)

