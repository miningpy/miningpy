# standard automated tests for miningpy->core->xyz function
import pandas as pd
import pytest
import miningpy

# test data
testdata = {
    'i': [0, 0, 2],
    'j': [0, 2, 4],
    'k': [0, 0, 0],
    'xdim': [5, 5, 5],
    'ydim': [5, 5, 5],
    'zdim': [5, 5, 5],
}


def test_xyz_1():
    # test xyz inplace
    data = pd.DataFrame(testdata)
    result = {
        'x': [5, 5, 15],
        'y': [5, 15, 25],
        'z': [5, 5, 5],
    }
    result = pd.DataFrame(result)
    data.xyz(
        origin=(2.5, 2.5, 2.5),
        dims=(5, 5, 5),
        rotation=(0, 0, 0),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes same
    result = result.astype(float)  # make sure dtypes same
    assert data[['x', 'y', 'z']].equals(result)

