# standard automated tests for miningpy->core->rotate_grid function
import pandas as pd
import pytest
import miningpy

# test data - 3 blocks
testdata = {
    'x': [5, 8, 9],
    'y': [5, 8, 6],
    'z': [0, 0, 0],
}


def test_rotate_grid_1():
    # test rotate_grid
    # 45 degree rotation around the z-axis (i.e. rotation of the xy plane)
    data = pd.DataFrame(testdata)
    result = {
        'x': [2.5,          2.5,           4.6213203436],
        'y': [6.0355339059, 10.2781745931, 9.5710678119],
        'z': [0,            0,             0]
    }
    result = pd.DataFrame(result)
    data.rotate_grid(
        origin=(2.5, 2.5, 2.5),
        rotation=(0, 0, 45),
        inplace=True
    )
    data = data.astype(float)  # make sure dtypes are float
    result = result.astype(float)  # make sure dtypes are float

    # compare float values
    test = abs(data - result)
    check = (test < 0.0001).all(axis=None)

    assert check


if __name__ == '__main':
    test_rotate_grid_1()

