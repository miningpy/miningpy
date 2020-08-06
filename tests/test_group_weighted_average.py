# standard automated tests for miningpy->core->group_weighted_average function
import pandas as pd
import pytest
import miningpy

# test data 1
testdata1 = {
    'x': [5, 8, 9, 10],
    'y': [5, 8, 6, 10],
    'z': [0, 0, 0, 10],
    'tonnage': [50, 100, 50, 100],
    'cu': [5.0, 10.0, 25.0, 50.0],
    'rocktype': ['ox', 'ox', 'sulph', 'sulph']
}


def test_group_weighted_average_1():
    # test rotate_grid
    # 45 degree rotation around the z-axis (i.e. rotation of the xy plane)
    data = pd.DataFrame(testdata1)
    result = {
        'cu': {'ox': 8.33333333, 'sulph': 41.66666667},
    }
    result = pd.DataFrame(result)
    data_group = data.group_weighted_average(
        avg_cols='cu',
        weight_col='tonnage',
        group_cols='rocktype'
    )
    data_group = data_group.astype(float)  # make sure dtypes are float
    result = result.astype(float)  # make sure dtypes are float

    # compare float values
    test = abs(data_group - result)
    check = (test < 0.0001).all(axis=None)

    assert check


# test data 2
testdata2 = {
    'x': [5, 8, 9, 10],
    'y': [5, 8, 6, 10],
    'z': [0, 0, 0, 10],
    'tonnage': [50, 100, 50, 100],
    'cu': [5.0, 10.0, 25.0, 50.0],
    'au': [1.0, 2.0, 3.0, 4.0],
    'rocktype': ['ox', 'ox', 'sulph', 'sulph']
}


def test_group_weighted_average_2():
    # test rotate_grid
    # 45 degree rotation around the z-axis (i.e. rotation of the xy plane)
    data = pd.DataFrame(testdata2)
    result = {
        'cu': {'ox': 8.33333333, 'sulph': 41.66666667},
        'au': {'ox': 1.66666667, 'sulph': 3.66666667}
    }
    result = pd.DataFrame(result)
    data_group = data.group_weighted_average(
        avg_cols=['cu', 'au'],
        weight_col='tonnage',
        group_cols='rocktype'
    )
    data_group = data_group.astype(float)  # make sure dtypes are float
    result = result.astype(float)  # make sure dtypes are float

    # compare float values
    test = abs(data_group - result)
    check = (test < 0.0001).all(axis=None)

    assert check


if __name__ == '__main':
    test_group_weighted_average_1()
    test_group_weighted_average_2()
