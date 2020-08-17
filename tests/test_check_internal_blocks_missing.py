# standard automated tests for miningpy->core->check_internal_blocks_missing function
import pandas as pd
import pytest
import miningpy

# test data1
testdata1 = {
    'x': [],
    'y': [],
    'z': [],
}
for x in [5, 10, 15, 20]:
    for y in [5, 10, 15, 20]:
        for z in [5, 15, 20]:
            testdata1['x'].append(x)
            testdata1['y'].append(y)
            testdata1['z'].append(z)


def test_check_internal_blocks_missing_1():
    data = pd.DataFrame(testdata1)
    check = data.check_internal_blocks_missing(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5)
    )
    assert check  # check should be true because there are missing blocks


# test data2
testdata2 = {
    'x': [],
    'y': [],
    'z': [],
}
for x in [5, 10, 15, 20]:
    for y in [5, 10, 15, 20]:
        for z in [5, 10, 15, 20]:
            testdata2['x'].append(x)
            testdata2['y'].append(y)
            testdata2['z'].append(z)


def test_check_internal_blocks_missing_2():
    data = pd.DataFrame(testdata2)
    check = data.check_internal_blocks_missing(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5)
    )
    assert check is False  # check should be false because there are NOT missing blocks
