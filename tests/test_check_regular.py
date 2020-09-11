# standard automated tests for miningpy->core->block_dims function
import pandas as pd
import pytest
import miningpy

# test data 1
testdata1 = {
    'x': [],
    'y': [],
    'z': [],
}

xblocks1 = [5, 10, 15, 20, 25, 30]
yblocks1 = [5, 15, 25, 35, 45]
zblocks1 = [5, 10, 15, 20]

for x in xblocks1:
    for y in yblocks1:
        for z in zblocks1:
            testdata1['x'].append(x)
            testdata1['y'].append(y)
            testdata1['z'].append(z)


def test_check_regular_1():
    data = pd.DataFrame(testdata1)
    check = data.check_regular(
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 0.0, 2.5),
        dims=(5, 10, 5)
    )
    assert check is True


# test data 2
testdata2 = {
    'x': [],
    'y': [],
    'z': [],
}

xblocks2 = [5, 10, 15, 20, 25, 30]
yblocks2 = [5, 15, 24, 35, 45]  # not regular
zblocks2 = [5, 10, 15, 20]

for x in xblocks2:
    for y in yblocks2:
        for z in zblocks2:
            testdata2['x'].append(x)
            testdata2['y'].append(y)
            testdata2['z'].append(z)


def test_check_regular_2():
    data = pd.DataFrame(testdata2)
    check = data.check_regular(
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 0.0, 2.5),
        dims=(5, 10, 5)
    )
    assert check is False


