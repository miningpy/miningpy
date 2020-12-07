# standard automated tests for miningpy->core->block_dims function
import pandas as pd
import pytest
import miningpy

# test data
testdata1 = {
    'x': [],
    'y': [],
    'z': [],
}

xblocks = [5, 10, 15, 20, 25, 30]
yblocks = [5, 15, 25, 35, 45]
zblocks = [5, 10, 15, 20]

for x in xblocks:
    for y in yblocks:
        for z in zblocks:
            testdata1['x'].append(x)
            testdata1['y'].append(y)
            testdata1['z'].append(z)


def test_model_origin_1():
    data = pd.DataFrame(testdata1)
    xorigin, yorigin, zorigin = data.model_origin(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 10, 5)
    )
    assert (xorigin == 2.5) & (yorigin == 0.0) & (zorigin == 2.5)


# test with dimensions as columns in the block model
def test_model_origin_2():
    data = pd.DataFrame(testdata1)
    data['xdim'] = 5
    data['ydim'] = 10
    data['zdim'] = 5
    xorigin, yorigin, zorigin = data.model_origin(
        xyz_cols=('x', 'y', 'z'),
        dims=('xdim', 'ydim', 'zdim')
    )
    assert (xorigin == 2.5) & (yorigin == 0.0) & (zorigin == 2.5)

