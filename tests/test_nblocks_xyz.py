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

nblocks_x1 = 6
nblocks_y1 = 5
nblocks_z1 = 4

for x in xblocks:
    for y in yblocks:
        for z in zblocks:
            testdata1['x'].append(x)
            testdata1['y'].append(y)
            testdata1['z'].append(z)


def test_nblocks_xyz_1():
    data = pd.DataFrame(testdata1)
    nx, ny, nz = data.nblocks_xyz(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 10, 5),
        origin=(2.5, 2.5, 2.5),
        rotation=(0, 0, 0),
    )
    assert (nx == nblocks_x1) & (ny == nblocks_y1) & (nz == nblocks_z1)

