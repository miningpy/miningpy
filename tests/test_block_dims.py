# standard automated tests for miningpy->core->block_dims function
import pandas as pd
import pytest
import miningpy

# test data
testdata = {
    'x': [],
    'y': [],
    'z': [],
}
for x in [5, 10, 15, 20]:
    for y in [5, 15, 25, 35]:
        for z in [5, 10, 15, 20]:
            testdata['x'].append(x)
            testdata['y'].append(y)
            testdata['z'].append(z)


def test_block_dims_1():
    data = pd.DataFrame(testdata)
    dims = data.block_dims(
        xyz_cols=('x', 'y', 'z'),
        origin=(2.5, 2.5, 2.5),
        rotation=(0, 0, 0),
    )
    dims = tuple([float(dim) for dim in dims])
    assert dims == (5.0, 10.0, 5.0)

