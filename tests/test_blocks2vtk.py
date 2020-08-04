# standard automated tests for miningpy->visualisation->blocks2vtk function
import pandas as pd
import pytest
import miningpy

# test data
testdata = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
}


def test_blocks2vtk_1(tmpdir):
    # test blocks2vtk
    # all params specified
    file = tmpdir.mkdir("data").join("test.vtu")
    data = pd.DataFrame(testdata)
    output = data.blocks2vtk(
        path=str(file),
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        rotation=(0, 0, 0)
    )
    assert output
