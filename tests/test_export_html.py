# standard automated tests for miningpy->visualisation->plot3D function
import pandas as pd
import os
import pytest
import miningpy

# test data
testdata = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
}


def test_export_html_1(tmpdir):
    file = tmpdir.join("export_test.html")
    data = pd.DataFrame(testdata)
    plot = data.export_html(
        path=file,
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
    )

    assert os.path.exists(file)  # check file was created
