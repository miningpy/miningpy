# standard automated tests for miningpy->core->vulcan_csv function
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
testresult1 = {
    'centroid_x': ['Variable descriptions:', 'Variable types:', 'Variable defaults:', 5, 8, 9, 10],
    'centroid_y': ['', '', '', 5, 8, 6, 10],
    'centroid_z': ['', '', '', 0, 0, 0, 10],
    'dim_x': ['', '', '', 5, 5, 5, 5],
    'dim_y': ['', '', '', 5, 5, 5, 5],
    'dim_z': ['', '', '', 5, 5, 5, 5],
    'volume': ['', '', '', 125, 125, 125, 125],
    'tonnage': ['', 'double', -99.0, 50, 100, 50, 100],
    'cu': ['', 'double', -99.0, 5.0, 10.0, 25.0, 50.0],
    'rocktype': ['', 'double', -99.0, 'ox', 'ox', 'sulph', 'sulph']
}


def test_vulcan_csv_1(tmpdir):
    # test blocks2vtk
    # all params specified
    file = tmpdir.mkdir("data").join("vulcan.csv")
    data = pd.DataFrame(testdata1)
    result = pd.DataFrame(testresult1)
    output = data.vulcan_csv(
        path=str(file),
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
    )
    assert output.equals(result)

