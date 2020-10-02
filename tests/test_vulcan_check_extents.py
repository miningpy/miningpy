# standard automated tests for miningpy->core->vulcan_bdf function
import pandas as pd
import pytest
import miningpy

# test data 1
testdata1 = {
    'centroid_x': ['Variable descriptions:', 'Variable types:', 'Variable defaults:', 5, 8, 9, 10],
    'centroid_y': ['', '', '', 5, 8, 6, 10],
    'centroid_z': ['', '', '', 0, 0, 0, 10],
    'dim_x': ['', '', '', 1, 1, 1, 1],
    'dim_y': ['', '', '', 1, 1, 1, 1],
    'dim_z': ['', '', '', 1, 1, 1, 1],
    'volume': ['', '', '', 125, 125, 125, 125],
    'tonnage': ['', 'double', -99.0, 50, 100, 50, 100],
    'cu': ['', 'double', -99.0, 5.0, 10.0, 25.0, 50.0],
    'rocktype': ['', 'double', -99.0, 'ox', 'ox', 'sulph', 'sulph']
}


def test_vulcan_check_extents():
    # test blocks2vtk
    # all params specified
    data = pd.read_csv('C:\\Projects\\01237_IAMGOLD\\02_Vulcan\\JZone\\L20_KH20_BM_mini.csv')
    output = data.check_regular_extents(
        end_offset=(5000, 1914, 666),
        dims=(8, 6, 9),
        origin=(46181, 88486, 4),
        original_rotation=(0, 0, 90 - 108),
        xyz_cols=('centroid_x', 'centroid_y', 'centroid_z'),
    )

    assert output

if __name__ == "__main__":
    test_vulcan_check_extents()

    


