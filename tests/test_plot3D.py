# standard automated tests for miningpy->visualisation->plot3D function
import pandas as pd
import pytest
import miningpy
import pyvista as pv

# test data
testdata = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
}


def test_plot3D_1():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=False
    )

    assert isinstance(plot, pv.Plotter)
