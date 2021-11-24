# standard automated tests for miningpy->visualisation->plot3D function
import pandas as pd
import pytest
import miningpy
import pyvista as pv

# test data
testdata1 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
}


def test_plot3D_1():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata1)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=False
    )

    assert isinstance(plot, pv.Plotter)


# test data - only one block
testdata2 = {
    'x': [5],
    'y': [5],
    'z': [5],
    'ton': [50],
}


def test_plot3D_2():
    # test blocks2vtk
    data = pd.DataFrame(testdata2)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=False
    )

    assert isinstance(plot, pv.Plotter)

# test data with slider widget
testdata3 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
}

def test_plot3D_3():
    # all params specified
    data = pd.DataFrame(testdata3)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=True,
        widget='section'
    )

    # assert isinstance(plot, pv.Plotter)
test_plot3D_3()
print('hello')
