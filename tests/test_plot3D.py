# standard automated tests for miningpy->visualisation->plot3D function
import pandas as pd
import pytest
import miningpy
import pyvista as pv
import pyvistaqt
from pyvistaqt import BackgroundPlotter
import numpy as np

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

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


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

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


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
        show_plot=False,
        widget='section',
        window_size=(3000, 2000),
    )
    # assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)

# test data with Pandas dtypes
testdata4 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
    'pitname': ['pit1', 'pit2', 'pit3'],
    'messy_field': [1, 1.0, '1']
}

def test_plot3D_4():
    # all params specified
    data = pd.DataFrame(testdata4)
    data_conv = data.convert_dtypes()  # Pandas dtypes aren't supported by Plot3D
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions

    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='messy_field',
        show_plot=True,
        widget=None
    )
    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)
