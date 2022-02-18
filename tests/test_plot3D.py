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
        widget='section'
    )
    # assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)

# test data with Pandas dtypes
testdata4 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
    'pitname': ['pit1', 'pit2', 'pit3',],
    'messy_field':[1, 1.0, '1']
}


def test_plot3D_4():
    # all params specified
    data = pd.DataFrame(testdata4)
    data = data.convert_dtypes()  # Pandas dtypes aren't supported by Plot3D
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions

    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=True,
        widget='section'
    )
    # assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)

# method 1
"""
# pandas dtype error handling issue #67
data = pd.DataFrame(testdata3)  # df
data = data.convert_dtypes()  # turning into pandas dtypes (handle na) -> this won't plot3d

data_np = data.to_numpy(dtype=np.float64)  # no column headers
columns_to_take_over = list(data.columns)

data_pd = pd.DataFrame(data_np, columns=columns_to_take_over )

plot = data_pd.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(5, 5, 5),
    col='ton',
    show_plot=True,
    widget=None
)

"""

# method 2
data = pd.DataFrame(testdata3)  # df
data = data.convert_dtypes()  # turning into pandas dtypes (handle na) -> this won't plot3d

selection_numbers = data.select_dtypes(include=[np.number])
selection_strings = data.select_dtypes(exclude=[np.number])

num_cols = list(selection_numbers.columns)
str_cols = list(selection_strings.columns)

# data_num_np = selection_numbers.to_numpy(dtype=float)
# data_str_np = selection_strings.to_numpy(dtype=str)

data_numbers_pd = pd.DataFrame(selection_numbers, columns=num_cols, dtype=np.float64)
data_strings_pd = pd.DataFrame(selection_strings, columns=str_cols)

data_pd = pd.concat([data_numbers_pd, data_strings_pd], axis=1)
print(data_pd.dtypes)

plot = data_pd.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(5, 5, 5),
    col='ton',
    show_plot=True,
    widget=None
)

# simplifying method 2

testdata3 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
    'string': ['25', '100', '50'],
    'obj': ['25', 100, '50'],
}
data = pd.DataFrame(testdata3)  # df
data = data.convert_dtypes()  # turning into pandas dtypes (handle na) -> this won't plot3d

selection_numbers = data.select_dtypes(include=[np.number])
selection_strings = data.select_dtypes(exclude=[np.number])

data_numbers_pd = pd.DataFrame(selection_numbers,  dtype=float)
data_strings_pd = pd.DataFrame(selection_strings, dtype=str)

data_pd = pd.concat([data_numbers_pd, data_strings_pd], axis=1)
print(data_pd.dtypes)

data_pd.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(5, 5, 5),
    col='obj',
    widget=None
)