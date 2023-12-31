# standard automated tests for miningpy->visualisation->plot3D function
import pandas as pd
import pytest
import miningpy
import pyvistaqt
import numpy as np
import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# test data
testdata1 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_1():
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_2():
    # test blocks2vtk
    data = pd.DataFrame(testdata2)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='ton',
        show_plot=False
    )

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


# test data with widgets
testdata3 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
}

widget_list = ['section', 'slider']


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_3():
    # all params specified
    data = pd.DataFrame(testdata3)
    for widget in widget_list:
        plot = data.plot3D(
            xyz_cols=('x', 'y', 'z'),
            dims=(5, 5, 5),
            col='ton',
            show_plot=False,
            widget=widget,
            window_size=None,
        )
        assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter), f'error with plot3D {widget} widget'


# test data with Pandas dtypes
testdata4 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50.1, 100, 50],
    'pitname': ['pit1', 'pit2', 'pit3'],
    'messy_field': [1, 1.0, '1'],
    'bool': [True, False, True],
    'nan': [1, 2, np.nan]
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_4():
    list_of_fields = ['pitname', 'messy_field', 'bool', 'ton', 'nan']  # these will all be converted to pandas dtypes
    data = pd.DataFrame(testdata4)
    data = data.convert_dtypes()  # Pandas dtypes aren't supported by Plot3D
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions

    for col in list_of_fields:
        if col == 'nan':  # this should fail with dtype pd.NA
            with pytest.raises(Exception):
                data.plot3D(
                    xyz_cols=('x', 'y', 'z'),
                    dims=(5, 5, 5),
                    col=col,
                    show_plot=False
                )
        else:
            plot = data.plot3D(
                xyz_cols=('x', 'y', 'z'),
                dims=(5, 5, 5),
                col=col,
                show_plot=False
            )
            assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter), 'pandas dtype error'


# test data with standard dtypes
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_5():
    list_of_fields = ['pitname', 'messy_field', 'bool', 'ton', 'nan']  # these will NOT be converted to pandas dtypes
    data = pd.DataFrame(testdata4)

    for col in list_of_fields:
        plot = data.plot3D(
            xyz_cols=('x', 'y', 'z'),
            dims=(5, 5, 5),
            col=col,
            show_plot=False
        )
        assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


# test data
testdata6 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'pit': ['pit1', 'pit2', 'pit3'],
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_6():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata6)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='pit',
        show_plot=False
    )

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


# test data
testdata7 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'pit': ['pit1', 'pit2', 'pit3'],
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_7():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata7)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='pit',
        widget='slider',
        show_plot=False
    )

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


# test data
testdata8 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'pit': ['pit1', 'pit2', 'pit3'],
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_8():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata8)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='pit',
        widget='section',
        show_plot=False
    )

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


# test data
testdata9 = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'mask': [True, False, True],
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_plot3d_9():
    # test blocks2vtk
    # all params specified
    data = pd.DataFrame(testdata9)
    plot = data.plot3D(
        xyz_cols=('x', 'y', 'z'),
        dims=(5, 5, 5),
        col='mask',
        show_plot=False
    )

    assert isinstance(plot, pyvistaqt.plotting.BackgroundPlotter)


