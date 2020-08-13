# import libraries
import pyvista as pv
import pandas as pd
from pandas.core.base import PandasObject
import numpy as np
from typing import Union, Tuple


def plot3D(blockmodel:  pd.DataFrame,
           xyz_cols:    Tuple[str, str, str] = ('x', 'y', 'z'),
           col:         str = None,
           dims:        Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
           rotation:    Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
           widget:      str = None,
           show_grid:   bool = True,
           show_plot:   bool = True) -> pv.Plotter:
    """
    create activate 3D vtk plot of block model that is fully interactive

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    col: str
        attribute column to plot (i.e. tonnage, grade, etc)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    widget: {"COG","section", "both"}
        add widgets such as slider (cut off grade) or cross-section.
        to add both a COG and cross-section widget use "both"
    show_grid: bool
        add x,y,z grid to see coordinates on plot
    show_plot: bool
        whether to open active window or just return pyvista.Plotter object
        to .show() later

    Returns
    -------
    pyvista.Plotter object & active window of block model 3D plot
    """

    # check col data to plot is int or float data - not string or bool
    if blockmodel[col].dtype != 'int64' or blockmodel[col].dtype != 'float64':
        raise Exception(f'MiningPy ERROR - column to plot: {col} must be Pandas int64 or float64')

    # check for duplicate blocks and return warning
    dup_check = list(blockmodel.duplicated(subset=[xyz_cols[0], xyz_cols[1], xyz_cols[2]]).unique())
    assert True not in dup_check, 'MiningPy ERROR - duplicate blocks in dataframe'

    # definitions for simplicity
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # make shallow copy of required columns
    xyz_cols = list(xyz_cols)
    cols = list(xyz_cols)
    cols.append(col)
    block_model = blockmodel[cols]

    # Create the spatial reference
    grid = pv.UniformGrid()

    x_orig = block_model[xyz_cols[0]].min() - (0.5 * dims[0])
    y_orig = block_model[xyz_cols[1]].min() - (0.5 * dims[1])
    z_orig = block_model[xyz_cols[2]].min() - (0.5 * dims[2])
    origin = (x_orig, y_orig, z_orig)

    # Edit the spatial reference
    grid.origin = origin  # block model origin -  bottom left corner of the model
    grid.spacing = dims  # x,y,z dimensions of blocks

    # build the block model framework
    nx = int((block_model[xyz_cols[0]].max() - block_model[xyz_cols[0]].min()) / grid.spacing[0]) + 1  # number of blocks in x dimension
    ny = int((block_model[xyz_cols[1]].max() - block_model[xyz_cols[1]].min()) / grid.spacing[1]) + 1  # number of blocks in y dimension
    nz = int((block_model[xyz_cols[2]].max() - block_model[xyz_cols[2]].min()) / grid.spacing[2]) + 1  # number of blocks in z dimension
    grid.dimensions = [nx+1, ny+1, nz+1]  # need extra dimension - 4th dimension is for data (x,y,z,data)

    if x_rotation > 0:
        grid.rotate_x(x_rotation)
    if y_rotation > 0:
        grid.rotate_y(y_rotation)
    if z_rotation > 0:
        grid.rotate_z(z_rotation)

    # only keep blocks actually in block model dataframe
    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = xyz_cols
    cell_coords['idx'] = cell_coords.index
    mask = pd.merge(cell_coords, block_model, on=xyz_cols, how='inner')

    grid = grid.extract_cells(mask['idx'].values)

    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = xyz_cols
    cell_coords['idx'] = cell_coords.index

    cell_coords.set_index(xyz_cols, inplace=True)
    block_model.set_index(xyz_cols, inplace=True)
    block_model['idx'] = cell_coords['idx']
    block_model.sort_values(by=['idx'], inplace=True)

    grid.cell_arrays[col] = block_model[col].values  # add the data values to visualise

    # set theme
    pv.set_plot_theme("ParaView")  # just changes colour scheme

    p = pv.Plotter(notebook=False)

    if widget is None:
        p.add_mesh(grid, show_edges=True, scalars=col)
    if widget == "section":
        p.add_mesh_clip_plane(mesh=grid, show_edges=True, scalars=col)
    if widget == "COG":
        p.add_mesh_threshold(mesh=grid, show_edges=True, scalars=col)
    if widget == "both":
        p.add_mesh_threshold(mesh=grid, show_edges=True, scalars=col)
        p.add_mesh_clip_plane(mesh=grid, show_edges=True, scalars=col)
    if show_grid:
        p.show_grid()

    p.show_axes()

    if show_plot:
        p.show(full_screen=True)

    return p


def extend_pandas_plot():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.plot3D = plot3D

