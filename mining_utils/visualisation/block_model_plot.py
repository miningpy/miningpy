# import libraries
import pyvista as pv
import pandas as pd
from pandas.core.base import PandasObject
import numpy as np


def plot3D(model=None,                  # block model dataframe
           coord_cols=('x', 'y', 'z'),  # column names for x,y,z coordinates
           col=None,                    # attribute column to plot (i.e. tonnage, grade, etc)
           dims=None,                   # tuple (xdim,ydim,zdim) of blocks
           widget=None,                 # add widgets such as slider (cut off grade) or cross-section. "COG" or "section"
           show_grid=True):             # add x,y,z grid to see coordinates on plot

    # read in block model
    block_model = model[coord_cols+[col]].copy()

    # Create the spatial reference
    grid = pv.UniformGrid()

    x_orig = block_model[coord_cols[0]].min() - (0.5 * dims[0])
    y_orig = block_model[coord_cols[1]].min() - (0.5 * dims[1])
    z_orig = block_model[coord_cols[2]].min() - (0.5 * dims[2])
    origin = (x_orig, y_orig, z_orig)

    # Edit the spatial reference
    grid.origin = origin  # block model origin -  bottom left corner of the model
    grid.spacing = dims  # x,y,z dimensions of blocks

    # build the block model framework
    nx = int((block_model[coord_cols[0]].max() - block_model[coord_cols[0]].min()) / grid.spacing[0]) + 1  # number of blocks in x dimension
    ny = int((block_model[coord_cols[1]].max() - block_model[coord_cols[1]].min()) / grid.spacing[1]) + 1  # number of blocks in y dimension
    nz = int((block_model[coord_cols[2]].max() - block_model[coord_cols[2]].min()) / grid.spacing[2]) + 1  # number of blocks in z dimension
    grid.dimensions = [nx+1, ny+1, nz+1]  # need extra dimension - 4th dimension is for data (x,y,z,data)

    # only keep blocks actually in block model dataframe
    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = coord_cols
    cell_coords['idx'] = cell_coords.index
    mask = pd.merge(cell_coords, block_model, on=coord_cols, how='inner')

    grid = grid.extract_cells(mask['idx'].values)

    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = coord_cols
    cell_coords['idx'] = cell_coords.index

    cell_coords.set_index(coord_cols, inplace=True)
    block_model.set_index(coord_cols, inplace=True)
    block_model['idx'] = cell_coords['idx']
    block_model.sort_values(by=['idx'], inplace=True)

    grid.cell_arrays[col] = block_model[col].values  # add the data values to visualise

    # set theme
    pv.set_plot_theme("ParaView")  # just changes colour scheme

    p = pv.Plotter(notebook=False)

    if widget == None:
        p.add_mesh(grid, show_edges=True, scalars=col)
    if widget == "section":
        p.add_mesh_clip_plane(mesh=grid, show_edges=True, scalars=col)
    if widget == "COG":
        p.add_mesh_threshold(mesh=grid, show_edges=True, scalars=col)
    if show_grid:
        p.show_grid()

    p.show_axes()
    p.show(full_screen=True)


def extend_pandas():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.plot3D = plot3D

