# import libraries
import pyvista as pv
from pyvista.utilities import generate_plane, get_array
import pandas as pd
from pandas.core.base import PandasObject
import numpy as np
from typing import Union, Tuple
import vtk
import secrets
import warnings
from pyvistaqt import BackgroundPlotter


def plot3D(blockmodel:      pd.DataFrame,
           xyz_cols:        Tuple[str, str, str] = ('x', 'y', 'z'),
           col:             str = None,
           dims:            Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
           rotation:        Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
           widget:          str = None,
           min_max:         Tuple[Union[int, float], Union[int, float]] = None,
           legend_colour:   str = 'bwr',
           window_size:     Tuple[Union[int], Union[int]] = None,
           show_edges:      bool = True,
           show_grid:       bool = True,
           shadows:         bool = True,
           show_plot:       bool = True) -> pv.Plotter:
    """
    create activate 3D vtk plot of block model that is fully interactive

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings, default ('x', 'y', 'z')
        names of x,y,z columns in model
    col: str
        attribute column to plot (e.g., tonnage, grade, etc)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints, default (0, 0, 0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    widget: {"slider","section"}
        add widgets such as slider (cut off grade) or cross-section.
    min_max: tuple of floats or ints
        minimum and maximum to colour by
        values above/below these values will just be coloured the max/min colours
    legend_colour: {optional} str, default 'bwr'
        set the legend colour scale. can be any matplotlib cmap colour spectrum.

        see: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

        see: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    window_size: {optional} tuple of ints, default (1920, 1080)
        size of plot window in pixels
    show_edges: bool, default True
        whether to show the edges of blocks or not.
    show_grid: bool, default True
        add x,y,z grid to see coordinates on plot.
    shadows: bool, default True
        whether to model shadows with a light source from the users perspective.
        if False, it is like the block model has been lit up with lights from all angles.
    show_plot: bool, default True
        whether to open active window or just return pyvistaqt.plotting.BackgroundPlotter object
        to .show() later.

    Returns
    -------
    pyvistaqt.plotting.BackgroundPlotter object & active window of block model 3D plot
    """

    # check col data to plot is int or float data - not string or bool
    """
    data_types = blockmodel.dtypes
    _dtype = str(data_types[col])

    if _dtype[0:3] != 'int' and \
       _dtype[0:5] != 'float' and \
       _dtype != 'object' and \
       _dtype != 'string' and \
       _dtype != 'bool':
        raise Exception(f'MiningPy ERROR - column to plot: {col} must be one of Pandas dtypes: int, float, object, string, boolean.')
    """
    # check for duplicate blocks and return warning
    dup_check = blockmodel.duplicated(subset=[xyz_cols[0], xyz_cols[1], xyz_cols[2]])
    xyz_cols = list(xyz_cols)

    if dup_check.sum() > 0:
        warnings.warn("There are duplicate blocks in dataframe, dropping duplicates except for the first occurrence.")
        blockmodel = blockmodel.drop_duplicates(subset=xyz_cols, keep='first')

    # check widget choice is allowed
    _widgets = ['slider',
                'COG',
                'section']
    for orientation in ['+x', '-x', '+y', '-y', '+z', '-z']:
        for section_type in ['free', 'box']:
            _widgets.append(f'section {orientation} {section_type}')
            _widgets.append(f'section {section_type} {orientation}')
            _widgets.append(f'section {orientation}')
            _widgets.append(f'section {section_type}')

    if widget is not None:
        if widget not in _widgets:
            raise Exception(f'MiningPy ERROR - widget naming not parsable.')

    # definitions for simplicity
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # check col data to plot is int or float non pandas dtype
    data_types = blockmodel.dtypes
    _dtype = str(data_types[col])

    # pandas dtypes such as Int8Dtype which support pd.na are not supported in Plot3D
    if _dtype[0:3] != 'int' and \
       _dtype[0:5] != 'float':

        # if not int or float, take copy of dataframe and explicitly define dtype
        block_model = blockmodel[[xyz_cols[0], xyz_cols[1], xyz_cols[2], col]]

        # split by np.numbers
        selection_numbers = block_model.select_dtypes(include=[np.number])
        selection_strings = block_model.select_dtypes(exclude=[np.number])

        # create new dataframes using python dtypes
        data_numbers_pd = pd.DataFrame(selection_numbers, dtype=float)
        data_strings_pd = pd.DataFrame(selection_strings, dtype=str)

        # concat together
        block_model = pd.concat([data_numbers_pd, data_strings_pd], axis=1)
        del selection_numbers, selection_strings, data_numbers_pd, data_strings_pd

    else:
        # make copy of required columns
        cols = list(xyz_cols)
        if col not in xyz_cols:
            cols.append(col)
        block_model = blockmodel[cols].copy()

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

    # rotate grid if rotation exists
    if x_rotation > 0:
        grid.rotate_x(x_rotation)
    if y_rotation > 0:
        grid.rotate_y(y_rotation)
    if z_rotation > 0:
        grid.rotate_z(z_rotation)

    # create temporary column for
    # also check if random column name exists as column
    # and if so create a different name
    _temp_idx = secrets.token_hex(nbytes=16)  # create random string

    # check if string is a column in model and create a new string
    while _temp_idx in block_model.columns:
        _temp_idx = secrets.token_hex(nbytes=16)  # update string

    # only keep blocks actually in block model dataframe
    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = xyz_cols
    cell_coords[_temp_idx] = cell_coords.index
    mask = pd.merge(cell_coords, block_model, on=xyz_cols, how='inner')

    grid = grid.extract_cells(mask[_temp_idx].values)

    centers = np.array(grid.cell_centers().points)
    cell_coords = pd.DataFrame(data=centers)
    cell_coords.columns = xyz_cols
    cell_coords[_temp_idx] = cell_coords.index

    cell_coords.set_index(xyz_cols, inplace=True)
    block_model.set_index(xyz_cols, inplace=True)
    block_model[_temp_idx] = cell_coords[_temp_idx]
    block_model.sort_values(by=[_temp_idx], inplace=True)

    block_model = block_model.reset_index()

    # inject data into VTK cells
    # handle string columns
    # (anything that isnt an int or float becomes a string)

    # non-int / float column
    if _dtype[0:3] != 'int' and \
       _dtype[0:5] != 'float':
        vtk_array = vtk.vtkStringArray()
        for idx in block_model[col].values:
            vtk_array.InsertNextValue(str(idx))
        vtk_array.SetName(str(col))

        grid.GetCellData().AddArray(vtk_array)

    else:  # int or float column
        grid.cell_data[col] = block_model[col].values  # add the data values to visualise

    # set theme
    pv.set_plot_theme("ParaView")  # just changes colour scheme

    if window_size is None:
        window_size = (1920, 1080)

    # background plotter
    plot = BackgroundPlotter(title="MiningPy 3D Plot", window_size=window_size)

    # legend settings
    if _dtype[0:5] == 'float':
        sargs = dict(interactive=True,
                     title_font_size=26,
                     label_font_size=20,
                     fmt="%.2f")

    elif _dtype[0:3] == 'int':
        sargs = dict(interactive=True,
                     title_font_size=26,
                     label_font_size=20,
                     fmt="%.0f")

    else:
        sargs = dict(interactive=True,
                     title_font_size=26,
                     label_font_size=5)

    # add mesh to plot
    if widget is None:
        plot.add_mesh(mesh=grid,
                      style='surface',
                      show_edges=show_edges,
                      scalars=col,
                      scalar_bar_args=sargs,
                      cmap=legend_colour,
                      lighting=shadows,
                      clim=min_max)

    # call appropriate function for widget
    if widget is not None:
        if widget[0:7] == "section":
            if _dtype[0:3] == 'int' or _dtype[0:5] == 'float':
                add_section_num(widget=widget,
                                plot=plot,
                                mesh=grid,
                                style='surface',
                                show_edges=show_edges,
                                scalars=col,
                                scalar_bar_args=sargs,
                                cmap=legend_colour,
                                lighting=shadows,
                                clim=min_max)
            else:
                add_section_string(widget=widget,
                                   plot=plot,
                                   mesh=grid,
                                   style='surface',
                                   show_edges=show_edges,
                                   scalars=col,
                                   scalar_bar_args=sargs,
                                   cmap=legend_colour,
                                   lighting=shadows,
                                   clim=min_max)

        if widget == "slider" or widget == "COG":
            if _dtype[0:3] == 'int' or _dtype[0:5] == 'float':
                if min_max is None:
                    _min = block_model[col].min()
                    _max = block_model[col].max()
                else:
                    _min = min_max[0]
                    _max = min_max[1]
                add_slider_num(dtype=_dtype,
                               plot=plot,
                               mesh=grid,
                               style='surface',
                               show_edges=show_edges,
                               scalars=col,
                               scalar_bar_args=sargs,
                               cmap=legend_colour,
                               lighting=shadows,
                               clim=(_min, _max))
            else:
                add_slider_string(plot=plot,
                                  mesh=grid,
                                  style='surface',
                                  show_edges=show_edges,
                                  scalars=col,
                                  scalar_bar_args=sargs,
                                  cmap=legend_colour,
                                  lighting=shadows)

    if show_grid:
        plot.show_grid()

    plot.show_axes()  # add xyz arrows to plot

    # add quick keys to plot for people
    hotkeys = 'q - quit   v - reset view'
    plot.add_text(text=hotkeys,
                  font_size=6)

    if show_plot:
        plot.app.exec()
        return plot  # pyvistaqt.BackgroundPlotter

    if not show_plot:
        plot.close()  # dont want to show plot
        return plot  # pyvistaqt.BackgroundPlotter


def add_section_num(widget, plot, mesh, style, show_edges, scalars, scalar_bar_args,
                    cmap, lighting, clim):

    # parse widget parameters from user input
    params = widget.split(' ')
    normal = (-1, 0, 0)  # default
    implicit = True  # default

    if 'free' in params:
        implicit = False
    if 'box' in params:
        implicit = True
    if '-x' in params:
        normal = (-1, 0, 0)
    if '+x' in params:
        normal = (1, 0, 0)
    if '-y' in params:
        normal = (0, -1, 0)
    if '+y' in params:
        normal = (0, 1, 0)
    if '-z' in params:
        normal = (0, 0, -1)
    if '+z' in params:
        normal = (0, 0, 1)

    # add section
    name = mesh.memory_address
    rng = mesh.get_data_range(scalars)
    if clim is None:
        clim = rng
    # mesh.set_active_scalars(scalars)

    plot.add_mesh(mesh.outline(), name=name + "outline", opacity=0.0)

    if isinstance(mesh, vtk.vtkPolyData):
        alg = vtk.vtkClipPolyData()
    else:
        alg = vtk.vtkTableBasedClipDataSet()
    alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
    # alg.SetValue(0.0)

    if not hasattr(plot, "plane_clipped_meshes"):
        plot.plane_clipped_meshes = []
    plane_clipped_mesh = pv.wrap(alg.GetOutput())
    plot.plane_clipped_meshes.append(plane_clipped_mesh)

    def callback(normal, origin):
        function = generate_plane(normal, origin)
        alg.SetClipFunction(function)  # the implicit function
        alg.Update()  # Perform the Cut
        plane_clipped_mesh.shallow_copy(alg.GetOutput())

    plane = plot.add_plane_widget(callback=callback, bounds=mesh.bounds,
                                  factor=1.25, normal=normal,
                                  color=None, tubing=False,
                                  assign_to_axis=None,
                                  origin_translation=True,
                                  outline_translation=False,
                                  implicit=implicit, origin=mesh.center)

    actor = plot.add_mesh(plane_clipped_mesh,
                          style=style,
                          scalars=scalars,
                          show_edges=show_edges,
                          scalar_bar_args=scalar_bar_args,
                          cmap=cmap,
                          lighting=lighting,
                          clim=clim)

    return actor


def add_section_string(widget, plot, mesh, style, show_edges, scalars, scalar_bar_args,
                       cmap, lighting, clim):

    # parse widget parameters from user input
    params = widget.split(' ')
    normal = (-1, 0, 0)  # default
    implicit = True  # default

    if 'free' in params:
        implicit = False
    if 'box' in params:
        implicit = True
    if '-x' in params:
        normal = (-1, 0, 0)
    if '+x' in params:
        normal = (1, 0, 0)
    if '-y' in params:
        normal = (0, -1, 0)
    if '+y' in params:
        normal = (0, 1, 0)
    if '-z' in params:
        normal = (0, 0, -1)
    if '+z' in params:
        normal = (0, 0, 1)

    # add section
    name = mesh.memory_address

    plot.add_mesh(mesh.outline(), name=name + "outline", opacity=0.0)
    plot.add_mesh(mesh=mesh,
                  style=style,
                  scalars=scalars,
                  show_edges=show_edges,
                  scalar_bar_args=scalar_bar_args,
                  cmap=cmap,
                  lighting=lighting,
                  clim=clim,
                  opacity=0.0)

    strings = mesh.get_array(scalars)
    strings_unique = set(strings)
    strings_unique = sorted(strings_unique)
    for idx, string in enumerate(strings_unique):
        strings[strings == string] = idx

    strings = strings.astype(int)
    mesh.cell_data[scalars+'_int'] = strings

    rng = mesh.get_data_range(scalars+'_int')
    if clim is None:
        clim = rng

    if isinstance(mesh, vtk.vtkPolyData):
        alg = vtk.vtkClipPolyData()
    else:
        alg = vtk.vtkTableBasedClipDataSet()
    alg.SetInputDataObject(mesh)  # Use the grid as the data we desire to cut
    # alg.SetValue(0.0)

    if not hasattr(plot, "plane_clipped_meshes"):
        plot.plane_clipped_meshes = []
    plane_clipped_mesh = pv.wrap(alg.GetOutput())
    plot.plane_clipped_meshes.append(plane_clipped_mesh)

    def callback(normal, origin):
        function = generate_plane(normal, origin)
        alg.SetClipFunction(function)  # the implicit function
        alg.Update()  # Perform the Cut
        plane_clipped_mesh.shallow_copy(alg.GetOutput())

    plane = plot.add_plane_widget(callback=callback, bounds=mesh.bounds,
                                  factor=1.25, normal=normal,
                                  color=None, tubing=False,
                                  assign_to_axis=None,
                                  origin_translation=True,
                                  outline_translation=False,
                                  implicit=implicit, origin=mesh.center)

    actor = plot.add_mesh(plane_clipped_mesh,
                          style=style,
                          scalars=scalars+'_int',
                          show_edges=show_edges,
                          show_scalar_bar=False,
                          cmap=cmap,
                          lighting=lighting,
                          clim=clim)

    return actor


def add_slider_num(dtype, plot, mesh, style, show_edges, scalars, scalar_bar_args,
                   cmap, lighting, clim):

    name = mesh.memory_address

    field, scalars = mesh.active_scalars_info
    arr = get_array(mesh=mesh, name=scalars, preference='cell', err=True)

    if arr is None:
        raise ValueError('No arrays present to threshold.')

    # rng = mesh.get_data_range(scalars)  # get clim
    # if clim is None:
    #     clim = rng

    mesh.set_active_scalars(scalars)

    plot.add_mesh(mesh.outline(), name=name + "outline", opacity=0.0)

    alg = vtk.vtkThreshold()
    alg.SetInputDataObject(mesh)
    alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)  # args: (idx, port, connection, field, name)
    alg.SetUseContinuousCellRange(False)

    if not hasattr(plot, "threshold_meshes"):
        plot.threshold_meshes = []
    threshold_mesh = pv.wrap(alg.GetOutput())
    plot.threshold_meshes.append(threshold_mesh)

    def callback_float(value, widget):
        alg.ThresholdByUpper(value)
        alg.Update()
        threshold_mesh.shallow_copy(alg.GetOutput())

    def callback_int(value, widget):
        _rounded_value = int(round(float(value), 0))
        widget.GetRepresentation().SetValue(_rounded_value)

        alg.ThresholdByUpper(_rounded_value)
        alg.Update()
        threshold_mesh.shallow_copy(alg.GetOutput())

    if dtype[0:3] == 'int' or dtype[0:5] == 'float':
        if dtype[0:3] == 'int':
            callback = callback_int
        if dtype[0:5] == 'float':
            callback = callback_float

        plot.add_slider_widget(callback=callback,
                               rng=clim,
                               title=f'{scalars} slider',
                               color=None,
                               pointa=(0.25, 0.92),
                               pointb=(0.75, 0.92),
                               value=clim[0],
                               event_type='always',
                               pass_widget=True)

    actor = plot.add_mesh(mesh=threshold_mesh,
                          scalars=scalars,
                          reset_camera=False,
                          show_edges=show_edges,
                          scalar_bar_args=scalar_bar_args,
                          cmap=cmap,
                          lighting=lighting,
                          style=style,
                          clim=clim)

    return actor


def add_slider_string(plot, mesh, style, show_edges, scalars, scalar_bar_args,
                      cmap, lighting):

    name = mesh.memory_address

    arr, field = get_array(mesh, scalars, preference='cell',)

    data = set(mesh.cell_data[scalars])
    data = list(sorted(data))

    plot.add_mesh(mesh.outline(), name=name + "outline", opacity=0.0)
    plot.add_mesh(mesh=mesh,
                  style=style,
                  scalars=scalars,
                  show_edges=show_edges,
                  scalar_bar_args=scalar_bar_args,
                  cmap=cmap,
                  lighting=lighting,
                  opacity=0.0)

    strings = mesh.get_array(scalars)
    strings_unique = set(strings)
    strings_unique = sorted(strings_unique)
    for idx, string in enumerate(strings_unique):
        strings[strings == string] = idx

    strings = strings.astype(int)
    mesh.cell_data[scalars+'_int'] = strings

    alg = vtk.vtkThreshold()
    alg.SetInputDataObject(mesh)
    alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars+'_int')  # args: (idx, port, connection, field, name)
    alg.SetUseContinuousCellRange(False)

    if not hasattr(plot, "threshold_meshes"):
        plot.threshold_meshes = []
    threshold_mesh = pv.wrap(alg.GetOutput())
    plot.threshold_meshes.append(threshold_mesh)

    n_states = len(data)
    if n_states == 0:
        raise ValueError("The input list of values is empty")
    delta = (n_states - 1) / float(n_states)
    # avoid division by zero in case there is only one element
    delta = 1 if delta == 0 else delta

    def callback(value, widget):
        _rounded_value = int(round(float(value), 0))
        widget.GetRepresentation().SetValue(_rounded_value)
        alg.ThresholdByUpper(_rounded_value)
        alg.Update()
        threshold_mesh.shallow_copy(alg.GetOutput())

    slider_widget = plot.add_slider_widget(callback=callback,
                                           rng=[0, n_states - 1],
                                           value=0,
                                           pointa=(0.25, 0.92),
                                           pointb=(0.75, 0.92),
                                           color=None,
                                           event_type='always',
                                           pass_widget=True)

    slider_rep = slider_widget.GetRepresentation()
    slider_rep.ShowSliderLabelOff()

    def title_callback(widget, event):
        value = widget.GetRepresentation().GetValue()
        idx = int(value / delta)
        # handle limit index
        if idx == n_states:
            idx = n_states - 1
        slider_rep.SetTitleText(data[idx])

    slider_widget.AddObserver(vtk.vtkCommand.InteractionEvent, title_callback)

    title_callback(slider_widget, None)

    actor = plot.add_mesh(mesh=threshold_mesh,
                          scalars=scalars,
                          reset_camera=False,
                          show_edges=show_edges,
                          show_scalar_bar=False,
                          cmap=cmap,
                          lighting=lighting,
                          style=style)

    return actor


def extend_pandas_plot():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.plot3D = plot3D

