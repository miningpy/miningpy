"""
applies model calculations to pandas dataframe of block model as a pandas method
extends the pandas API for mining engineering applications
inspiration of this methodology from: https://github.com/pmorissette/ffn

Authors: Iain Fullelove

License: MIT
"""

# import libraries
import pandas as pd
import numpy as np
from math import sin, cos, pi
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
import datetime
import warnings


def ijk(blockmodel: pd.DataFrame,
        method: str = 'ijk',
        indexing: int = 0,
        xyz_cols: Tuple[str, str, str] = None,
        origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
        rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
        ijk_cols: Tuple[str, str, str] = ('i', 'j', 'k'),
        print_warnings: bool = True,
        inplace: bool = False) -> pd.DataFrame:
    """
    Calculate block ijk indexes from their xyz cartesian coordinates

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    method: str, default 'ijk'
        can be used to only calculate i, or j, or k
    indexing: int, default 0
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    ijk_cols: tuple of strings, default ('i', 'j', 'k')
        name of the i,j,k columns added to the model
    print_warnings: bool, default True
        if True then will check if blocks are on a regular grid
        before the IJK calculation and print a warning to the user if
        the blocks are not regular (i.e. could get funky IJK values).
    inplace: bool, default False
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        indexed block model

    Examples
    --------
    >>> import pandas as pd
    >>> import miningpy
    ...
    >>> # block model data and framework
    >>> data = {'x': [5, 5, 15],
    ...         'y': [5, 15, 25],
    ...         'z': [5, 5, 5]}
    >>> xdim, ydim, zdim = 5, 5, 5  # regular block dimensions
    >>> xorg, yorg, zorg = 2.5, 2.5, 2.5  # model origin (corner of first block)
    ...
    >>> # Create block model from data
    >>> blockmodel = pd.DataFrame(data)
    >>> print(blockmodel)
        x   y  z
    0   5   5  5
    1   5  15  5
    2  15  25  5
    >>> # calculate i, j, k indexes
    >>> blockmodel.ijk(indexing=1,  # use ijk function just like any other Pandas function
    ...                 xyz_cols=('x', 'y', 'z'),  # input the x, y, z column names as a tuple
    ...                 origin=(xorg, yorg, zorg),
    ...                 dims=(xdim, ydim, zdim),
    ...                 inplace=True) # can do inplace True/False like other standard Pandas functions
    >>> # print results of ijk calculation
    >>> print(blockmodel)
        x   y  z  i  j  k
    0   5   5  5  1  1  1
    1   5  15  5  1  3  1
    2  15  25  5  3  5  1

    """

    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    methods_accepted = ['ijk', 'ij', 'ik', 'jk', 'i', 'j', 'k']
    indexing_accepted = [0, 1]

    # check input indexing
    if indexing in indexing_accepted:
        pass
    else:
        raise ValueError('IJK FAILED - indexing value not accepted - only 1 or 0 can be used')

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]
    icol, jcol, kcol = ijk_cols[0], ijk_cols[1], ijk_cols[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # inverse rotations to unrotate grid
    unrotation = ((-1.0 * rotation[0]), (-1.0 * rotation[1]), (-1.0 * rotation[2]))

    if x_rotation == 0 and y_rotation == 0 and z_rotation == 0:
        bm_xcol = blockmodel[xcol]
        bm_ycol = blockmodel[ycol]
        bm_zcol = blockmodel[zcol]
    else:
        bm_xcol = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                         origin=origin,
                                         rotation=unrotation,
                                         return_full_model=False,
                                         inplace=False)[xcol].copy()

        bm_ycol = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                         origin=origin,
                                         rotation=unrotation,
                                         return_full_model=False,
                                         inplace=False)[ycol].copy()

        bm_zcol = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                         origin=origin,
                                         rotation=unrotation,
                                         return_full_model=False,
                                         inplace=False)[zcol].copy()

    if method in methods_accepted:
        if 'i' in method:
            try:
                if print_warnings:
                    # check integer value isn't far from float - this can cause indexing issues
                    # throw a warning to the user if this is the case
                    # float should be within 0.00001 tolerance of integer
                    indexed_float = ((bm_xcol - xsize / 2 - xorigin) / xsize)
                    indexed_int = np.rint((bm_xcol - xsize / 2 - xorigin) / xsize).astype(int)
                    check_float = (indexed_float - indexed_int).abs()
                    if check_float.any() > 0.00001:
                        warnings.warn(
                            "MiningPy WARNING - xcol block centroids not on a regular grid - calculated IJK values may be wrong")

                icol_float = (bm_xcol - xsize / 2 - xorigin) / xsize
                icol_float = icol_float.round(decimals=0)
                blockmodel[icol] = (np.rint(icol_float) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either xcol, xorigin or xsize not defined properly')

        if 'j' in method:
            try:
                if print_warnings:
                    # check integer value isn't far from float - this can cause indexing issues
                    # throw a warning to the user if this is the case
                    # float should be within 0.00001 tolerance of integer
                    indexed_float = ((bm_ycol - ysize / 2 - yorigin) / ysize)
                    indexed_int = np.rint((bm_ycol - ysize / 2 - yorigin) / ysize).astype(int)
                    check_float = (indexed_float - indexed_int).abs()
                    if check_float.any() > 0.00001:
                        warnings.warn(
                            "MiningPy WARNING - ycol block centroids not on a regular grid - calculated IJK values may be wrong")

                jcol_float = (bm_ycol - ysize / 2 - yorigin) / ysize
                jcol_float = jcol_float.round(decimals=0)
                blockmodel[jcol] = (np.rint(jcol_float) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either ycol, yorigin or ysize not defined properly')

        if 'k' in method:
            try:
                if print_warnings:
                    # check integer value isn't far from float - this can cause indexing issues
                    # throw a warning to the user if this is the case
                    # float should be within 0.00001 tolerance of integer
                    indexed_float = ((bm_zcol - zsize / 2 - zorigin) / zsize)
                    indexed_int = np.rint((bm_zcol - zsize / 2 - zorigin) / zsize).astype(int)
                    check_float = (indexed_float - indexed_int).abs()
                    if check_float.any() > 0.00001:
                        warnings.warn(
                            "MiningPy WARNING - zcol block centroids not on a regular grid - calculated IJK values may be wrong")

                kcol_float = (bm_zcol - zsize / 2 - zorigin) / zsize
                kcol_float = kcol_float.round(decimals=0)
                blockmodel[kcol] = (np.rint(kcol_float) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either zcol, zorigin or zsize not defined properly')

        # check inplace for return
        if not inplace:
            return blockmodel
    else:
        raise ValueError('IJK FAILED - IJK method not accepted')


def xyz(blockmodel: pd.DataFrame,
        method: str = 'xyz',
        indexing: int = 0,
        ijk_cols: Tuple[str, str, str] = ('i', 'j', 'k'),
        origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
        rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
        xyz_cols: Tuple[str, str, str] = ('x', 'y', 'z'),
        inplace: bool = False) -> pd.DataFrame:
    """
    Calculate xyz cartesian cooridinates of blocks from their ijk indexes

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    method: str, default 'xyz'
        can be used to only calculate i, or j, or k
    indexing: int, default 0
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    ijk_cols: tuple of strings, default ('i', 'j', 'k')
        name of the i,j,k columns added to the model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    xyz_cols: tuple of strings, default ('x', 'y', 'z')
        names of x,y,z columns in model
    inplace: bool, default False
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        indexed block model
    """

    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    methods_accepted = ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']
    indexing_accepted = [0, 1]

    # check input indexing
    if indexing in indexing_accepted:
        pass
    else:
        raise ValueError('XYZ FAILED - indexing value not accepted - only 1 or 0 can be used')

    # definitions for simplicity
    icol, jcol, kcol = ijk_cols[0], ijk_cols[1], ijk_cols[2]
    xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    if method in methods_accepted:
        if 'x' in method:
            try:
                blockmodel[xcol] = ((blockmodel[icol] - indexing) * xsize) + xorigin + (xsize / 2)
            except ValueError:
                raise ValueError('XYZ FAILED - either icol, xorigin or xsize not defined properly')

        if 'y' in method:
            try:
                blockmodel[ycol] = ((blockmodel[jcol] - indexing) * ysize) + yorigin + (ysize / 2)
            except ValueError:
                raise ValueError('XYZ FAILED - either jcol, yorigin or ysize not defined properly')

        if 'z' in method:
            try:
                blockmodel[zcol] = ((blockmodel[kcol] - indexing) * zsize) + zorigin + (zsize / 2)
            except ValueError:
                raise ValueError('XYZ FAILED - either kcol, zorigin or zsize not defined properly')

    else:
        raise ValueError('XYZ FAILED - XYZ method not accepted')

    if x_rotation != 0 or y_rotation != 0 or z_rotation != 0:
        mod_rotated = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                             origin=origin,
                                             rotation=rotation,
                                             return_full_model=False,
                                             inplace=False)

        blockmodel[xcol] = mod_rotated[xcol].copy()
        blockmodel[ycol] = mod_rotated[ycol].copy()
        blockmodel[zcol] = mod_rotated[zcol].copy()

    # check inplace for return
    if not inplace:
        return blockmodel


def rotate_grid(blockmodel: pd.DataFrame,
                xyz_cols: Tuple[str, str, str] = ('x', 'y', 'z'),
                origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                return_full_model: bool = True,
                derotate: bool = False,
                inplace: bool = False) -> Union[pd.DataFrame, dict]:
    """
    Rotate block model relative to cartesian grid
    This method uses a rotation matrix method
    Rotation is done using the right hand rule

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings, default ('x', 'y', 'z')
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    return_full_model: bool, default True
        whether to return the full block model or just a dict of the rotated x,y,z coordinates
    derotate: bool
        whether to rotate a model or derotate it back to it's normal orthogonal coordinate system
        this parameter exists because using the reverse angles in more than one dimension will not derotate a model
    inplace: bool, default False
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        rotated block model or dict of rotated x,y,z coordinates
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # check inplace
    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    if x_rotation != 0 or y_rotation != 0 or z_rotation != 0:
        try:
            trxx = blockmodel[xcol] - xorigin
        except ValueError:
            raise ValueError('GRID ROTATION FAILED - either xcol or xorigin not defined properly')
        try:
            tryy = blockmodel[ycol] - yorigin
        except ValueError:
            raise ValueError('GRID ROTATION FAILED - either ycol or yorigin not defined properly')
        try:
            trzz = blockmodel[zcol] - zorigin
        except ValueError:
            raise ValueError('GRID ROTATION FAILED - either zcol or zorigin not defined properly')
    else:
        return blockmodel

    x_sin = sin(x_rotation * (pi / 180.0))
    x_cos = cos(x_rotation * (pi / 180.0))
    y_sin = sin(y_rotation * (pi / 180.0))
    y_cos = cos(y_rotation * (pi / 180.0))
    z_sin = sin(z_rotation * (pi / 180.0))
    z_cos = cos(z_rotation * (pi / 180.0))

    # define rotation matrix
    rotation_matrix = np.zeros((3, 3), dtype=np.float64)
    rotation_matrix[0][0] = z_cos * y_cos
    rotation_matrix[0][1] = (z_cos * y_sin * x_sin) - (z_sin * x_cos)
    rotation_matrix[0][2] = (z_cos * y_sin * x_cos) + (z_sin * x_sin)

    rotation_matrix[1][0] = z_sin * y_cos
    rotation_matrix[1][1] = (z_sin * y_sin * x_sin) + (z_cos * x_cos)
    rotation_matrix[1][2] = (z_sin * y_sin * x_cos) - (z_cos * x_sin)

    rotation_matrix[2][0] = -1.0 * y_sin
    rotation_matrix[2][1] = y_cos * x_sin
    rotation_matrix[2][2] = y_cos * x_cos

    if derotate:
        rotation_matrix = np.transpose(rotation_matrix)

    # rotation matrix multiplication
    xrot = (trxx * (rotation_matrix[0][0])) + (tryy * (rotation_matrix[0][1])) + (trzz * (rotation_matrix[0][2]))
    yrot = (trxx * (rotation_matrix[1][0])) + (tryy * (rotation_matrix[1][1])) + (trzz * (rotation_matrix[1][2]))
    zrot = (trxx * (rotation_matrix[2][0])) + (tryy * (rotation_matrix[2][1])) + (trzz * (rotation_matrix[2][2]))
    del trxx, tryy, trzz

    blockmodel[xcol] = xrot + xorigin
    blockmodel[ycol] = yrot + yorigin
    blockmodel[zcol] = zrot + zorigin
    del xrot, yrot, zrot

    # check inplace for return
    if not inplace:
        if return_full_model:
            return blockmodel
        if not return_full_model:
            return {'x': blockmodel[xcol],
                    'y': blockmodel[ycol],
                    'z': blockmodel[zcol]}


def group_weighted_average(blockmodel: pd.DataFrame,
                           avg_cols: Union[str, List[str]],
                           weight_col: str,
                           group_cols: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    weighted average of block model attribute(s)

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    avg_cols: str or list of str
        column(s) to take the weighted average
    weight_col: str
        column to weight on. Example the tonnes column
    group_cols: str or list of str
        the columns you want to group on. Either single column or list of columns

    Returns
    -------
    pandas.DataFrame
        block model
    """

    average_cols = []
    if group_cols is not None:  # group cols can be None if doing weight average over all data
        groupby_cols = []

    if isinstance(avg_cols, str):
        average_cols.append(avg_cols)
    elif isinstance(avg_cols, list):
        average_cols = avg_cols
    else:
        raise Exception('Average columns parameter must be single column name or list of column names')

    if group_cols is not None:
        if isinstance(group_cols, str):
            groupby_cols.append(group_cols)
        elif isinstance(group_cols, list):
            groupby_cols = group_cols
        else:
            raise Exception('Groupby columns parameter must be single column name or list of column names')

    # make a copy of input dataframe so we don't alter the original model
    # only copy columns we need to do group weight average
    if group_cols is not None:
        data_in = blockmodel[average_cols + [weight_col] + groupby_cols].copy()
    else:
        data_in = blockmodel[average_cols + [weight_col]].copy()

    if group_cols is not None:
        # set index for data in based on group cols
        data_in = data_in.set_index(groupby_cols)
        # mass multiply all the grade by the weight col
        data_units = data_in[average_cols].multiply(data_in[weight_col], axis='index')
        data_units = data_units.reset_index()
        data_units = data_units.groupby(groupby_cols)[average_cols].sum()
        # sum the weights
        data_weights = pd.DataFrame(data_in.groupby(groupby_cols)[weight_col].sum())
        data_units[weight_col] = data_weights[weight_col]  # paste the weight col back into the units DF
        # mass divide the units by the weight col and presto
        data_out = data_units[average_cols].divide(data_units[weight_col], axis='index')
        data_out = data_out.fillna(0.0)  # in case of zero weights
        return data_out
    else:  # not grouping by any attributes
        # mass multiply all the grade by the weight col
        data_units = data_in[average_cols].multiply(data_in[weight_col], axis='index')
        data_units = data_units[average_cols].sum()
        # sum the weights
        data_weights = data_in[weight_col].sum()
        # mass divide the units by the weight col and presto
        data_out = data_units / data_weights
        data_out = data_out.fillna(0.0)  # in case of zero weights
        return data_out


def nblocks_xyz(blockmodel: pd.DataFrame,
                xyz_cols: Tuple[str, str, str] = None,
                dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0)
                ) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
    """
    Number of blocks along the x,y,z axis.
    If the model is rotated, it is unrotated and then the number
    of blocks in the x,y,z axis is calculated.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees

    Returns
    -------
    tuple of floats
        Number of blocks along the x,y,z axis.
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    mod = blockmodel[[xcol, ycol, zcol]].copy()
    mod.ijk(xyz_cols=xyz_cols,
            origin=origin,
            dims=dims,
            rotation=rotation,
            inplace=True)

    nx = mod['i'].max() - mod['i'].min() + 1
    ny = mod['j'].max() - mod['j'].min() + 1
    nz = mod['k'].max() - mod['k'].min() + 1

    return nx, ny, nz


def vulcan_csv(blockmodel: pd.DataFrame,
               path: str = None,
               var_path: str = None,
               xyz_cols: Tuple[str, str, str] = None,
               dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
               inplace: bool = False) -> pd.DataFrame:
    """
    transform pandas.Dataframe block model into Vulcan import compatible CSV format.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for vulcan csv block model file
    var_path: {optional} str
        filename for csv that lists the Vulcan dtype of each column in block model (used if manually creating bdf)
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    inplace: bool, default False
        whether to do calculation inplace (i.e. add Vulcan headers inplace) or return pandas.DataFrame with Vulcan headers

    Returns
    -------
    pandas.DataFrame
        block model in Vulcan import CSV format
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]

    # check inplace
    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    # create columns required for vulcan
    if type(xsize) is int or type(xsize) is float:
        blockmodel['dim_x'] = xsize
    elif type(xsize) is str:
        blockmodel = blockmodel.rename(columns={xsize: "dim_x"}, errors="raise")
    else:
        raise Exception("x dimension must be int, float or str (block model column name).")

    if type(ysize) is int or type(ysize) is float:
        blockmodel['dim_y'] = ysize
    elif type(ysize) is str:
        blockmodel = blockmodel.rename(columns={ysize: "dim_y"}, errors="raise")
    else:
        raise Exception("y dimension must be int, float or str (block model column name).")

    if type(zsize) is int or type(zsize) is float:
        blockmodel['dim_z'] = zsize
    elif type(zsize) is str:
        blockmodel = blockmodel.rename(columns={zsize: "dim_z"}, errors="raise")
    else:
        raise Exception("z dimension must be int, float or str (block model column name).")

    blockmodel['volume'] = blockmodel['dim_x'] * blockmodel['dim_y'] * blockmodel['dim_z']

    # order fields correctly for Vulcan
    vulcan_fields = [xcol, ycol, zcol, 'dim_x', 'dim_y', 'dim_z', 'volume']
    blockmodel.set_index(vulcan_fields, inplace=True)
    blockmodel.reset_index(inplace=True)

    # Export model to Vulcan formatted CSV (use this to create the bdf file)
    vulcan_variables = pd.DataFrame({'Variable': list(blockmodel.columns.values), 'Data Type': list(blockmodel.dtypes)})
    vulcan_variables['Default Value'] = -99.0
    vulcan_header = vulcan_variables.copy()

    pandas_vulcan_dtypes = {'int64': 'Integer (Integer * 4)',
                            'float64': 'Double (Real * 8)',
                            'object': 'Name (Translation Table)'}

    for pandas, vulcan in pandas_vulcan_dtypes.items():
        mask = vulcan_header['Data Type'] == pandas
        vulcan_header.loc[mask, 'Data Type'] = vulcan

    if var_path is not None:
        vulcan_variables.to_csv(var_path, index=False)

    # rename column headers to vulcan format
    blockmodel.rename(columns={xcol: 'centroid_x',
                               ycol: 'centroid_y',
                               zcol: 'centroid_z'}, inplace=True)

    # create vulcan header df to append to start of block model

    vulcan_header['Data Type'] = 'double'  # data types are read from bdf, not csv. so doesn't matter
    vulcan_header.rename(columns={'Variable': 'Variable descriptions:',
                                  'Data Type': 'Variable types:',
                                  'Default Value': 'Variable defaults:'}, inplace=True)
    mask = vulcan_header['Variable descriptions:'].isin(vulcan_fields)
    vulcan_header.loc[mask, 'Variable types:'] = ''
    vulcan_header.loc[mask, 'Variable defaults:'] = ''
    vulcan_header['Variable descriptions:'] = ''

    vulcan_header = vulcan_header.T

    vulcan_header.columns = blockmodel.columns

    vulcan_header['centroid_x'] = vulcan_header.index
    vulcan_header.reset_index(inplace=True, drop=True)

    blockmodel = pd.concat([vulcan_header, blockmodel], axis=0, sort=False, ignore_index=True)

    blockmodel.to_csv(path, index=False)

    if not inplace:
        return blockmodel


def check_regular_extents(blockmodel: pd.DataFrame,
                          end_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]],
                          dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]],
                          origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                          original_rotation: Tuple[
                              Union[int, float, str], Union[int, float, str], Union[int, float, str]] = (0, 0, 0),
                          xyz_cols: Tuple[str, str, str] = None,
                          start_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (
                          0.0, 0.0, 0.0),
                          ):
    """Verify if the model's blocks are physically within the extents specified.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    end_offset: {required} tuple of ints or floats
        furthest distance of the unrotated block model edges from the origin in the x, y and z directions.
    dims: {required} tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    origin: {optional} tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    original_rotation: {optional} tuple of ints and floats
        rotation which has already been applied to the block model
        the script will de-rotate the bm with these values to orientate it back to the orthonormal xyz axes
    xyz_cols: {optional} tuple of strings
        names of x,y,z columns in model
        defaults to None
    start_offset: {optional} tuple of ints or floats
        closest distance of the unrotated block model edges from the origin in the x, y and z directions.
        defaults to 0,0,0

    Returns
    -------
    bool
        True if test succeeds, False if it doesn't
    """

    # TODO make this better, or cancel this error checking entirely
    bm = pd.DataFrame()
    try:
        bm = blockmodel[[xyz_cols[0], xyz_cols[1], xyz_cols[2]]]
    except:
        try:
            bm = blockmodel[['centroid_x', 'centroid_y', 'centroid_z']]
        except:
            try:
                bm = blockmodel[['x', 'y', 'z']]
            except:
                try:
                    bm = blockmodel[['east', 'north', 'elev']]
                except:
                    try:
                        bm = blockmodel[['easting', 'northing', 'elevation']]
                    except:
                        raise Exception("Could not determine wich columns of the Block Model are the xyz coordinates")

    bm.columns = ['x', 'y', 'z']

    # Checks if the block model is a vulcan formatted block model
    if bm.at[0, 'x'] == 'Variable descriptions:':
        bm = bm.iloc[3:, :]
        bm = bm.astype('float64')

    # TODO make sure this actually derotates a model
    bm = bm.rotate_grid(origin=origin,
                        rotation=original_rotation,
                        derotate=True)

    bm_xmax = bm['x'].max()
    bm_ymax = bm['y'].max()
    bm_zmax = bm['z'].max()
    bm_xmin = bm['x'].min()
    bm_ymin = bm['y'].min()
    bm_zmin = bm['z'].min()

    th_xmax = origin[0] + end_offset[0] - dims[0] / 2
    th_ymax = origin[1] + end_offset[1] - dims[1] / 2
    th_zmax = origin[2] + end_offset[2] - dims[2] / 2
    th_xmin = origin[0] + start_offset[0] + dims[0] / 2
    th_ymin = origin[1] + start_offset[1] + dims[1] / 2
    th_zmin = origin[2] + start_offset[2] + dims[2] / 2

    out_xmax = abs(bm_xmax - th_xmax) / dims[0] > 0.01
    out_ymax = abs(bm_ymax - th_ymax) / dims[1] > 0.01
    out_zmax = abs(bm_zmax - th_zmax) / dims[2] > 0.01
    out_xmin = abs(bm_xmin - th_xmin) / dims[0] > 0.01
    out_ymin = abs(bm_ymin - th_ymin) / dims[1] > 0.01
    out_zmin = abs(bm_zmin - th_zmin) / dims[2] > 0.01
    blocks_outside = out_xmax or out_ymax or out_zmax or \
                     out_xmin or out_ymin or out_zmin

    in_xmax = abs(bm_xmax - th_xmax) / dims[0] > 0.01
    in_ymax = abs(bm_ymax - th_ymax) / dims[1] > 0.01
    in_zmax = abs(bm_zmax - th_zmax) / dims[2] > 0.01
    in_xmin = abs(bm_xmin - th_xmin) / dims[0] > 0.01
    in_ymin = abs(bm_ymin - th_ymin) / dims[1] > 0.01
    in_zmin = abs(bm_zmin - th_zmin) / dims[2] > 0.01
    blocks_inside = in_xmax or in_ymax or in_zmax or \
                    in_xmin or in_ymin or in_zmin

    if blocks_outside:
        warnings.warn('Big Warning, blocks appears to exist outside the bm extents!\n')

    if blocks_inside:
        warnings.warn('Small Warning, blocks appears to not reach the bm extents!\n')

    if not (blocks_outside or blocks_inside):
        return True
    else:
        return False


def vulcan_bdf(blockmodel: pd.DataFrame,
               path: str = None,
               origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
               start_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0.0, 0.0, 0.0),
               end_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               format: str = 'T') -> bool:
    """
    create a Vulcan block definition file from a vulcan block model.
    This script creates a BDF from a vulcan block model csv that can be imported into Vulcan.
    It assumes that bearing, dip and plunge are the default.
    values for the block model. Variables are given a default value of -99.0, a blank description and type 'double'.
    This script will define the parent schema.
    All of these values can be edited within Vulcan once the script has been run and bdf imported.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model in Vulcan CSV import compatible format (use funtion: "vulcan_csv")
    path: str
        filename for vulcan bdf file
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    start_offset: tuple of floats or ints, default (0.0, 0.0, 0.0)
        minimum offset along the x,y,z axes
    end_offset: tuple of floats or ints
        maximum offset along the x,y,z axes
    format: {'C', 'T'}, default 'T'
        block model file format (classic = C, extended = T)

    Returns
    -------
    True if Vulcan .bdf file is exported with no errors
    """

    # definitions for simplicity
    xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    start_xoffset, start_yoffset, start_zoffset = start_offset[0], start_offset[1], start_offset[2]
    end_xoffset, end_yoffset, end_zoffset = end_offset[0], end_offset[1], end_offset[2]

    # input checks for format
    assert format == 'T' or format == 'C', "Define format. format='C' for classic, format='T' for extended"

    # listing variables in vulcan block model (removing vulcan headers)
    variables = list(blockmodel.columns)
    del variables[0:7]

    # writing bdf file header
    bdf = open(path, "w+")
    bdf.write("*\n")
    bdf.write(f"*  Written: {datetime.datetime.now()}*\n")
    bdf.write("*\n")
    bdf.write("BEGIN$DEF header\n")
    bdf.write(" ")
    bdf.write("NO_align_blocks\n")
    bdf.write(" ")
    bdf.write("bearing=90.000000000000\n")
    bdf.write(" ")
    bdf.write("dip=0.000000000000\n")
    bdf.write(" ")
    bdf.write("n_schemas=1.000000000000\n")
    bdf.write(" ")
    bdf.write(f"n_variables={len(variables)}\n")
    bdf.write(" ")
    bdf.write("plunge=0.000000000000\n")
    bdf.write(" ")
    bdf.write(f"x_origin={xorigin}\n")
    bdf.write(" ")
    bdf.write(f"y_origin={yorigin}\n")
    bdf.write(" ")
    bdf.write(f"z_origin={zorigin}\n")
    bdf.write("END$DEF header\n")

    # writing variable default, description, name and type
    count = 1
    for variable in variables:
        bdf.write("*\n")
        bdf.write(f"BEGIN$DEF variable_{count}\n")
        bdf.write(" ")
        bdf.write("default='-99.0'\n")
        bdf.write(" ")
        bdf.write("description=' '\n")
        bdf.write(" ")
        bdf.write(f"name='{variables[count - 1]}'\n")
        bdf.write(" ")
        bdf.write("type='double'\n")
        bdf.write(f"END$DEF variable_{count}\n")

        count += 1

    # Writing parent schema
    bdf.write("*\n")
    bdf.write("BEGIN$DEF schema_1\n")
    bdf.write(" ")
    bdf.write(f"block_max_x={xsize}\n")
    bdf.write(" ")
    bdf.write(f"block_max_y={ysize}\n")
    bdf.write(" ")
    bdf.write(f"block_max_z={zsize}\n")
    bdf.write(" ")
    bdf.write(f"block_min_x={xsize}\n")
    bdf.write(" ")
    bdf.write(f"block_min_y={ysize}\n")
    bdf.write(" ")
    bdf.write(f"block_min_z={zsize}\n")
    bdf.write(" ")
    bdf.write("description='parent'\n")
    bdf.write(" ")
    bdf.write(f"schema_max_x={end_xoffset}\n")
    bdf.write(" ")
    bdf.write(f"schema_max_y={end_yoffset}\n")
    bdf.write(" ")
    bdf.write(f"schema_max_z={end_zoffset}\n")
    bdf.write(" ")
    bdf.write(f"schema_min_x={start_xoffset}\n")
    bdf.write(" ")
    bdf.write(f"schema_min_y={start_yoffset}\n")
    bdf.write(" ")
    bdf.write(f"schema_min_z={start_zoffset}\n")
    bdf.write("END$DEF schema_1\n")
    bdf.write("*\n")

    # Writing Boundaries
    bdf.write("BEGIN$DEF boundaries\n")
    bdf.write(" ")
    bdf.write("n_boundaries=0.000000000000\n")
    bdf.write(" ")
    bdf.write("n_exceptions=0.000000000000\n")
    bdf.write(" ")
    bdf.write("n_limits=0.000000000000\n")
    bdf.write("END$DEF boundaries\n")
    bdf.write("*\n")

    # Writing File format
    bdf.write("BEGIN$DEF file_format\n")
    bdf.write(" ")
    bdf.write(f"file_format='{format}'\n")
    bdf.write("END$DEF\n")

    bdf.write("END$FILE")
    bdf.close()

    return True


def geometric_reblock(blockmodel: pd.DataFrame,
                      xyz_cols: Tuple[str, str, str] = None,
                      origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                      dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                      reblock_multiplier: Union[int, float] = None,
                      varlist_agg: dict = None,
                      varlist_no_agg: dict = None,
                      min_cols: list = None,
                      max_cols: list = None,
                      ):
    """
    reblock regular block model into larger or smaller blocks (split or aggregate blocks)
    can be used as a tool for geometrically aggregating blocks in bench-phases
    reblock factor (n) must be 2^n (i.e. blocks are either doubled, halved, quartered, etc in size)
    cannot just define any new x,y,z dimension, must be multiple of current parent block size.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    pandas.DataFrame
        reblocked block model
    """

    # make ijks
    blockmodel = blockmodel.ijk(xyz_cols=xyz_cols,
                                origin=origin,
                                dims=dims)

    # check for duplicates
    if blockmodel.duplicated(subset=['i', 'j', 'k']).sum() > 0:
        blockmodel = blockmodel.drop_duplicates(subset=['i', 'j', 'k'])  # remove duplicate blocks
        warnings.UserWarning("duplicate blocks removed")

    # ijk on multiplier to create super or sub blocks
    blockmodel['i2'] = np.ceil(blockmodel['i'] / reblock_multiplier).astype(int)
    blockmodel['j2'] = np.ceil(blockmodel['j'] / reblock_multiplier).astype(int)
    blockmodel['k2'] = np.ceil(blockmodel['k'] / reblock_multiplier).astype(int)

    group_cols = ['i2', 'j2', 'k2']

    # calculate average grades for each tonnage
    tonne_cols = list(varlist_agg.keys())
    inv_grades = []
    for ton_var in tonne_cols:
        if len(varlist_agg[ton_var]) > 0:  # check the tonnage actually has associated grades
            inv_grades.append(blockmodel.group_weighted_average(avg_cols=varlist_agg[ton_var], weight_col=ton_var,
                                                                group_cols=group_cols))

    # calculate total tonnage/volumes
    inv_tonnes = blockmodel.groupby(group_cols)[tonne_cols].sum()

    # take highest value
    if len(max_cols) > 0:
        inv_max = blockmodel.groupby(group_cols)[max_cols].max()
        inv_grades.append(inv_max)

    # take lowest value
    if len(min_cols) > 0:
        inv_min = blockmodel.groupby(group_cols)[min_cols].min()
        inv_grades.append(inv_min)

    # join average grades / total tonnages
    inv_all = pd.concat([inv_tonnes] + inv_grades, axis='columns')
    blockmodel = inv_all
    return blockmodel


def attribute_reblock(blockmodel: pd.DataFrame):
    """
    bin the attributes & tonnes of a block model for each bench/phase
    heavily used in mine strategic scheduling

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model

    Returns
    -------
    pandas.DataFrame
        block model with binned attributes & tonnes for each bench
    """
    raise Exception("MiningPy function {attribute_reblock} hasn't been created yet")


def model_origin(blockmodel: pd.DataFrame,
                 xyz_cols: Tuple[str, str, str] = None,
                 dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                 ) -> Tuple[float, float, float]:
    """
    calculate the origin of a block model grid relative to its current xyz grid
    origin is the corner of the block with min xyz coordinates

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe

    Returns
    -------
    tuple of floats
        origin of a block model for each axis (x,y,z)
    """

    # x origin
    if type(dims[0]) is int or type(dims[0]) is float:
        xorigin = blockmodel[xyz_cols[0]].min() - (dims[0] / 2)
    elif type(dims[0]) is str:
        xorigin = (blockmodel[xyz_cols[0]] - (blockmodel[dims[0]] / 2)).min()
    else:
        raise Exception("x dimension must be int, float or str (block model column name).")

    # y origin
    if type(dims[1]) is int or type(dims[1]) is float:
        yorigin = blockmodel[xyz_cols[1]].min() - (dims[1] / 2)
    elif type(dims[1]) is str:
        yorigin = (blockmodel[xyz_cols[1]] - (blockmodel[dims[1]] / 2)).min()
    else:
        raise Exception("y dimension must be int, float or str (block model column name).")

    # z origin
    if type(dims[2]) is int or type(dims[2]) is float:
        zorigin = blockmodel[xyz_cols[2]].min() - (dims[2] / 2)
    elif type(dims[2]) is str:
        zorigin = (blockmodel[xyz_cols[2]] - (blockmodel[dims[2]] / 2)).min()
    else:
        raise Exception("z dimension must be int, float or str (block model column name).")

    return xorigin, yorigin, zorigin


def block_dims(blockmodel: pd.DataFrame,
               xyz_cols: Tuple[str, str, str] = None,
               origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0)) -> Tuple[
    float, float, float]:
    """
    estimate the x, y, z dimensions of blocks
    if the blocks are rotated then they are unrotated first
    then the x, y, z dimensions are estimated

    note that this function just estimates the dimensions of the blocks
    it may not always get the perfectly correct answer
    if there are alot of blocks missing in the grid (i.e. a sparse array of blocks)
    the estimation is less likely to be correct

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees

    Returns
    -------
    tuple of floats (xdim, ydim, zdim)
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # make copy of xyz cols
    mod = blockmodel[list(xyz_cols)].copy()

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # inverse rotations to unrotate grid
    unrotation = ((-1.0 * rotation[0]), (-1.0 * rotation[1]), (-1.0 * rotation[2]))

    if x_rotation != 0 or y_rotation != 0 or z_rotation != 0:
        mod_rotated = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                             origin=origin,
                                             rotation=unrotation,
                                             return_full_model=False,
                                             inplace=False)

        mod[xcol] = mod_rotated[xcol].copy()
        mod[ycol] = mod_rotated[ycol].copy()
        mod[zcol] = mod_rotated[zcol].copy()

    # estimate x dimension
    # select each y-z columm
    group_yz = mod.groupby([ycol, zcol])
    xdim_array = pd.Series([])
    for name, group in group_yz:
        group.sort_values(by=xcol, inplace=True, ignore_index=True)
        group['xtemp'] = group[xcol].shift(1)
        group.fillna(0.0, inplace=True)
        group['xdim'] = group[xcol] - group['xtemp']
        mask = group['xdim'] > 0
        xdim = group.loc[mask, 'xdim'].copy()  # series
        # take the mode (most common dimension)
        xdim_array = xdim_array.append(xdim.mode(), ignore_index=True)

    # take the mode (most common dimension) from all groups
    xdim = xdim_array.mode().max()

    # estimate y dimension
    # select each x-z columm
    group_xz = mod.groupby([xcol, zcol])
    ydim_array = pd.Series([])
    for name, group in group_xz:
        group.sort_values(by=ycol, inplace=True, ignore_index=True)
        group['ytemp'] = group[ycol].shift(1)
        group.fillna(0.0, inplace=True)
        group['ydim'] = group[ycol] - group['ytemp']
        mask = group['ydim'] > 0
        ydim = group.loc[mask, 'ydim'].copy()  # series
        # take the mode (most common dimension)
        ydim_array = ydim_array.append(ydim.mode(), ignore_index=True)

    # take the mode (most common dimension) from all groups
    ydim = ydim_array.mode().max()

    # estimate z dimension
    # select each x-y columm
    group_xy = mod.groupby([xcol, ycol])
    zdim_array = pd.Series([])
    for name, group in group_xy:
        group.sort_values(by=zcol, inplace=True, ignore_index=True)
        group['ztemp'] = group[zcol].shift(1)
        group.fillna(0.0, inplace=True)
        group['zdim'] = group[zcol] - group['ztemp']
        mask = group['zdim'] > 0
        zdim = group.loc[mask, 'zdim'].copy()  # series
        # take the mode (most common dimension)
        zdim_array = zdim_array.append(zdim.mode(), ignore_index=True)

    # take the mode (most common dimension) from all groups
    zdim = zdim_array.mode().max()

    return xdim, ydim, zdim


def check_regular(blockmodel: pd.DataFrame,
                  xyz_cols: Tuple[str, str, str] = None,
                  origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                  dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                  rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                  tolerance: Union[int, float] = 0.00001) -> bool:
    """
    check if the blocks in a block model are actually
    on a regular grid (including a rotated grid).
    note this is just an estimatimation of regularity, it is not perfect

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    tolerance: float or int, default 0.00001
        the difference of a blocks centroid from the point on a grid it should be located
        generally in the range of 0.1 to 0.000001

    Returns
    -------
    bool
        whether block model is regular or not. True if regular.
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # make copy of xyz cols
    mod = blockmodel[list(xyz_cols)].copy()

    # inverse rotations to unrotate grid
    unrotation = ((-1.0 * rotation[0]), (-1.0 * rotation[1]), (-1.0 * rotation[2]))

    if x_rotation != 0 or y_rotation != 0 or z_rotation != 0:
        mod_rotated = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                             origin=origin,
                                             rotation=unrotation,
                                             return_full_model=False,
                                             inplace=False)

        mod[xcol] = mod_rotated[xcol].copy()
        mod[ycol] = mod_rotated[ycol].copy()
        mod[zcol] = mod_rotated[zcol].copy()

    # check integer value isn't far from float - this can cause indexing issues
    # throw a warning to the user if this is the case
    # float should be within 0.00001 tolerance of integer
    indexed_float = ((mod[xcol] - xsize / 2 - xorigin) / xsize)
    indexed_int = np.rint((mod[xcol] - xsize / 2 - xorigin) / xsize).astype(int)
    check_float = (indexed_float - indexed_int).abs()
    if check_float.any() > tolerance:
        return False

    # check integer value isn't far from float - this can cause indexing issues
    # throw a warning to the user if this is the case
    # float should be within 0.00001 tolerance of integer
    indexed_float = ((mod[ycol] - ysize / 2 - yorigin) / ysize)
    indexed_int = np.rint((mod[ycol] - ysize / 2 - yorigin) / ysize).astype(int)
    check_float = (indexed_float - indexed_int).abs()
    if check_float.any() > tolerance:
        return False

    # check integer value isn't far from float - this can cause indexing issues
    # throw a warning to the user if this is the case
    # float should be within 0.00001 tolerance of integer
    indexed_float = ((mod[zcol] - zsize / 2 - zorigin) / zsize)
    indexed_int = np.rint((mod[zcol] - zsize / 2 - zorigin) / zsize).astype(int)
    check_float = (indexed_float - indexed_int).abs()
    if check_float.any() > tolerance:
        return False

    return True


def check_internal_blocks_missing(blockmodel: pd.DataFrame,
                                  xyz_cols: Tuple[str, str, str] = None,
                                  dims: Tuple[
                                      Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                                  rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                                  origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                                  ) -> bool:
    """
    check if there are missing internal blocks (not side blocks) within a regular block model

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    origin: tuple of floats or ints, default (0,0,0)
        ONLY NEEDED IF MODEL IS ROTATED
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)

    Returns
    -------
    bool
        whether block model contains missing internal blocks
        returns True if so
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # make copy of xyz cols
    mod = blockmodel[list(xyz_cols)].copy()

    # inverse rotations to unrotate grid
    unrotation = ((-1.0 * rotation[0]), (-1.0 * rotation[1]), (-1.0 * rotation[2]))

    if x_rotation != 0 or y_rotation != 0 or z_rotation != 0:
        mod_rotated = blockmodel.rotate_grid(xyz_cols=xyz_cols,
                                             origin=origin,
                                             rotation=unrotation,
                                             return_full_model=False,
                                             inplace=False)

        mod[xcol] = mod_rotated[xcol].copy()
        mod[ycol] = mod_rotated[ycol].copy()
        mod[zcol] = mod_rotated[zcol].copy()

    # select each x-y columm
    group_xy = mod.groupby([xcol, ycol])
    for name, group in group_xy:
        group.sort_values(by=zcol, inplace=True, ignore_index=True)
        group['ztemp'] = group[zcol].shift(1)
        group.fillna(0.0, inplace=True)
        group['zstep'] = group[zcol] - group['ztemp']
        group.at[0, 'zstep'] = zsize
        mask = group['zstep'] == zsize
        if mask.sum() != len(group):
            return True

    return False


def index_3Dto1D(blockmodel: pd.DataFrame,
                 indexing: int = 0,
                 xyz_cols: Tuple[str, str, str] = None,
                 origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                 dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                 rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                 nblocks_xyz: Tuple[int, int, int] = None,
                 idxcol: str = 'ijk',
                 inplace: bool = False) -> pd.DataFrame:
    """
    Convert 3D array of xyz block centroids to 1D index that is reversible.
    Opposite of the function index_1Dto3D()

    This is identical to the "ijk" parameter in Datamine block models.
    Note that "ijk" value from this function and from Datamine may be different,
    depending on which axis Datamine uses as the major indexing axis.
    Bot "ijk" indexing values are still valid.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    indexing: int, default 0
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    nblocks_xyz: tuple of ints or None
        number of blocks along the x,y,z axis.
        If the model is rotated, it is unrotated and then the number
        of blocks in the x,y,z axis is calculated.
        If "None" (default value) then the nx,ny,nz is automatically estimated
    idxcol: str, default 'ijk'
        name of the 1D index column added to the model
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        block model with 1D indexed column
    """

    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    # check input indexing
    indexing_accepted = [0, 1]
    if indexing in indexing_accepted:
        pass
    else:
        raise ValueError('IJK FAILED - indexing value not accepted - only 1 or 0 can be used')

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # take subset of xyz columns
    subset = blockmodel[list(xyz_cols)].copy()

    # automatically calulcate nx, ny, nz if needed
    if nblocks_xyz is None:
        nblocks_xyz = subset.nblocks_xyz(xyz_cols=xyz_cols,
                                         dims=dims,
                                         origin=origin,
                                         rotation=rotation)

    # ijk calculation
    subset.ijk(indexing=indexing,
               xyz_cols=xyz_cols,
               origin=origin,
               dims=dims,
               rotation=rotation,
               inplace=True)

    # definitions for simplicity
    nx = nblocks_xyz[0]
    ny = nblocks_xyz[1]

    # ijk calculation
    subset[idxcol] = subset['i'] + (subset['j'] * nx) + (subset['k'] * nx * ny)

    # add ijk column back to original block model
    blockmodel.insert(0, idxcol, subset[idxcol])

    # check inplace for return
    if not inplace:
        return blockmodel


def index_1Dto3D(blockmodel: pd.DataFrame,
                 indexing: int = 0,
                 idxcol: str = 'ijk',
                 origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                 dims: Tuple[Union[int, float, str], Union[int, float, str], Union[int, float, str]] = None,
                 rotation: Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                 nblocks_xyz: Tuple[int, int, int] = None,
                 xyz_cols: Tuple[str, str, str] = ('x', 'y', 'z'),
                 inplace: bool = False) -> pd.DataFrame:
    """
    Convert IJK index back to xyz block centroids.
    Opposite of the function index_3Dto1D()

    This is identical to the "ijk" parameter in Datamine block models.
    Note that "ijk" value from this function and from Datamine may be different,
    depending on which axis Datamine uses as the major indexing axis.
    Bot "ijk" indexing values are still valid.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    indexing: int, default 0
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    idxcol: str, default 'ijk'
        name of the 1D index column added to the model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats, ints or str
        x,y,z dimension of regular parent blocks
        can either be a number or the columns names of the x,y,z
        columns in the dataframe
    rotation: tuple of floats or ints, default (0,0,0)
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    nblocks_xyz: tuple of ints or None
        number of blocks along the x,y,z axis.
        If the model is rotated, it is unrotated and then the number
        of blocks in the x,y,z axis is calculated.
    xyz_cols: tuple of strings, default ('x', 'y', 'z')
        names of x,y,z columns in model
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        block model with 1D indexed column
    """

    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    # check input indexing
    indexing_accepted = [0, 1]
    if indexing in indexing_accepted:
        pass
    else:
        raise ValueError('IJK FAILED - indexing value not accepted - only 1 or 0 can be used')

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # copy ijk column
    subset = pd.DataFrame(blockmodel[idxcol].copy())

    # definitions for simplicity
    nx = nblocks_xyz[0]
    ny = nblocks_xyz[1]
    nz = nblocks_xyz[2]

    # back calculate i, j, k values
    subset['i'] = (subset[idxcol] % nx).astype(int)
    subset['j'] = ((subset[idxcol] / nx) % ny).astype(int)
    subset['k'] = (((subset[idxcol] / nx) / ny) % nz).astype(int)

    # back calculate x, y, z values
    subset.xyz(indexing=indexing,
               origin=origin,
               dims=dims,
               rotation=rotation,
               xyz_cols=xyz_cols,
               inplace=True)

    # add xyz columns back to original block model
    blockmodel.insert(0, xyz_cols[0], subset[xyz_cols[0]])
    blockmodel.insert(1, xyz_cols[1], subset[xyz_cols[1]])
    blockmodel.insert(2, xyz_cols[2], subset[xyz_cols[2]])

    # check inplace for return
    if not inplace:
        return blockmodel


def extend_pandas():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.ijk = ijk
    PandasObject.xyz = xyz
    PandasObject.rotate_grid = rotate_grid
    PandasObject.group_weighted_average = group_weighted_average
    PandasObject.nblocks_xyz = nblocks_xyz
    PandasObject.vulcan_csv = vulcan_csv
    PandasObject.vulcan_bdf = vulcan_bdf
    PandasObject.geometric_reblock = geometric_reblock
    PandasObject.model_origin = model_origin
    PandasObject.check_regular = check_regular
    PandasObject.attribute_reblock = attribute_reblock
    PandasObject.check_internal_blocks_missing = check_internal_blocks_missing
    PandasObject.block_dims = block_dims
    PandasObject.index_3Dto1D = index_3Dto1D
    PandasObject.index_1Dto3D = index_1Dto3D
    PandasObject.check_regular_extents = check_regular_extents
