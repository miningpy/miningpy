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


def ijk(blockmodel:     pd.DataFrame,
        method:         str = 'ijk',
        indexing:       int = 0,
        xyz_cols:       Tuple[str, str, str] = None,
        origin:         Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        dims:           Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        rotation:       Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
        ijk_cols:       Tuple[str, str, str] = ('i', 'j', 'k'),
        inplace:        bool = False) -> pd.DataFrame:
    """
    Calculate block ijk indexes from their xyz cartesian coordinates

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    method: str
        can be used to only calculate i, or j, or k
    indexing: int
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    ijk_cols: tuple of strings
        name of the i,j,k columns added to the model
    inplace: bool
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

    if x_rotation == 0:  # x rotation
        bm_xcol = blockmodel[xcol]
    else:
        bm_xcol = blockmodel.rotate_grid(xcol=xcol,
                                         ycol=ycol,
                                         zcol=zcol,
                                         xorigin=xorigin,
                                         yorigin=yorigin,
                                         zorigin=zorigin,
                                         x_rotation=x_rotation,
                                         y_rotation=y_rotation,
                                         z_rotation=z_rotation,
                                         return_full_model=False,
                                         inplace=True)['x']
    if y_rotation == 0:
        bm_ycol = blockmodel[ycol]
    else:
        bm_ycol = blockmodel.rotate_grid(xcol=xcol,
                                         ycol=ycol,
                                         zcol=zcol,
                                         xorigin=xorigin,
                                         yorigin=yorigin,
                                         zorigin=zorigin,
                                         x_rotation=x_rotation,
                                         y_rotation=y_rotation,
                                         z_rotation=z_rotation,
                                         return_full_model=False,
                                         inplace=True)['y']
    if z_rotation == 0:
        bm_zcol = blockmodel[zcol]
    else:
        bm_zcol = blockmodel.rotate_grid(xcol=xcol,
                                         ycol=ycol,
                                         zcol=zcol,
                                         xorigin=xorigin,
                                         yorigin=yorigin,
                                         zorigin=zorigin,
                                         x_rotation=x_rotation,
                                         y_rotation=y_rotation,
                                         z_rotation=z_rotation,
                                         return_full_model=False,
                                         inplace=True)['z']
    
    if method in methods_accepted:
        if 'i' in method:
            try:
                blockmodel[icol] = (np.rint((bm_xcol - xsize/2 - xorigin) / xsize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either xcol, xorigin or xsize not defined properly')

        if 'j' in method:
            try:
                blockmodel[jcol] = (np.rint((bm_ycol - ysize/2 - yorigin) / ysize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either ycol, yorigin or ysize not defined properly')

        if 'k' in method:
            try:
                blockmodel[kcol] = (np.rint((bm_zcol - zsize / 2 - zorigin) / zsize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either zcol, zorigin or zsize not defined properly')

        # check inplace for return
        if not inplace:
            return blockmodel
    else:
        raise ValueError('IJK FAILED - IJK method not accepted')


def xyz(blockmodel:     pd.DataFrame,
        method:         str = 'xyz',
        indexing:       int = 0,
        ijk_cols:       Tuple[str, str, str] = ('i', 'j', 'k'),
        origin:         Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        dims:           Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
        rotation:       Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
        xyz_cols:       Tuple[str, str, str] = ('x', 'y', 'z'),
        inplace:        bool = False) -> pd.DataFrame:
    """
    Calculate xyz cartesian cooridinates of blocks from their ijk indexes

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    method: str
        can be used to only calculate i, or j, or k
    indexing: int
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    ijk_cols: tuple of strings
        name of the i,j,k columns added to the model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    inplace: bool
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

    methods_accepted = ['xyz', 'xy', 'xz', 'yz', 'x',  'y', 'z']
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

    # change sign of rotation - reversing rotation applied to ijk calculation
    x_rotation = 0 - x_rotation
    y_rotation = 0 - y_rotation
    z_rotation = 0 - z_rotation

    if method in methods_accepted:
        if 'x' in method:
            try:
                bm_xcol = ((blockmodel[icol] - indexing) * xsize) + xorigin + (xsize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either icol, xorigin or xsize not defined properly')

        if 'y' in method:
            try:
                bm_ycol = ((blockmodel[jcol] - indexing) * ysize) + yorigin + (ysize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either jcol, yorigin or ysize not defined properly')

        if 'z' in method:
            try:
                bm_zcol = ((blockmodel[kcol] - indexing) * zsize) + zorigin + (zsize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either kcol, zorigin or zsize not defined properly')

    else:
        raise ValueError('XYZ FAILED - XYZ method not accepted')

    if x_rotation == 0:
        blockmodel[xcol] = bm_xcol
    else:
        blockmodel[xcol] = blockmodel.rotate_grid(xcol=xcol,
                                                  ycol=ycol,
                                                  zcol=zcol,
                                                  xorigin=xorigin,
                                                  yorigin=yorigin,
                                                  zorigin=zorigin,
                                                  x_rotation=x_rotation,
                                                  y_rotation=y_rotation,
                                                  z_rotation=z_rotation,
                                                  return_full_model=False,
                                                  inplace=True)['x']
    if y_rotation == 0:
        blockmodel[ycol] = bm_ycol
    else:
        blockmodel[ycol] = blockmodel.rotate_grid(xcol=xcol,
                                                  ycol=ycol,
                                                  zcol=zcol,
                                                  xorigin=xorigin,
                                                  yorigin=yorigin,
                                                  zorigin=zorigin,
                                                  x_rotation=x_rotation,
                                                  y_rotation=y_rotation,
                                                  z_rotation=z_rotation,
                                                  return_full_model=False,
                                                  inplace=True)['y']
    if z_rotation == 0:
        blockmodel[zcol] = bm_zcol
    else:
        blockmodel[zcol] = blockmodel.rotate_grid(xcol=xcol,
                                                  ycol=ycol,
                                                  zcol=zcol,
                                                  xorigin=xorigin,
                                                  yorigin=yorigin,
                                                  zorigin=zorigin,
                                                  x_rotation=x_rotation,
                                                  y_rotation=y_rotation,
                                                  z_rotation=z_rotation,
                                                  return_full_model=False,
                                                  inplace=True)['z']
    
    # check inplace for return
    if not inplace:
        return blockmodel


def rotate_grid(blockmodel:         pd.DataFrame,
                xyz_cols:           Tuple[str, str, str] = ('x', 'y', 'z'),
                origin:             Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                rotation:           Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                return_full_model:  bool = True,
                inplace:            bool = False) -> Union[pd.DataFrame, dict]:
    """
    Rotate block model relative to cartesian grid
    This method uses a rotation matrix method
    Rotation is done using the right hand rule

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    return_full_model: bool
        whether to return the full block model or just a dict of the rotated x,y,z coordinates
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        rotated block model or dict of rotated x,y,z coordinates
    """
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

    x_sin = sin(x_rotation*(pi/180.0))
    x_cos = cos(x_rotation*(pi/180.0))
    y_sin = sin(y_rotation*(pi/180.0))
    y_cos = cos(y_rotation*(pi/180.0))
    z_sin = sin(z_rotation*(pi/180.0))
    z_cos = cos(z_rotation*(pi/180.0))

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


def group_weighted_average(blockmodel:   pd.DataFrame,
                           avg_cols:     Union[str, List[str]],
                           weight_col:   str,
                           group_cols:    Union[str, List[str]],
                           inplace:      bool = False) -> pd.DataFrame:
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
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        block model
    """

    # check inplace
    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    average_cols = []
    groupby_cols = []

    if isinstance(avg_cols, str):
        average_cols.append(avg_cols)
    elif isinstance(avg_cols, list):
        average_cols = avg_cols
    else:
        raise Exception('Average columns parameter must be single column name or list of column names')

    if isinstance(group_cols, str):
        groupby_cols.append(group_cols)
    elif isinstance(group_cols, list):
        groupby_cols = group_cols
    else:
        raise Exception('Groupby columns parameter must be single column name or list of column names')

    dfs = []
    for count, col in enumerate(average_cols, 1):
        vars()[str(count) + '_' + col] = (blockmodel[col] * blockmodel[weight_col]).groupby(groupby_cols).sum() / blockmodel[weight_col].groupby(groupby_cols).sum()
        dfs.append((str(count) + '_' + col))

    blockmodel = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]

    # check inplace for return
    if not inplace:
        return blockmodel


def vulcan_csv(blockmodel: pd.DataFrame,
               path: str = None,
               xyz_cols: Tuple[str, str, str] = None,
               dims: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               inplace: bool = False) -> None:
    """
    transform pandas.Dataframe block model into Vulcan import compatible CSV format

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for vulcan csv block model file
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        block model in Vulcan import CSV format
    """

    # check inplace
    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    # create columns required for vulcan
    blockmodel['dim_x'] = xsize
    blockmodel['dim_y'] = ysize
    blockmodel['dim_z'] = zsize
    blockmodel['volume'] = xsize * ysize * zsize

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

    for pandas,vulcan in pandas_vulcan_dtypes.items():
        mask = vulcan_header['Data Type'] == pandas
        vulcan_header.loc[mask, 'Data Type'] = vulcan

    vulcan_variables.to_csv('vulcan_variables.csv', index=False)

    # rename column headers to vulcan format
    blockmodel.rename(columns={'x': 'centroid_x',
                                    'y': 'centroid_y',
                                    'z': 'centroid_z',
                                    'xdim': 'dim_x',
                                    'ydim': 'dim_y',
                                    'zdim': 'dim_z'}, inplace=True)

    # create vulcan header df to append to start of block model

    vulcan_header['Data Type'] = 'double'
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

    vulcan_model = pd.concat([vulcan_header, blockmodel], axis=0, sort=False, ignore_index=True)

    print("vulcan block model header:")
    print(tabulate(vulcan_model.head(5), headers="keys", tablefmt='fancy_grid'))

    date_object = datetime.date.today()
    print(date_object)

    if csv:
        if csv_name != None:
            blockmodel.to_csv(csv_name, index=False)
        else:
            blockmodel.to_csv(f'vulcan_output_{str(date_object)}.csv', index=False)
            print(f'written to vulcan_output_{str(date_object)}.csv')

    if inplace:
        return
    if not inplace:
        return blockmodel


def vulcan_bdf(blockmodel: pd.DataFrame,
               path: str = None,
               origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               dims: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               start_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               end_offset: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               format: str = 'T') -> None:
    """
    create a Vulcan block definition file from a vulcan block model

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for vulcan bdf file
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    start_offset: tuple of floats or ints
        minimum offset along the x,y,z axes
    end_offset: tuple of floats or ints
        maximum offset along the x,y,z axes
    format: str
        block model file format (classic = C, extended = T)

    Returns
    -------
    vulcan bdf file
    """

    # Caveats
    print("\n***\n\n"
          "This script creates a BDF from a vulcan block model csv that can be imported into Vulcan.\nIt assumes that bearing, dip and plunge are the default"
          " values for the block model.\nVariables are given a default value of -99.0, a blank description and type 'double'.\n"
          "This script will define the parent schema.\n"
          "All of these values can be edited within Vulcan once the script has been run and bdf "
          "imported.\n\n***\n ")

    # input checks for format
    assert format == 'T' or format == 'C', "Define format. format='C' for classic, format='T' for extended"

    # listing variables in vulcan block model (removing vulcan headers)
    variables = list(blockmodel.columns)
    del variables[0:7]

    # writing bdf file header
    bdf = open("vulcan_bdf.txt", "w+")
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
        bdf.write(f"name='{variables[count-1]}'\n")
        bdf.write(" ")
        bdf.write("type='double'\n")
        bdf.write(f"END$DEF variable_{count}\n")

        count+=1

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

    # Writing BDF file
    date_object = datetime.date.today()

    if bdf_name != None:
        vulcan_bdf, txt = os.path.splitext("vulcan_bdf.txt")
        os.rename("vulcan_bdf.txt", f"{bdf_name}" + ".bdf")
        print(f"written bdf to '{bdf_name}.bdf'")
    else:
        vulcan_bdf, txt = os.path.splitext("vulcan_bdf.txt")
        os.rename("vulcan_bdf.txt", f"vulcan_bdf_{date_object}" + ".bdf")
        print(f'written bdf to vulcan_bdf_{str(date_object)}.bdf')

    return


def geometric_reblock(blockmodel: pd.DataFrame):
    """
    reblock regular block model into larger or smaller blocks (split or aggregate blocks)
    can be used as a tool for geometrically aggregating blocks in bench-phases
    reblock factor (n) must be 2^n (i.e. blocks are either doubled, halved, quartered, etc in size)
    cannot just define any new x,y,z dimension, must be multiple of current parent block size

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    pandas.DataFrame
        reblocked block model
    """
    pass


def whittle_mod(blockmodel: pd.DataFrame):
    """
    create a Whittle 4D .MOD file from a block model

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    Whittle .MOD text file
    """
    pass


def whittle_par():
    return


def model_rotation(blockmodel: pd.DataFrame,
                   xyz_cols: Tuple[str, str, str] = None,
                   dims: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                   origin: Tuple[Union[int, float], Union[int, float], Union[int, float]] = None) -> Tuple[float, float, float]:
    """
    calculate the rotation of a block model grid relative to its current xyz grid
    rotation is calculated using the right hand rule

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)

    Returns
    -------
    tuple of floats
        block model around each axis (x,y,z)
    """
    pass


def model_origin(blockmodel: pd.DataFrame,
                 xyz_cols: Tuple[str, str, str] = None) -> Tuple[float, float, float]:
    """
    calculate the origin of a block model grid relative to its current xyz grid
    origin is the corner of the block with min xyz coordinates

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    xyz_cols: tuple of strings
        names of x,y,z columns in model

    Returns
    -------
    tuple of floats
        origin of a block model for each axis (x,y,z)
    """
    xorigin = blockmodel[xcol].min()
    yorigin = blockmodel[ycol].min()
    zorigin = blockmodel[zcol].min()

    return xorigin, yorigin, zorigin


def model_block_size(blockmodel:     pd.DataFrame,
                     xcol:           str = None,
                     ycol:           str = None,
                     zcol:           str = None,
                     xorigin:        Union[int, float] = None,
                     yorigin:        Union[int, float] = None,
                     zorigin:        Union[int, float] = None,
                     x_rotation:     Union[int, float] = 0,
                     y_rotation:     Union[int, float] = 0,
                     z_rotation:     Union[int, float] = 0,
                     inplace:        bool = False) -> Tuple[float, float, float]:
    return


def check_regular(blockmodel: pd.DataFrame) -> None:
    """
    check if the blocks in a block model are actually on a regular grid (including a rotated grid)

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    bool
        whether block model is regular or not. True if regular.
    """
    return


def check_internal_blocks_missing(blockmodel: pd.DataFrame):
    """
    check if there are missing internal blocks (not side blocks) within a regular block model

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model

    Returns
    -------
    bool
        whether block model contains missing internal blocks
    """
    return


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
    return


def index_3D_to_1D(blockmodel:  pd.DataFrame,
                   indexing:    int = 0,
                   xyz_cols:    Tuple[str, str, str] = None,
                   origin:      Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                   dims:        Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                   rotation:    Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                   idxcol:      str = 'idx',
                   inplace:     bool = False) -> pd.DataFrame:
    """
    Convert 3D array of xyz block centroids to 1D index that is reversible.
    This is identical to the "ijk" parameter in Datamine block models.

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    indexing: int
        controls whether origin block has coordinates 0,0,0 or 1,1,1
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    origin: tuple of floats or ints
        x,y,z origin of model - this is the corner of the bottom block (not the centroid)
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    idxcol: str
        name of the 1D index column added to the model
    inplace: bool
        whether to do calculation inplace on pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        1D indexed block model
    """

    if inplace:
        blockmodel = blockmodel
    if not inplace:
        blockmodel = blockmodel.copy()

    indexing_accepted = [0, 1]

    # check input indexing
    if indexing in indexing_accepted:
        pass
    else:
        raise ValueError('IJK FAILED - indexing value not accepted - only 1 or 0 can be used')

    if x_rotation == 0 and y_rotation == 0 and z_rotation == 0:
        pass
    else:
        blockmodel.rotate_grid(
            xcol=xcol,
            ycol=ycol,
            zcol=zcol,
            xorigin=xorigin,
            yorigin=yorigin,
            zorigin=zorigin,
            x_rotation=x_rotation,
            y_rotation=y_rotation,
            z_rotation=z_rotation,
            inplace=True)

    # check inplace for return
    if inplace:
        return
    if not inplace:
        return blockmodel

    return


def index_1D_to_3D():
    return


def table_to_4D_array():
    return


def table_to_1D_array():
    return


def extend_pandas():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.ijk = ijk
    PandasObject.xyz = xyz
    PandasObject.rotate_grid = rotate_grid
    PandasObject.group_weighted_average = group_weighted_average
    PandasObject.vulcan_csv = vulcan_csv
    PandasObject.vulcan_bdf = vulcan_bdf
    PandasObject.geometric_reblock = geometric_reblock
    PandasObject.whittle_mod = whittle_mod
    PandasObject.model_rotation = model_rotation
    PandasObject.model_origin = model_origin
    PandasObject.check_regular = check_regular
    PandasObject.attribute_reblock = attribute_reblock
    PandasObject.check_internal_blocks_missing = check_internal_blocks_missing
    PandasObject.index_3D_to_1D = index_3D_to_1D
    PandasObject.index_1D_to_3D = index_1D_to_3D
    PandasObject.table_to_4D_array = table_to_4D_array
    PandasObject.table_to_1D_array = table_to_1D_array
