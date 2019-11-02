# import libraries
import pandas as pd
import numpy as np
from math import sin, cos, pi
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
import vtk
import vtk.util.numpy_support as vtknumpy
from mining_utils.numpy_vtk import *
import ezdxf


# applies model calculations to pandas dataframe of block model as a pandas method
# extends the pandas API for mining engineering applications
# inspiration of this methodology from: https://github.com/pmorissette/ffn


def ijk(blockmodel:     pd.DataFrame,
        method:         str = 'ijk',
        indexing:       int = 1,
        xcol:           str = None,
        ycol:           str = None,
        zcol:           str = None,
        xorigin:        Union[int, float] = None,
        yorigin:        Union[int, float] = None,
        zorigin:        Union[int, float] = None,
        xsize:          Union[int, float] = None,
        ysize:          Union[int, float] = None,
        zsize:          Union[int, float] = None,
        x_rotation:     Union[int, float] = 0,
        y_rotation:     Union[int, float] = 0,
        z_rotation:     Union[int, float] = 0,
        icol:           str = 'i',
        jcol:           str = 'j',
        kcol:           str = 'k',
        inplace:        bool = False) -> pd.DataFrame:
    """
    Calculate block ijk indexes from their xyz cartesian coordinates
    :param blockmodel: pandas dataframe of block model
    :param method: can be used to only calculate i, or j, or k
    :param indexing: controls whether origin block has coordinates 0,0,0 or 1,1,1
    :param xcol: x column in model
    :param ycol: y column in model
    :param zcol: z column in model
    :param xorigin: x origin of model - this is the corner of the bottom block (not the centroid)
    :param yorigin: y origin of model - this is the corner of the bottom block (not the centroid)
    :param zorigin: z origin of model - this is the corner of the bottom block (not the centroid)
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of block model grid around x axis, -180 to 180 degrees
    :param y_rotation: rotation of block model grid around y axis, -180 to 180 degrees
    :param z_rotation: rotation of block model grid around z axis (xy plane), -180 to 180 degrees
    :param icol: name of the i column added to the model
    :param jcol: name of the j column added to the model
    :param kcol: name of the k column added to the model
    :param inplace: whether to do calculation inplace on pandas.DataFrame

    :return pandas.DataFrame of indexed block model
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

    rotations = [x_rotation, y_rotation, z_rotation]

    # check rotation is within parameters
    for rotation in rotations:
        if -180 <= rotation <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

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

    if method in methods_accepted:
        if 'i' in method:
            try:
                blockmodel[icol] = (np.rint((blockmodel[xcol] - xsize/2 - xorigin) / xsize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either xcol, xorigin or xsize not defined properly')

        if 'j' in method:
            try:
                blockmodel[jcol] = (np.rint((blockmodel[ycol] - ysize/2 - yorigin) / ysize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either ycol, yorigin or ysize not defined properly')

        if 'k' in method:
            try:
                blockmodel[kcol] = (np.rint((blockmodel[zcol] - zsize / 2 - zorigin) / zsize) + indexing).astype(int)
            except ValueError:
                raise ValueError('IJK FAILED - either zcol, zorigin or zsize not defined properly')

        # check inplace for return
        if inplace:
            return
        if not inplace:
            return blockmodel
    else:
        raise ValueError('IJK FAILED - IJK method not accepted')


def xyz(blockmodel:     pd.DataFrame,
        method:         str = 'xyz',
        indexing:       int = 1,
        icol:           str = 'i',
        jcol:           str = 'j',
        kcol:           str = 'k',
        xorigin:        Union[int, float] = None,
        yorigin:        Union[int, float] = None,
        zorigin:        Union[int, float] = None,
        xsize:          Union[int, float] = None,
        ysize:          Union[int, float] = None,
        zsize:          Union[int, float] = None,
        x_rotation:     Union[int, float] = 0,
        y_rotation:     Union[int, float] = 0,
        z_rotation:     Union[int, float] = 0,
        xcol:           str = 'xc',
        ycol:           str = 'yc',
        zcol:           str = 'zc',
        inplace:        bool = False) -> pd.DataFrame:
    """
    Calculate xyz cartesian cooridinates of blocks from their ijk indexes
    :param blockmodel: pandas dataframe of block model
    :param method: can be used to only calculate x, y or z
    :param indexing: controls whether origin block has coordinates 0,0,0 or 1,1,1
    :param icol: i column in model
    :param jcol: j column in model
    :param kcol: k column in model
    :param xorigin: x origin of model - this is the corner of the bottom block (not the centroid)
    :param yorigin: y origin of model - this is the corner of the bottom block (not the centroid)
    :param zorigin: z origin of model - this is the corner of the bottom block (not the centroid)
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of block model grid around x axis, -180 to 180 degrees
    :param y_rotation: rotation of block model grid around y axis, -180 to 180 degrees
    :param z_rotation: rotation of block model grid around z axis (xy plane), -180 to 180 degrees
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param inplace: whether to do calculation inplace on pandas.DataFrame

    :return pandas.DataFrame of rotated block model
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

    rotations = [x_rotation, y_rotation, z_rotation]

    # check rotation is within parameters
    for rotation in rotations:
        if -180 <= rotation <= 180:
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
                blockmodel[xcol] = ((blockmodel[icol] - indexing) * xsize) + xorigin + (xsize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either icol, xorigin or xsize not defined properly')

        if 'y' in method:
            try:
                blockmodel[ycol] = ((blockmodel[jcol] - indexing) * ysize) + yorigin + (ysize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either jcol, yorigin or ysize not defined properly')

        if 'z' in method:
            try:
                blockmodel[zcol] = ((blockmodel[kcol] - indexing) * zsize) + zorigin + (zsize/2)
            except ValueError:
                raise ValueError('XYZ FAILED - either kcol, zorigin or zsize not defined properly')

    else:
        raise ValueError('XYZ FAILED - XYZ method not accepted')

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


def rotate_grid(blockmodel:   pd.DataFrame,
                xcol:         str = None,
                ycol:         str = None,
                zcol:         str = None,
                xorigin:      Union[int, float] = None,
                yorigin:      Union[int, float] = None,
                zorigin:      Union[int, float] = None,
                x_rotation:   Union[int, float] = 0,
                y_rotation:   Union[int, float] = 0,
                z_rotation:   Union[int, float] = 0,
                inplace:      bool = False) -> pd.DataFrame:
    """
    Rotate block model relative to cartesian grid
    This method uses a rotation matrix method
    Rotation is done using the right hand rule
    :param blockmodel: pandas dataframe of block model
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xorigin: x origin of model - this is the corner of the bottom block (not the centroid)
    :param yorigin: y origin of model - this is the corner of the bottom block (not the centroid)
    :param zorigin: z origin of model - this is the corner of the bottom block (not the centroid)
    :param x_rotation: rotation of block model grid around x axis, -180 to 180 degrees
    :param y_rotation: rotation of block model grid around y axis, -180 to 180 degrees
    :param z_rotation: rotation of block model grid around z axis (xy plane), -180 to 180 degrees
    :param inplace: whether to do calculation inplace on pandas.DataFrame

    :return pandas.DataFrame of rotated block model
    """

    rotations = [x_rotation, y_rotation, z_rotation]

    # check rotation is within parameters
    for rotation in rotations:
        if -180 <= rotation <= 180:
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
    if inplace:
        return
    if not inplace:
        return blockmodel


def group_weighted_average(blockmodel:   pd.DataFrame,
                           avg_cols:     Union[str, List[str]],
                           weight_col:   str,
                           group_cols:    Union[str, List[str]],
                           inplace:      bool = False) -> pd.DataFrame:
    """
    weighted average of block model attribute(s)
    :param blockmodel: pandas dataframe of block model
    :param avg_cols: column(s) to take the weighted average
    :param weight_col: column to weight on. Example the tonnes column
    :param group_cols: the columns you want to group on. Either single column or list of columns
    :param inplace: whether to do calculation inplace on pandas.DataFrame
    :return pandas.DataFrame of block model
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
    if inplace:
        return
    if not inplace:
        return blockmodel


def vulcan_csv(blockmodel: pd.DataFrame,
               path: str = None,
               xcol: str = None,
               ycol: str = None,
               zcol: str = None,
               xsize: Union[int, float] = None,
               ysize: Union[int, float] = None,
               zsize: Union[int, float] = None,
               inplace: bool = False) -> None:
    """
    transform pandas.Dataframe block model into Vulcan import CSV format
    :param blockmodel: pandas.Dataframe of block model
    :param path: filename for vulca csv block model file
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param inplace: whether to do calculation on dataframe in place

    :return pandas.DataFrame of block model in Vulcan import CSV format
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
    vulcan_fields = [xcol, ycol, zcol, xsize, ysize, zsize, 'volume']
    blockmodel.set_index(vulcan_fields, inplace=True)
    blockmodel.reset_index(inplace=True)

    # Export model to Vulcan formatted CSV (use this to create the bdf file)
    vulcan_variables = pd.DataFrame({'Variable': list(mod_flat_vulcan.columns.values), 'Data Type': list(mod_flat_vulcan.dtypes)})
    vulcan_variables['Default Value'] = -99.0
    vulcan_header = vulcan_variables.copy()

    pandas_vulcan_dtypes = {'int64': 'Integer (Integer * 4)',
                            'float64': 'Double (Real * 8)',
                            'object': 'Name (Translation Table)'}

    return


def vulcan_bdf(blockmodel: pd.DataFrame):
    """
    create a Vulcan block definition file of a block model
    :param blockmodel: pandas.Dataframe of block model

    :return Vulcan bdf file
    """
    return


def vulcan_bmf(blockmodel: pd.DataFrame):
    """
    create a Vulcan block model file of a block model
    :param blockmodel: pandas.Dataframe of block model

    :return Vulcan bmf file
    """
    pass


def geometric_reblock(blockmodel: pd.DataFrame):
    """
    reblock regular block model into larger or smaller blocks (split or aggregate blocks)
    can be used as a tool for geometrically aggregating blocks in bench-phases
    reblock factor (n) must be 2^n (i.e. blocks are either doubled, halved, quartered, etc in size)
    cannot just define any new x,y,z dimension, must be multiple of current parent block size
    :param blockmodel: pandas.Dataframe of block model

    :return pandas.DataFrame of reblocked block model
    """
    pass


def whittle_mod(blockmodel: pd.DataFrame):
    """
    create a Whittle 4D .MOD file from a block model
    :param blockmodel: pandas.Dataframe of block model

    :return Whittle .MOD text file
    """
    pass


def whittle_par():
    return


def model_rotation(blockmodel: pd.DataFrame,
                   xcol: str = None,
                   ycol: str = None,
                   zcol: str = None,
                   xsize: Union[int, float] = None,
                   ysize: Union[int, float] = None,
                   zsize: Union[int, float] = None,
                   xorigin: Union[int, float] = None,
                   yorigin: Union[int, float] = None,
                   zorigin: Union[int, float] = None) -> Tuple[float, float, float]:
    """
    calculate the rotation of a block model grid relative to its current xyz grid
    rotation is calculated using the right hand rule
    :param blockmodel: pandas.Dataframe of block model
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param xorigin: x origin of model - this is the corner of the bottom block (not the centroid)
    :param yorigin: y origin of model - this is the corner of the bottom block (not the centroid)
    :param zorigin: z origin of model - this is the corner of the bottom block (not the centroid)

    :return tuple of the rotation of a block model around each axis (x,y,z)
    """
    pass


def model_origin(blockmodel: pd.DataFrame,
                   xcol: str = None,
                   ycol: str = None,
                   zcol: str = None,
                   xsize: Union[int, float] = None,
                   ysize: Union[int, float] = None,
                   zsize: Union[int, float] = None,
                   x_rotation: Union[int, float] = None,
                   y_rotation: Union[int, float] = None,
                   z_rotation: Union[int, float] = None) -> Tuple[float, float, float]:
    """
    calculate the origin of a block model grid relative to its current xyz grid
    origin is the corner of the block with min xyz coordinates
    :param blockmodel: pandas.Dataframe of block model
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of blocks around x axis, -180 to 180 degrees
    :param y_rotation: rotation of blocks around y axis, -180 to 180 degrees
    :param z_rotation: rotation of blocks around z axis (xy plane), -180 to 180 degrees
    :return tuple of the origin of a block model for each axis (x,y,z)
    """
    return


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
    :param blockmodel: pandas.Dataframe of block model

    :return boolean of whether block model is regular or not. True if regular.
    """
    return


def check_internal_blocks_missing(blockmodel: pd.DataFrame):
    """
    check if there are missing internal blocks (not side blocks) within a regular block model
    :param blockmodel: pandas.Dataframe of block model

    :return boolean of whether block model contains missing internal blocks
    """
    return


def attribute_reblock(blockmodel: pd.DataFrame):
    """
    bin the attributes & tonnes of a block model for each bench/phase
    :param blockmodel: pandas.Dataframe of block model
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model

    :return pandas.DataFrame of block model with binned attributes & tonnes for each bench
    """
    return


def blocks2vtk(blockmodel: pd.DataFrame,
               path:       str = None,
               xcol:       str = None,
               ycol:       str = None,
               zcol:       str = None,
               xsize:      Union[int, float] = None,
               ysize:      Union[int, float] = None,
               zsize:      Union[int, float] = None,
               x_rotation: Union[int, float] = 0,
               y_rotation: Union[int, float] = 0,
               z_rotation: Union[int, float] = 0,
               cols:       List[str] = None) -> None:
    """
    exports blocks and attributes of block model to a vtk file to visualise in paraview
    :param blockmodel: pandas.Dataframe of block model
    :param path: filename for vtk file
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of blocks around x axis, -180 to 180 degrees
    :param y_rotation: rotation of blocks around y axis, -180 to 180 degrees
    :param z_rotation: rotation of blocks around z axis (xy plane), -180 to 180 degrees
    :param cols: columns of attributes to visualise using vtk. If None then exports all columns

    :return .vtu vtk file
    """

    # check if cols parameter is string or list or None
    # if string (single column), make into single element list
    if cols is None:
        cols = list(blockmodel.columns)
    if cols is not None:
        if isinstance(cols, list):
            pass
        if isinstance(cols, str):
            cols = [cols]

    # check for duplicate blocks and return warning
    dup_check = list(blockmodel.duplicated(subset=[xcol, ycol, zcol]).unique())
    assert True not in dup_check, 'duplicate blocks in dataframe'

    # check rotation is within parameters
    rotations = [x_rotation, y_rotation, z_rotation]
    for rotation in rotations:
        if -180 <= rotation <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # add extension to path name for vtk file
    # 'vtu' because unstructured grid
    if not path.lower().endswith('.vtu'):
        path = path + '.vtu'

    # prepare block model xyz columns as numpy arrays
    x = blockmodel[xcol].values  # numpy 1D array
    y = blockmodel[ycol].values  # numpy 1D array
    z = blockmodel[zcol].values  # numpy 1D array

    # number of blocks
    nc = x.shape[0]

    # prepare block model vector dimensions columns as numpy arrays
    dx = np.ones(nc) * float(xsize)
    dy = np.ones(nc) * float(ysize)
    dz = np.ones(nc) * float(zsize)

    var = []
    varname = []
    for i in cols:
        var.append(blockmodel[i].values)
        varname.append(i)

    # number of blocks
    nc = x.shape[0]

    # number of block variables
    nvar = len(var)

    # number of points (block vertexes)
    npt = nc * 8

    # Create array of the points and ID
    pcoords = vtk.vtkFloatArray()
    pcoords.SetNumberOfComponents(3)
    pcoords.SetNumberOfTuples(npt)

    points = vtk.vtkPoints()
    voxelArray = vtk.vtkCellArray()

    # define rotation matrix - used for rotated models
    x_sin = sin(x_rotation*(pi/180.0))
    x_cos = cos(x_rotation*(pi/180.0))
    y_sin = sin(y_rotation*(pi/180.0))
    y_cos = cos(y_rotation*(pi/180.0))
    z_sin = sin(z_rotation*(pi/180.0))
    z_cos = cos(z_rotation*(pi/180.0))

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

    # create vertex (points)
    id = 0
    # create corners for each block
    for i in range(nc):
        # do rotation first then add to vtk object
        block_corners = np.zeros((8, 3), dtype=np.float64)
        block_corners[0] = [(x[i]+dx[i]/2.0), (y[i]-dy[i]/2.0), (z[i]-dz[i]/2.0)]
        block_corners[1] = [(x[i]-dx[i]/2.0), (y[i]-dy[i]/2.0), (z[i]-dz[i]/2.0)]
        block_corners[2] = [(x[i]+dx[i]/2.0), (y[i]+dy[i]/2.0), (z[i]-dz[i]/2.0)]
        block_corners[3] = [(x[i]-dx[i]/2.0), (y[i]+dy[i]/2.0), (z[i]-dz[i]/2.0)]
        block_corners[4] = [(x[i]+dx[i]/2.0), (y[i]-dy[i]/2.0), (z[i]+dz[i]/2.0)]
        block_corners[5] = [(x[i]-dx[i]/2.0), (y[i]-dy[i]/2.0), (z[i]+dz[i]/2.0)]
        block_corners[6] = [(x[i]+dx[i]/2.0), (y[i]+dy[i]/2.0), (z[i]+dz[i]/2.0)]
        block_corners[7] = [(x[i]-dx[i]/2.0), (y[i]+dy[i]/2.0), (z[i]+dz[i]/2.0)]

        # rotate block points around block centroid
        block_corners[:,0] -= x[i]
        block_corners[:,1] -= y[i]
        block_corners[:,2] -= z[i]

        block_corners = block_corners.transpose()
        block_corners = np.dot(rotation_matrix, block_corners)
        block_corners = block_corners.transpose()

        block_corners[:,0] += x[i]
        block_corners[:,1] += y[i]
        block_corners[:,2] += z[i]

        pcoords.SetTuple3(id+0, block_corners[0,0], block_corners[0,1], block_corners[0,2])
        pcoords.SetTuple3(id+1, block_corners[1,0], block_corners[1,1], block_corners[1,2])
        pcoords.SetTuple3(id+2, block_corners[2,0], block_corners[2,1], block_corners[2,2])
        pcoords.SetTuple3(id+3, block_corners[3,0], block_corners[3,1], block_corners[3,2])
        pcoords.SetTuple3(id+4, block_corners[4,0], block_corners[4,1], block_corners[4,2])
        pcoords.SetTuple3(id+5, block_corners[5,0], block_corners[5,1], block_corners[5,2])
        pcoords.SetTuple3(id+6, block_corners[6,0], block_corners[6,1], block_corners[6,2])
        pcoords.SetTuple3(id+7, block_corners[7,0], block_corners[7,1], block_corners[7,2])
        id += 8

    # add points to the cell
    points.SetData(pcoords)

    # Create the cells for each block
    id=-1
    for i in range (nc):

        # add next cell
        voxelArray.InsertNextCell(8)

        # for each vertex
        for j in range (8):
            id+=1
            voxelArray.InsertCellPoint(id)

    # create the unstructured grid
    grid = vtk.vtkUnstructuredGrid()
    # Assign points and cells
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_VOXEL, voxelArray)

    # asign scalar
    for i in range(nvar):
        if var[i].dtype == np.object:
            vtk_array = vtk.vtkStringArray()
            for idx in var[i]:
                vtk_array.InsertNextValue(idx)
            vtk_array.SetName(varname[i])
        else:
            vtk_array = np_to_vtk(var[i], deep=0)
            vtk_array.SetName(varname[i])
        grid.GetCellData().AddArray(vtk_array)

        '''
        cscalars = vtknumpy.numpy_to_vtk(var[i])
        cscalars.SetName(varname[i])
        grid.GetCellData().AddArray(cscalars)
        '''

    # Clean before saving...
    # this will remove duplicated points
    extractGrid = vtk.vtkExtractUnstructuredGrid()
    extractGrid.SetInputData(grid)
    extractGrid.PointClippingOff()
    extractGrid.ExtentClippingOff()
    extractGrid.CellClippingOn()
    extractGrid.MergingOn()
    extractGrid.SetCellMinimum(0)
    extractGrid.SetCellMaximum(nc)
    extractGrid.Update()

    # save results
    writer = vtk.vtkXMLUnstructuredGridWriter();
    writer.SetFileName(path);
    writer.SetInputData(extractGrid.GetOutput())
    writer.Write()

    return


def blocks2dxf(blockmodel:              pd.DataFrame,
               path:                    str = None,
               dxf_split:               str = None,
               delete_internal_points:  bool = False,
               xcol:                    str = None,
               ycol:                    str = None,
               zcol:                    str = None,
               xsize:                   Union[int, float] = None,
               ysize:                   Union[int, float] = None,
               zsize:                   Union[int, float] = None,
               x_rotation:              Union[int, float] = 0,
               y_rotation:              Union[int, float] = 0,
               z_rotation:              Union[int, float] = 0) -> None:
    """
    exports blocks and attributes of block model to a vtk file to visualise in paraview
    :param blockmodel: pandas.Dataframe of block model
    :param path: filename for dxf files. If multiple dxfs produced, this will be used as the file suffix
    :param dxf_split: column to split dxf files by.
                      for example, could be the year mined column from minemax
                      if None then one dxf is made of every block in blockmodel
    :param delete_internal_points: True/False to delete internal points.
                                   i.e. just have the dxf skin of the blocks = True (only outer shell)
                                   have all the faces of all the blocks = False (inner meshes)
    :param xcol: name of the x centroid column added to the model
    :param ycol: name of the y centroid column added to the model
    :param zcol: name of the z centroid column added to the model
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of block model grid around x axis, -180 to 180 degrees
    :param y_rotation: rotation of block model grid around y axis, -180 to 180 degrees
    :param z_rotation: rotation of block model grid around z axis (xy plane), -180 to 180 degrees

    :return single or multiple .dxf file(s)
    """

    # check for duplicate blocks and return warning
    dup_check = list(blockmodel.duplicated(subset=[xcol, ycol, zcol]).unique())
    assert True not in dup_check, 'duplicate blocks in dataframe'

    # check rotation is within parameters
    rotations = [x_rotation, y_rotation, z_rotation]
    for rotation in rotations:
        if -180 <= rotation <= 180:
            pass
        else:
            raise Exception('Rotation is limited to between -180 and +180 degrees')

    # add extension to path name for vtk file
    # 'vtu' because unstructured grid
    if not path.lower().endswith('.dxf'):
        path = path + '.dxf'

    # prepare block model xyz columns as numpy arrays
    x = blockmodel[xcol].values  # numpy 1D array
    y = blockmodel[ycol].values  # numpy 1D array
    z = blockmodel[zcol].values  # numpy 1D array

    # number of blocks
    nc = x.shape[0]

    # prepare block model vector dimensions columns as numpy arrays
    dx = np.ones(nc) * float(xsize)
    dy = np.ones(nc) * float(ysize)
    dz = np.ones(nc) * float(zsize)

    # define rotation matrix - used for rotated models
    x_sin = sin(x_rotation*(pi/180.0))
    x_cos = cos(x_rotation*(pi/180.0))
    y_sin = sin(y_rotation*(pi/180.0))
    y_cos = cos(y_rotation*(pi/180.0))
    z_sin = sin(z_rotation*(pi/180.0))
    z_cos = cos(z_rotation*(pi/180.0))

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

    # 6 cube faces
    block_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [3, 2, 6, 7],
        [0, 3, 7, 4]]

    # create dxf object to fill
    dwg = ezdxf.new('AC1015')  # mesh requires the DXF 2000 or newer format
    msp = dwg.modelspace()

    # create vertex (points)
    id = 0
    # create corners for each block
    for i in range(nc):
        # do rotation first then add to vtk object
        block_corners = np.zeros((8, 3), dtype=np.float64)
        block_corners[1] = [(x[i] + dx[i] / 2.0), (y[i] - dy[i] / 2.0), (z[i] - dz[i] / 2.0)]
        block_corners[5] = [(x[i] + dx[i] / 2.0), (y[i] - dy[i] / 2.0), (z[i] + dz[i] / 2.0)]
        block_corners[2] = [(x[i] + dx[i] / 2.0), (y[i] + dy[i] / 2.0), (z[i] - dz[i] / 2.0)]
        block_corners[6] = [(x[i] + dx[i] / 2.0), (y[i] + dy[i] / 2.0), (z[i] + dz[i] / 2.0)]
        block_corners[0] = [(x[i] - dx[i] / 2.0), (y[i] - dy[i] / 2.0), (z[i] - dz[i] / 2.0)]
        block_corners[4] = [(x[i] - dx[i] / 2.0), (y[i] - dy[i] / 2.0), (z[i] + dz[i] / 2.0)]
        block_corners[3] = [(x[i] - dx[i] / 2.0), (y[i] + dy[i] / 2.0), (z[i] - dz[i] / 2.0)]
        block_corners[7] = [(x[i] - dx[i] / 2.0), (y[i] + dy[i] / 2.0), (z[i] + dz[i] / 2.0)]

        # rotate block points around block centroid
        block_corners[:,0] -= x[i]
        block_corners[:,1] -= y[i]
        block_corners[:,2] -= z[i]

        block_corners = block_corners.transpose()
        block_corners = np.dot(rotation_matrix, block_corners)
        block_corners = block_corners.transpose()

        block_corners[:,0] += x[i]
        block_corners[:,1] += y[i]
        block_corners[:,2] += z[i]

        block_corners = block_corners.tolist()

        mesh = msp.add_mesh()
        mesh.dxf.subdivision_levels = 0  # do not subdivide cube, 0 is the default value
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = block_corners
            mesh_data.faces = block_faces

    dwg.saveas(path)
    return


def face_position_dxf():
    return


def index_3D_to_1D(blockmodel:     pd.DataFrame,
                indexing:       int = 1,
                xcol:           str = None,
                ycol:           str = None,
                zcol:           str = None,
                xorigin:        Union[int, float] = None,
                yorigin:        Union[int, float] = None,
                zorigin:        Union[int, float] = None,
                xsize:          Union[int, float] = None,
                ysize:          Union[int, float] = None,
                zsize:          Union[int, float] = None,
                x_rotation:     Union[int, float] = 0,
                y_rotation:     Union[int, float] = 0,
                z_rotation:     Union[int, float] = 0,
                idxcol:         str = 'idx',
                inplace:        bool = False) -> pd.DataFrame:
    """
    Convert 3D array of xyz block centroids to 1D index that is reversible.
    :param blockmodel: pandas dataframe of block model
    :param indexing: controls whether origin block has coordinates 0,0,0 or 1,1,1
    :param xcol: x column in model
    :param ycol: y column in model
    :param zcol: z column in model
    :param xorigin: x origin of model - this is the corner of the bottom block (not the centroid)
    :param yorigin: y origin of model - this is the corner of the bottom block (not the centroid)
    :param zorigin: z origin of model - this is the corner of the bottom block (not the centroid)
    :param xsize: x dimension of regular parent blocks
    :param ysize: y dimension of regular parent blocks
    :param zsize: z dimension of regular parent blocks
    :param x_rotation: rotation of block model grid around x axis, -180 to 180 degrees
    :param y_rotation: rotation of block model grid around y axis, -180 to 180 degrees
    :param z_rotation: rotation of block model grid around z axis (xy plane), -180 to 180 degrees
    :param idxcol: name of the 1D index column added to the model
    :param inplace: whether to do calculation inplace on pandas.DataFrame

    :return pandas.DataFrame of 1D indexed block model
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
    '''
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    '''

    PandasObject.ijk = ijk
    PandasObject.xyz = xyz
    PandasObject.rotate_grid = rotate_grid
    PandasObject.group_weighted_average = group_weighted_average
    PandasObject.vulcan_csv = vulcan_csv
    PandasObject.vulcan_bdf = vulcan_bdf
    PandasObject.vulcan_bmf = vulcan_bmf
    PandasObject.geometric_reblock = geometric_reblock
    PandasObject.whittle_mod = whittle_mod
    PandasObject.model_rotation = model_rotation
    PandasObject.model_origin = model_origin
    PandasObject.check_regular = check_regular
    PandasObject.attribute_reblock = attribute_reblock
    PandasObject.check_internal_blocks_missing = check_internal_blocks_missing
    PandasObject.blocks2vtk = blocks2vtk
    PandasObject.blocks2dxf = blocks2dxf

    PandasObject.face_position_dxf = face_position_dxf


    PandasObject.index_3D_to_1D = index_3D_to_1D
    PandasObject.index_1D_to_3D = index_1D_to_3D
    PandasObject.table_to_4D_array = table_to_4D_array
    PandasObject.table_to_1D_array = table_to_1D_array


