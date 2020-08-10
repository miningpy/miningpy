import pandas as pd
from math import sin, cos, pi
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
from miningpy.utilities.numpy_vtk import *
import ezdxf
import vtk


def blocks2vtk(blockmodel:  pd.DataFrame,
               path:        str = None,
               xyz_cols:    Tuple[str, str, str] = ('x', 'y', 'z'),
               dims:        Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               rotation:    Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
               cols:        List[str] = None) -> bool:
    """
    exports blocks and attributes of block model to a vtk file to visualise in paraview

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for vtk file
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees
    cols: list of strings
        columns of attributes to visualise using vtk. If None then exports all columns

    Returns
    -------
    True if .vtu file is exported with no errors
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

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
    if isinstance(xsize, int) or isinstance(xsize, float):
        dx = np.ones(nc) * float(xsize)
    if isinstance(xsize, str):
        dx = blockmodel[xsize].values  # numpy 1D array

    if isinstance(ysize, int) or isinstance(ysize, float):
        dy = np.ones(nc) * float(ysize)
    if isinstance(ysize, str):
        dy = blockmodel[ysize].values  # numpy 1D array

    if isinstance(zsize, int) or isinstance(zsize, float):
        dz = np.ones(nc) * float(zsize)
    if isinstance(zsize, str):
        dz = blockmodel[zsize].values  # numpy 1D array

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

        pcoords.SetTuple3(id+0, block_corners[0, 0], block_corners[0, 1], block_corners[0, 2])
        pcoords.SetTuple3(id+1, block_corners[1, 0], block_corners[1, 1], block_corners[1, 2])
        pcoords.SetTuple3(id+2, block_corners[2, 0], block_corners[2, 1], block_corners[2, 2])
        pcoords.SetTuple3(id+3, block_corners[3, 0], block_corners[3, 1], block_corners[3, 2])
        pcoords.SetTuple3(id+4, block_corners[4, 0], block_corners[4, 1], block_corners[4, 2])
        pcoords.SetTuple3(id+5, block_corners[5, 0], block_corners[5, 1], block_corners[5, 2])
        pcoords.SetTuple3(id+6, block_corners[6, 0], block_corners[6, 1], block_corners[6, 2])
        pcoords.SetTuple3(id+7, block_corners[7, 0], block_corners[7, 1], block_corners[7, 2])
        id += 8

    # add points to the cell
    points.SetData(pcoords)

    # Create the cells for each block
    id = -1
    for i in range(nc):

        # add next cell
        voxelArray.InsertNextCell(8)

        # for each vertex
        for j in range(8):
            id += 1
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
                vtk_array.InsertNextValue(str(idx))
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
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(extractGrid.GetOutput())
    writer.Write()

    return True


def blocks2dxf(blockmodel:   pd.DataFrame,
               path:         str = None,
               dxf_split:    str = None,
               facetype:     str = '3DFACE',
               xyz_cols:     Tuple[str, str, str] = ('x', 'y', 'z'),
               dims:         Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
               rotation:     Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0)) -> bool:
    """
    exports blocks and attributes of block model to a vtk file to visualise in paraview

    Parameters
    ----------
    blockmodel: pd.DataFrame
        pandas dataframe of block model
    path: str
        filename for dxf files. If multiple dxfs produced, this will be used as the file suffix
    dxf_split: str
        column to split dxf files by.
        for example, could be the year mined column from minemax
        if None then one dxf is made of every block in blockmodel
    facetype: {'3DFACE', 'MESH', None}
        type of face for the blocks
        3DFACE will create standard dxf faces which are understood by most software
        MESH is a newer type which requires less space but might not work well
        None will create no face (could be useful when we add line drawing functionality in the function)
    xyz_cols: tuple of strings
        names of x,y,z columns in model
    dims: tuple of floats or ints
        x,y,z dimension of regular parent blocks
    rotation: tuple of floats or ints
        rotation of block model grid around x,y,z axis, -180 to 180 degrees

    Returns
    -------
    True if .dxf file(s) are exported with no errors
    """

    # definitions for simplicity
    xcol, ycol, zcol = xyz_cols[0], xyz_cols[1], xyz_cols[2]
    xsize, ysize, zsize = dims[0], dims[1], dims[2]
    x_rotation, y_rotation, z_rotation = rotation[0], rotation[1], rotation[2]

    # check for duplicate blocks and return warning
    dup_check = list(blockmodel.duplicated(subset=[xcol, ycol, zcol]).unique())
    assert True not in dup_check, 'duplicate blocks in dataframe'

    # check rotation is within parameters
    for rot in rotation:
        if -180 <= rot <= 180:
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
    x_sin = sin(x_rotation * (pi/180.0))
    x_cos = cos(x_rotation * (pi/180.0))
    y_sin = sin(y_rotation * (pi/180.0))
    y_cos = cos(y_rotation * (pi/180.0))
    z_sin = sin(z_rotation * (pi/180.0))
    z_cos = cos(z_rotation * (pi/180.0))

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
    dwg = ezdxf.new('AC1024')  # mesh requires the DXF 2000 or newer format
    # AC1009 = AutoCAD R12
    # AC1015 = AutoCAD R2000
    # AC1018 = AutoCAD R2004
    # AC1021 = AutoCAD R2007
    # AC1024 = AutoCAD R2010
    # AC1027 = AutoCAD R2013
    # AC1032 = AutoCAD R2018

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

        block_corners[:, 0] += x[i]
        block_corners[:, 1] += y[i]
        block_corners[:, 2] += z[i]

        # Creates Faces for the block sides
        if facetype == '3DFACE':
            bc = block_corners

            msp.add_3dface([bc[0], bc[1], bc[5], bc[4]])
            msp.add_3dface([bc[1], bc[2], bc[6], bc[5]])
            msp.add_3dface([bc[5], bc[4], bc[7], bc[6]])
            msp.add_3dface([bc[7], bc[4], bc[0], bc[3]])
            msp.add_3dface([bc[0], bc[1], bc[2], bc[3]])
            msp.add_3dface([bc[3], bc[2], bc[6], bc[7]])

        elif facetype == 'MESH':
            block_corners = block_corners.tolist()
            mesh = msp.add_mesh()
            mesh.dxf.subdivision_levels = 0  # do not subdivide cube, 0 is the default value
            with mesh.edit_data() as mesh_data:
                mesh_data.vertices = block_corners
                mesh_data.faces = block_faces

        else:
            print('No applicable facetype selected!')
            pass

    dwg.saveas(path)
    return True


def face_position_dxf():
    return


def extend_pandas_core():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.blocks2vtk = blocks2vtk
    PandasObject.blocks2dxf = blocks2dxf
    PandasObject.face_position_dxf = face_position_dxf
