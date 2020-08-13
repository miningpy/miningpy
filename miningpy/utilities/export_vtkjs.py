# coding: utf-8
"""
Serializer of vtk render windows
Adpation from :
https://kitware.github.io/vtk-js/examples/SceneExplorer.html
https://raw.githubusercontent.com/Kitware/vtk-js/master/Utilities/ParaView/export-scene-macro.py
Licence :
https://github.com/Kitware/vtk-js/blob/master/LICENSE
"""

import os, sys, json, hashlib, zipfile
from io import BytesIO

if sys.version_info < (3,):
    import imp
    vtk = imp.load_module('vtk', *imp.find_module('vtk'))
else:
    import vtk

if sys.version_info >= (2, 7):
    buffer = memoryview
else:
    buffer = buffer

from collections import namedtuple

SCALAR_MODE = namedtuple("SCALAR_MODE",
    "Default UsePointData UseCellData UsePointFieldData UseCellFieldData UseFieldData"
)(0, 1, 2, 3, 4, 5)

COLOR_MODE = namedtuple("COLOR_MODE", "DirectScalars MapScalars")(0, 1)

ACCESS_MODE = namedtuple("ACCESS_MODE", "ById ByName")(0, 1)

# NOTE:
# VTKjs does not support int64 data types (this is a general JS issue, not just VTKjs)
# so any int64 types will be treated as int32 - not this may cause data loss issues
# VTKjs also only handles numerical columns
# any string / character columns will be skipped
# BE AWARE!

arrayTypesMapping = [
    ' ',  # VTK_VOID            0
    ' ',  # VTK_BIT             1
    'b',  # VTK_CHAR            2
    'B',  # VTK_UNSIGNED_CHAR   3
    'h',  # VTK_SHORT           4
    'H',  # VTK_UNSIGNED_SHORT  5
    'i',  # VTK_INT             6
    'I',  # VTK_UNSIGNED_INT    7
    'l',  # VTK_LONG            8
    'L',  # VTK_UNSIGNED_LONG   9
    'f',  # VTK_FLOAT          10
    'd',  # VTK_DOUBLE         11
    'L',  # VTK_ID_TYPE        12
    ' ',  # VTK_STRING         13
    ' ',  # VTK_OPAQUE         14
    ' ',  # VTK_SIGNED_CHAR    15
    'l',  # VTK_LONG_LONG      16
    'l',  # VTK___INT64            18
    'l',  # VTK_UNSIGNED___INT64   19
]

_js_mapping = {
    'b': 'Int8Array',
    'B': 'Uint8Array',
    'h': 'Int16Array',
    'H': 'Int16Array',
    'i': 'Int32Array',
    'I': 'Uint32Array',
    'l': 'Int32Array',
    'L': 'Uint32Array',
    'f': 'Float32Array',
    'd': 'Float64Array',
}

_writer_mapping = {}


def _get_range_info(array, component):
    r = array.GetRange(component)
    compRange = {}
    compRange['min'] = r[0]
    compRange['max'] = r[1]
    compRange['component'] = array.GetComponentName(component)
    return compRange


def _get_ref(destDirectory, md5):
    ref = {}
    ref['id'] = md5
    ref['encode'] = 'BigEndian' if sys.byteorder == 'big' else 'LittleEndian'
    ref['basepath'] = destDirectory
    return ref


def _get_object_id(obj, objIds):
    try:
        idx = objIds.index(obj)
        return idx + 1
    except ValueError:
        objIds.append(obj)
        return len(objIds)


def _dump_data_array(scDirs, datasetDir, dataDir, array):
    root = {}
    if not array:
        return None

    if array.GetDataType() == 12:
        # IdType need to be converted to Uint32
        arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        newArray = vtk.vtkTypeUInt32Array()
        newArray.SetNumberOfTuples(arraySize)
        for i in range(arraySize):
            newArray.SetValue(i, -1 if array.GetValue(i) < 0 else array.GetValue(i))
        pBuffer = buffer(newArray)
    elif array.GetDataType() > 15:
        # convert all Int64 to Int32 arrays
        arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        newArray = vtk.vtkTypeInt32Array()
        newArray.SetNumberOfTuples(arraySize)
        for i in range(arraySize):
            newArray.SetValue(i, array.GetValue(i))
        pBuffer = buffer(newArray)
    else:
        pBuffer = buffer(array)

    pMd5 = str(array.GetName())+'_'+hashlib.md5(pBuffer).hexdigest()
    pPath = os.path.join(dataDir, pMd5)

    scDirs.append([pPath, bytes(pBuffer)])

    root['ref'] = _get_ref(os.path.relpath(dataDir, datasetDir), pMd5)
    root['vtkClass'] = 'vtkDataArray'

    if array.GetName() is None:
        root['name'] = 'default'
    else:
        root['name'] = array.GetName()

    root['dataType'] = _js_mapping[arrayTypesMapping[array.GetDataType()]]
    root['numberOfComponents'] = array.GetNumberOfComponents()
    root['size'] = array.GetNumberOfComponents() * array.GetNumberOfTuples()
    root['ranges'] = []
    if root['numberOfComponents'] > 1:
        for i in range(root['numberOfComponents']):
            root['ranges'].append(_get_range_info(array, i))
        root['ranges'].append(_get_range_info(array, -1))
    else:
        root['ranges'].append(_get_range_info(array, 0))
    return root


def _dump_tcoords(scDirs, datasetDir, dataDir, dataset, root):
    tcoords = dataset.GetPointData().GetTCoords()
    if tcoords:
        dumpedArray = _dump_data_array(scDirs, datasetDir, dataDir, tcoords)
        root['pointData']['activeTCoords'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({ 'data': dumpedArray })


def _dump_normals(scDirs, datasetDir, dataDir, dataset, root):
    normals = dataset.GetPointData().GetNormals()
    if normals:
        dumpedArray = _dump_data_array(scDirs, datasetDir, dataDir, normals)
        root['pointData']['activeNormals'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({ 'data': dumpedArray })


def _dump_all_arrays(scDirs, datasetDir, dataDir, dataset, root):
    for data_loc in ['pointData', 'cellData']:  # 'fieldData'
        root[data_loc] = {
            'vtkClass': 'vtkDataSetAttributes',
            "activeGlobalIds":-1,
            "activeNormals":-1,
            "activePedigreeIds":-1,
            "activeScalars":-1,
            "activeTCoords":-1,
            "activeTensors":-1,
            "activeVectors":-1,
            "arrays": []
        }

    # Point data
    pd = dataset.GetPointData()
    pd_size = pd.GetNumberOfArrays()
    for i in range(pd_size):
        array = pd.GetArray(i)
        if array:
            dumpedArray = _dump_data_array(scDirs, datasetDir, dataDir, array)
            root['pointData']['activeScalars'] = 0
            root['pointData']['arrays'].append({ 'data': dumpedArray })

    # Cell data
    cd = dataset.GetCellData()
    cd_size = cd.GetNumberOfArrays()
    for i in range(cd_size):
        array = cd.GetArray(i)
        if array:
            dumpedArray = _dump_data_array(scDirs, datasetDir, dataDir, array)
            root['cellData']['activeScalars'] = -1
            root['cellData']['arrays'].append({ 'data': dumpedArray })


def _dump_poly_data(scDirs, datasetDir, dataDir, dataset, root):
    root['vtkClass'] = 'vtkPolyData'

    # Points
    root['points'] = _dump_data_array(scDirs, datasetDir, dataDir, dataset.GetPoints().GetData())
    root['points']['vtkClass'] = 'vtkPoints'

    # Cells & polys
    for cell_type in ['verts', 'lines', 'polys', 'strips']:
        cell = getattr(dataset, 'Get' + cell_type.capitalize())()
        if cell and cell.GetData().GetNumberOfTuples() > 0:
            root[cell_type] = _dump_data_array(scDirs, datasetDir, dataDir, cell.GetData())
            root[cell_type]['vtkClass'] = 'vtkCellArray'

    # PointData TCoords
    _dump_tcoords(scDirs, datasetDir, dataDir, dataset, root)

    # PointData Normals
    _dump_normals(scDirs, datasetDir, dataDir, dataset, root)

    # PointData Normals
    _dump_all_arrays(scDirs, datasetDir, dataDir, dataset, root)


_writer_mapping['vtkPolyData'] = _dump_poly_data


def _write_data_set(scDirs, dataset, newDSName):

    dataDir = os.path.join(newDSName, 'data')

    root = {}
    root['metadata'] = {}
    root['metadata']['name'] = newDSName

    writer = _writer_mapping[dataset.GetClassName()]
    if writer:
        writer(scDirs, newDSName, dataDir, dataset, root)
    else:
        raise Warning('{} is not supported'.format(dataset.GetClassName()))

    scDirs.append([os.path.join(newDSName, 'index.json'), json.dumps(root, indent=2)])


def vtu_serializer(dataset, componentName, colour, camera):
    """
    Function to convert a vtk render window in a list of 2-tuple where first value
    correspond to a relative file path in the `vtkjs` directory structure and values
    of the binary content of the corresponding file.
    """
    objIds = []
    scDirs = []

    sceneComponents = []
    textureToSave = {}

    if dataset and dataset.GetPoints():
        componentName = componentName

        _write_data_set(scDirs, dataset, newDSName=componentName)

        sceneComponents.append({
            "name": componentName,
            "type": "httpDataSetReader",
            "httpDataSetReader": {
                "url": componentName
            },
            "actor": {
                "origin": [0, 0, 0],
                "scale": [1, 1, 1],
                "position": [0, 0, 0],
            },
            "actorRotation": [0, 0, 0, 1],
            "mapper": {
                "colorByArrayName": "",
                "colorMode": 1,
                "scalarMode": 0
            },
            "property": {
                "representation": 2,
                "edgeVisibility": 1,
                "diffuseColor": colour,
                "pointSize": 2,
                "opacity": 1
            }
        })

    sceneDescription = {
        "fetchGzip": False,
        "background": [0.32, 0.34, 0.43],
        "camera": {
            "focalPoint": camera['focalPoint'],
            "position": camera['position'],
            "viewUp": camera['viewUp'],
            "clippingRange": camera['clippingRange']
        },
        "centerOfRotation": camera['focalPoint'],
        "scene": sceneComponents,
        "lookupTables": {}
    }

    scDirs.append(['index.json', json.dumps(sceneDescription, indent=4)])

    # create binary stream of the vtkjs directory structure
    compression = zipfile.ZIP_DEFLATED
    with BytesIO() as in_memory:
        zf = zipfile.ZipFile(in_memory, mode="w")
        try:
            for dirPath, data in scDirs:
                zf.writestr(dirPath, data, compress_type=compression)
        finally:
                zf.close()
        in_memory.seek(0)
        return in_memory.read()

