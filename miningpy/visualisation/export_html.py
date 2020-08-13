# import libraries
import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from typing import Union, Tuple, List
import miningpy.visualisation.core
from miningpy.utilities import vtu_serializer
import base64
import os
import io
import vtk


def export_html(blockmodel:    pd.DataFrame,
                path:          str = None,
                xyz_cols:      Tuple[str, str, str] = ('x', 'y', 'z'),
                dims:          Tuple[Union[int, float], Union[int, float], Union[int, float]] = None,
                rotation:      Tuple[Union[int, float], Union[int, float], Union[int, float]] = (0, 0, 0),
                cols:          List[str] = None,
                data_name:     str = 'blockModel',
                colour:        Tuple[float] = (0.666667, 1, 0.498039),
                split_by:      str = None,
                colour_range:  Tuple[str] = ('blue', 'red')) -> bool:
    """
    exports blocks and attributes of block model
    and embeds the data in a paraview glance html app
    to visualise and distribute

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
    data_name: str
        base name used for dataset in Paraview Glance
        if the split_by column is specified then the unique values in
        the split_by colum are appended to the base name
    colour: tuple of floats
        default solid colouring of blocks
        if a column is chosen to split by then this colouring is not considered.
    split_by: str
        column that is used to split up the block model into components in the Paraview Glance
        app HTML file. The maximum number of unique values in this column is 256.
    colour_range: tuple of strings
        colouring range of values in the split_by column
        if no split_by then colour_range is ignored.
        accepted colours are: 'red', 'blue', 'green', 'white', 'black'
        default is ('blue', 'red'). i.e. colour blue to red

    Returns
    -------
    True if .html file is exported with no errors
    """

    # check number of unique values in split_by column
    if split_by is not None and len(blockmodel[split_by].unique()) > 255:
        raise Exception(f"Error - too many unique values in column used to split by: {split_by}")

    # get list of split_by values and order them ascending
    # then build colour range
    if split_by is not None:
        vtu_dict = dict()
        uniqueValues = blockmodel[split_by].copy()
        uniqueValues = uniqueValues.sort_values()  # ascending
        uniqueValues = uniqueValues.unique()

        uniqueColours = get_colours(uniqueValues, colour_range)

        for i, val in enumerate(uniqueValues):
            vtu_dict[val] = dict()
            vtu_dict[val]['colour'] = uniqueColours[i]

    # pv glance html template path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    template = os.path.join(__location__, r'ParaViewGlance.html')

    # get unstructured grid of block model in memory
    vtu = blockmodel.blocks2vtk(
        xyz_cols=xyz_cols,
        dims=dims,
        rotation=rotation,
        cols=cols,
        output_file=False
    )

    # get vtu for each split_by unique value
    if split_by is not None:
        for val in uniqueValues:
            mask = blockmodel[split_by] == val
            temp_mod = blockmodel[mask].copy()

            temp_vtu = temp_mod.blocks2vtk(
                            xyz_cols=xyz_cols,
                            dims=dims,
                            rotation=rotation,
                            cols=cols,
                            output_file=False
                        )

            data = vtk.vtkGeometryFilter()
            data.SetInputData(temp_vtu)
            data.Update()
            temp_vtu = data.GetOutput()
            vtu_dict[val]['vtu'] = temp_vtu

    # get camera setting default for viewing in Paraview Glance
    data = vtk.vtkGeometryFilter()
    data.SetInputData(vtu)
    data.Update()
    dataset = data.GetOutput()

    # estimate rotation centre
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(vtu)
    mapper.Update()

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor1)
    renderer.MakeCamera()
    renderer.ResetCamera()

    activeCamera = renderer.GetActiveCamera()

    camera = dict()
    camera['focalPoint'] = activeCamera.GetFocalPoint()
    camera['position'] = activeCamera.GetPosition()
    camera['viewUp'] = activeCamera.GetViewUp()
    camera['clippingRange'] = activeCamera.GetClippingRange()

    if split_by is None:
        # create byte representation of VTKjs file
        vtkjs = miningpy.utilities.vtu_serializer(dataset, data_name, colour, camera)
        # embed vtp into paraview glance html file
        addDataToViewer([vtkjs], template, path)
    else:
        for val in uniqueValues:
            vtu_dict[val]['vtkjs'] = miningpy.utilities.vtu_serializer(vtu_dict[val]['vtu'], data_name + '_' + str(val), list(vtu_dict[val]['colour']), camera)

        vtklist = []
        for val, data in vtu_dict.items():
            vtklist.append(data['vtkjs'])

        addDataToViewer(vtklist, template, path)

    return True


def addDataToViewer(injectData, srcHtmlPath, dstHtmlPath):
    # Extract data as base64
    base64dict = dict()
    for vtp in injectData:
        base64Content = base64.b64encode(vtp)
        base64Content = base64Content.decode().replace('\n', '')
        base64dict[vtp] = base64Content

    # Create new output file
    with io.open(srcHtmlPath, mode='r', encoding="utf-8") as srcHtml:
        with io.open(dstHtmlPath, mode='w', encoding="utf-8") as dstHtml:
            for line in srcHtml:
                if '</body>' in line:
                    for file, content in base64dict.items():
                        dstHtml.write('<script>\n')
                        dstHtml.write('var contentToLoad = "%s";\n\n' % content);
                        dstHtml.write('Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n' % "BBAA.vtkjs");
                        dstHtml.write('glanceInstance.showApp();\n');
                        dstHtml.write('</script>\n')

                dstHtml.write(line)

    return True


def get_colours(values, colour_range):
    # using a diverging colour spectrum

    colours = dict()

    colours['red'] = (1.0, 0.0, 0.0)
    colours['blue'] = (0.0, 0.0, 1.0)
    colours['green'] = (0.0, 1.0, 0.0)
    colours['white'] = (1.0, 1.0, 1.0)
    colours['black'] = (0.0, 0.0, 0.0)

    num_values = len(values)

    if num_values == 1:
        return colour_range[0]

    if num_values > 1:
        return np.linspace(colours[colour_range[0]], colours[colour_range[1]], num=num_values)


def extend_pandas_html():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.export_html = export_html
