import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from typing import Union, List, Tuple
import ezdxf
import pyvista as pv
import base64
import os
import io
import vtk
from miningpy.utilities import vtu_serializer


def plot3D_dxf(path:              str = None,
               colour:            Tuple[float] = (0.666667, 1, 0.498039),
               show_wireframe:    bool = False,
               show_grid:         bool = False,
               cross_section:     bool = False,
               show_plot:         bool = True) -> pv.Plotter:
    """
    create activate 3D vtk plot of dxf that is fully interactive
    the dxf has to either be strings or a triangulation

    Parameters
    ----------
    path: str
        path to dxf file to visualise
    colour: tuple of floats
        default solid colouring of the triangulation
    show_wireframe: bool
        whether to show the edges/lines of a wireframe or not
    show_grid: bool
        add x,y,z grid to see coordinates on plot
    cross_section: bool
        add widget cross-section to plot
    show_plot: bool
        whether to open active window or just return pyvista.Plotter object
        to .show() later

    Returns
    -------
    pyvista.Plotter object & active window of dxf visualisation
    """

    # read in dxf file
    doc = ezdxf.readfile(path)

    # get dxf data
    msp = doc.modelspace()

    # check whether the dxf is strings or a triangulation
    # check whether there are 3D faces or polylines and then
    # call the correct utility function
    faces = msp.query('3DFACE')
    strings = msp.query('POLYLINE')

    # call correct utility function
    if len(faces) > 0 and len(strings) == 0:
        mesh = __plot_triangles(faces)

    elif len(faces) == 0 and len(strings) > 0:
        mesh = __plot_strings(strings)

    else:
        raise Exception("MiningPy ERROR - DXF not recognised")

    # set theme
    pv.set_plot_theme("ParaView")  # just changes colour scheme

    # create polydata pyvista object
    # plot dxf in pyvista window
    p = pv.Plotter(notebook=False, title="MiningPy DXF Viewer")

    if not cross_section:
        p.add_mesh(mesh=mesh,
                   show_edges=show_wireframe,
                   color=colour)

    if cross_section:
        p.add_mesh_clip_plane(mesh=mesh,
                              show_edges=show_wireframe,
                              color=colour)

    if show_grid:
        p.show_grid()

    p.show_axes()

    if show_plot:
        p.show(full_screen=True)

    return p


def __plot_strings(strings):
    """
    Utility function to produce a PyVista surface that can
    then be plotted in a window for the user

    Used for DXF of strings (i.e. pit design)

    Returns a pv.PolyData object
    """

    # extract points and lines from dxf object
    polylines = pv.PolyData()
    for line in strings:
        points = np.array([[0, 0, 0]], dtype=np.float64)
        for row, pnt in enumerate(line.points()):
            if row == 0:
                points[0, :] = np.array([[float(pnt[0]), float(pnt[1]), float(pnt[2])]])
            else:
                points = np.append(points, np.array([[float(pnt[0]), float(pnt[1]), float(pnt[2])]]), axis=0)
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points), 3), 2, dtype=np.int_)
        cells[:-1, 1] = np.arange(0, len(points)-1, dtype=np.int_)
        cells[:-1, 2] = np.arange(1, len(points), dtype=np.int_)
        cells[-1:, 1] = len(points)-1
        cells[-1:, 2] = 0
        poly.lines = cells
        polylines += poly

    return polylines


def __plot_triangles(faces):
    """
    Utility function to produce a PyVista surface that can
    then be plotted in a window for the user

    Used for DXF triangulations

    Returns a pv.PolyData object
    """

    # extract points and faces from dxf object
    num3Dfaces = len(faces)
    pointArray = np.zeros((num3Dfaces * 3, 3))
    cellArray = np.zeros((num3Dfaces, 4), dtype='int')
    cellData = []

    i = 0
    faceCounter = 0
    name = []
    for e in faces:
        pointArray[i, 0] = e.dxf.vtx0[0]
        pointArray[i, 1] = e.dxf.vtx0[1]
        pointArray[i, 2] = e.dxf.vtx0[2]

        pointArray[i + 1, 0] = e.dxf.vtx1[0]
        pointArray[i + 1, 1] = e.dxf.vtx1[1]
        pointArray[i + 1, 2] = e.dxf.vtx1[2]

        pointArray[i + 2, 0] = e.dxf.vtx2[0]
        pointArray[i + 2, 1] = e.dxf.vtx2[1]
        pointArray[i + 2, 2] = e.dxf.vtx2[2]

        cellArray[faceCounter, :] = [3, i, i + 1, i + 2]  # need the 3 at the start to tell pyvista its a triangulation
        cellData.append(e.dxf.layer)
        i = i + 3
        faceCounter = faceCounter + 1

    cellArray = np.hstack(cellArray)
    return pv.PolyData(pointArray, cellArray)


def dxf2vtk(path:      str = None,
            output:    str = None,
            colour:    Tuple[float] = (0.666667, 1, 0.498039)) -> pv.PolyData:
    """
    save dxf to .vtp (vtk polydata) file format so that
    it can be opened in paraview for external viewing

    Parameters
    ----------
    path: str
        path of input dxf file
    output: str
        path of html file to export
    colour: tuple of floats
        default solid colouring of the triangulation

    Returns
    -------
    exports .vtp file and returns pyvista.PolyData object
    """

    # add extension to path name for vtk file
    # 'vtu' because unstructured grid
    if output is True:
        if not output.lower().endswith('.vtp'):
            output = output + '.vtp'

    polydxf = plot3D_dxf(path=path,
                     colour=colour,
                     show_plot=False)

    polydxf.mesh.save(output)

    return polydxf.mesh


def export_dxf_html(path:      str = None,
                    output:    str = None,
                    data_name:     str = 'DXF',
                    colour:    Tuple[float] = (0.666667, 1, 0.498039)) -> bool:
    """
    exports dxf file and embeds the data in a
    paraview glance html app to visualise and distribute

    Parameters
    ----------
    path: str
        path of input dxf file
    output: str
        path of .vtp file to export
    data_name: str
        base name used for dataset in Paraview Glance
    colour: tuple of floats
        default solid colouring of the triangulation

    Returns
    -------
    True if .html file is exported with no errors
    """

    # pv glance html template path
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    template = os.path.join(__location__, r'ParaViewGlance.html')

    # get polydata of dxf in memory
    polydxf = plot3D_dxf(path=path,
                     colour=colour,
                     show_plot=False).mesh

    # get camera setting default for viewing in Paraview Glance
    # estimate rotation centre
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(polydxf)
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

    vtkjs = vtu_serializer(polydxf, data_name, colour, camera, 0)
    addDataToViewer([vtkjs], template, output)

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
                        dstHtml.write('glanceInstance.showApp();\n')
                        dstHtml.write('</script>\n')
                        dstHtml.write('<script>\n')
                        dstHtml.write('var contentToLoad = "%s";\n\n' % content)
                        dstHtml.write('Glance.importBase64Dataset("%s" , contentToLoad, glanceInstance.proxyManager);\n' % "BBAA.vtkjs")
                        dstHtml.write('</script>\n')

                dstHtml.write(line)

    return True


def face_position_dxf():
    return


def extend_pandas_dxf():
    """
    Extends pandas' PandasObject (Series,
    DataFrame) with functions defined in this file.
    """

    PandasObject.face_position_dxf = face_position_dxf
