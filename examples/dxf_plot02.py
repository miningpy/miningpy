import ezdxf
import pandas as pd
import numpy as np
import pyvista as pv
import json
import requests
import hashlib
import os

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['dxf']['wireframe_vulcan']

file_id = url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
dxf_raw = requests.get(dwn_url).text
dxf_raw = dxf_raw.replace("\r\n", "\n")

temp_file = 'examples/' + hashlib.md5(dxf_raw.encode(encoding='UTF-8')).hexdigest()

# create temporary dxf file for ezdxf
with open(temp_file, 'w') as file:
    file.write(dxf_raw)

# read temporary dxf file into ezdxf object
doc = ezdxf.readfile(temp_file)

# delete temporary dxf file that was downloaded
os.remove(temp_file)

msp = doc.modelspace()

faces = msp.query('3DFACE')
lines = msp.query('LINE')
points = msp.query('POINT')

# extract points and faces from dxf object
num3Dfaces = len(faces)
pointArray = np.zeros((num3Dfaces*3, 3))
cellArray = np.zeros((num3Dfaces, 4), dtype='int')
cellData = []

i = 0
faceCounter = 0
name = []
for e in faces:
    pointArray[i, 0] = e.dxf.vtx0[0]
    pointArray[i, 1] = e.dxf.vtx0[1]
    pointArray[i, 2] = e.dxf.vtx0[2]

    pointArray[i+1, 0] = e.dxf.vtx1[0]
    pointArray[i+1, 1] = e.dxf.vtx1[1]
    pointArray[i+1, 2] = e.dxf.vtx1[2]

    pointArray[i+2, 0] = e.dxf.vtx2[0]
    pointArray[i+2, 1] = e.dxf.vtx2[1]
    pointArray[i+2, 2] = e.dxf.vtx2[2]

    cellArray[faceCounter, :] = [3, i, i+1, i+2]  # need the 3 at the start to tell pyvista its a triangulation
    cellData.append(e.dxf.layer)
    i = i+3
    faceCounter = faceCounter+1

cellArray = np.hstack(cellArray)
surf = pv.PolyData(pointArray, cellArray)

plotter = pv.Plotter()
plotter.add_mesh(surf, show_edges=False)
plotter.show()
