import json
import requests
import hashlib
import os
import miningpy

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['dxf']['wireframe_vulcan']

dxf_raw = requests.get(url).text
dxf_raw = dxf_raw.replace("\r\n", "\n")

temp_file = 'examples/' + hashlib.md5(dxf_raw.encode(encoding='UTF-8')).hexdigest()

# create temporary dxf file for ezdxf
with open(temp_file, 'w') as file:
    file.write(dxf_raw)

# read temporary dxf file and plot using MiningPy function
miningpy.dxf2vtk(temp_file, output='test.vtp')

# delete temporary dxf file that was downloaded
os.remove(temp_file)

