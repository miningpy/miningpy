import json
import requests
import hashlib
import os
import miningpy

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['dxf']['strings_vulcan']

file_id = url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
dxf_raw = requests.get(dwn_url).text
dxf_raw = dxf_raw.replace("\r\n", "\n")

temp_file = 'examples/' + hashlib.md5(dxf_raw.encode(encoding='UTF-8')).hexdigest()

# create temporary dxf file for ezdxf
with open(temp_file, 'w') as file:
    file.write(dxf_raw)

# read temporary dxf file and plot using MiningPy function
miningpy.export_dxf_html(temp_file, output='test.html')

# delete temporary dxf file that was downloaded
os.remove(temp_file)

