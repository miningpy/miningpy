import requests
import hashlib
import os
import miningpy

# dxf wireframe example
url = "https://drive.google.com/uc?export=download&id=1RyaNDSV4K_LrjoIiJrFZ4KAbzj7iySuh"

dxf_raw = requests.get(url).text
dxf_raw = dxf_raw.replace("\r\n", "\n")

temp_file = 'examples/' + hashlib.md5(dxf_raw.encode(encoding='UTF-8')).hexdigest()

# create temporary dxf file for ezdxf
with open(temp_file, 'w') as file:
    file.write(dxf_raw)

# read temporary dxf file and plot using MiningPy function
miningpy.export_dxf_html(temp_file, output='dxf_triangulation.html')

# delete temporary dxf file that was downloaded
os.remove(temp_file)

