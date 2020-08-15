import pandas as pd
import miningpy
import json

with open("examples/data_links.json", 'r') as file:
    data = json.load(file)

url = data['blockmodel']['mclaughlin']
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
blockModel = pd.read_csv(path, compression='zip')

blockModel.plot3D(
    xyz_cols=('x', 'y', 'z'),
    dims=(1, 1, 1),  # block dimensions (5m * 5m * 5m)
    col='blockvalue',  # block attribute to colour by
)
