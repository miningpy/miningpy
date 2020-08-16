import pandas as pd
import miningpy

# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
blockModel = pd.read_csv(url, compression='zip')

# remove blocks not in schedule
mask = blockModel['period'] >= 0
blockModel = blockModel[mask].copy()

# make period into an int
blockModel['period'] = blockModel['period'].astype(int)

blockModel.export_html(
    path='schedule.html',
    xyz_cols=('x', 'y', 'z'),
    dims=(1, 1, 1),  # block dimensions (5m * 5m * 5m)
    data_name='period',
    split_by='period'
)
