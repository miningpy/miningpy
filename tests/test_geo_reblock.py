# standard automated tests for miningpy->core->gemometric_reblock function
import pytest
import pandas as pd
import miningpy


# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
data = pd.read_csv(url, compression='zip')
varlist_agg = {
    'rock_tonnes': ['cost', 'value'],
    'ore_tonnes': [],

}
min_cols = ['final_pit']
max_cols = ['period']

reblock = data.geometric_reblock(
    dims=(1, 1, 1),
    xyz_cols=('x', 'y', 'z'),
    origin=(-0.5, -0.5, -0.5),
    reblock_multiplier=(2,2,2),
    varlist_agg=varlist_agg,
    min_cols=min_cols,
    max_cols=max_cols,
)

print('ho')
