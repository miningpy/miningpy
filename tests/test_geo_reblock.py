# standard automated tests for miningpy->core->gemometric_reblock function
import pytest
import pandas as pd
import miningpy

# block model example (zuck small - MineLib)
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"

# read in block model from link
data = pd.read_csv(url, compression='zip')

def test_geo_superblock():
    # listing attributes to carry through (n.b. dropping ID column)
    # keys are what to weight by and values are lists of attributes to be weighted
    varlist_agg = {
        'rock_tonnes': ['cost', 'value'],
        'ore_tonnes': [],
    }

    # take the max or min value of reblock
    min_cols = ['final_pit']
    max_cols = ['period']

    # reblock function
    reblock = data.geometric_reblock(
        dims=(1, 1, 1),  # original dims of model
        xyz_cols=('x', 'y', 'z'),
        origin=(-0.5, -0.5, -0.5),  # bottom left corner
        reblock_multiplier=(2, 2, 5),  # doubling x and y dim and keeping z dim the same
        varlist_agg=varlist_agg,
        min_cols=min_cols,
        max_cols=max_cols,
    )

    # check tonnages sum
    assert (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() < 1) and \
           (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() > -1), 'rock_tonnes lost superblocking'

    # check dims
    new_dims = reblock.block_dims(xyz_cols=('x', 'y', 'z'),
                                  origin=(-0.5, -0.5, -0.5))
    assert new_dims == (2, 2, 5), 'reblocking dims error'


def test_geo_subblock():
    # listing attributes to carry through (n.b. dropping ID column)
    # keys are what to weight by and values are lists of attributes to be weighted
    varlist_agg = {
        'rock_tonnes': ['cost', 'value'],
        'ore_tonnes': [],
    }

    # take the max or min value of reblock
    min_cols = ['final_pit']
    max_cols = ['period']

    # reblock function
    reblock = data.geometric_reblock(
        dims=(1, 1, 1),  # original dims of model
        xyz_cols=('x', 'y', 'z'),
        origin=(-0.5, -0.5, -0.5),  # bottom left corner
        reblock_multiplier=(0.5, 0.5, 0.5),  # doubling x and y dim and keeping z dim the same
        varlist_agg=varlist_agg,
        min_cols=min_cols,
        max_cols=max_cols,
    )

    # check tonnages sum
    assert (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() < 1) and \
           (data.rock_tonnes.sum() - reblock.rock_tonnes.sum() > -1), 'rock_tonnes lost subblocking'

    # check dims
    new_dims = reblock.block_dims(xyz_cols=('x', 'y', 'z'),
                                  origin=(-0.5, -0.5, -0.5))
    assert new_dims == (0.5, 0.5, 0.5), 'reblocking dims error'