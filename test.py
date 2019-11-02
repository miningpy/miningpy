import pandas as pd
import mining_utils

data ={'x': [1,2],
        'y': [1,2],
        'z': [1,1],
        }

bm = pd.DataFrame(data)
print(bm)

bm.ijk(
    xcol='x',
    ycol='y',
    zcol='z',
    xorigin=0,
    yorigin=0,
    zorigin=0,
    xsize=1,
    ysize=1,
    zsize=1,
    inplace=True,
    x_rotation=45,
)

print(bm)

