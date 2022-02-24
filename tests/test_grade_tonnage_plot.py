# testing grade tonnage plot miningpy->core->grade_tonnage_plot function
import pandas as pd
import miningpy

# test data
testdata = {
    'x': [5, 5, 15],
    'y': [5, 15, 25],
    'z': [5, 5, 5],
    'ton': [50, 100, 50],
    'grade': [0.1, 0.2, 0.3]

}

data = pd.DataFrame(testdata)

data.grade_tonnage_plot(grade_col='grade', ton_col='ton')
