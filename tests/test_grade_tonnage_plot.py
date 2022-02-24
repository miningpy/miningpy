# testing grade tonnage plot miningpy->core->grade_tonnage_plot function
import pandas as pd
import miningpy

# test data
url = "https://drive.google.com/uc?export=download&id=1SOrYhqiu5Tg8Zjb7be4fUWhbFDTU1sEk"
data = pd.read_csv(url, compression='zip')

data.grade_tonnage_plot(grade_col='value', ton_col='rock_tonnes', table_path='grade_tonnage_plot.xlsx')

