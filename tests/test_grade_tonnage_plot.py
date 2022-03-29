# testing grade tonnage plot miningpy->core->grade_tonnage_plot function
import pandas as pd
import miningpy


def test_grade_tonnage_plot_1():
    url = "https://drive.google.com/uc?export=download&id=1RWuGs5Sf54FLPyDM1K-fK0a7Im9apW0t"
    data = pd.read_csv(url, compression='zip')
    data.grade_tonnage_plot(grade_col='cu', ton_col='tonn', table_path='grade_tonnage_plot.xlsx',
                            plot_path='grade_t.png', show_plot=True)

