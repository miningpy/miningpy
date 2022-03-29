# testing grade tonnage plot miningpy->core->grade_tonnage_plot function
import pandas as pd
import miningpy
import tempfile

# test data
url = "https://drive.google.com/uc?export=download&id=1RWuGs5Sf54FLPyDM1K-fK0a7Im9apW0t"

# make temp directories
def test_grade_tonnage_1(tmp_path):

    table_path = tmp_path / "mydir/table.xlsx"
    table_path.parent.mkdir()  # create a directory "mydir" in temp folder

    plot_path = tmp_path / "mydir/plot.png"
    plot_path.parent.mkdir()  # create a directory "mydir" in temp folder

    data = pd.read_csv(url, compression='zip')
    data.grade_tonnage_plot(grade_col='cu', ton_col='tonn', table_path=table_path, plot_path=plot_path, show_plot=False)

    assert table_path.exists(), "grade tonnage table path error"
    assert plot_path.exists(), "grade tonnage plot path error"
