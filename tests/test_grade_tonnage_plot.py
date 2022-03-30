# testing grade tonnage plot miningpy->core->grade_tonnage_plot function
import pandas as pd
import miningpy
import tempfile
import imghdr

# test data
testdata1 = {
    'tonnage': [50, 100, 50, 100, 50, 100, 50, 100],
    'cu': [5.0, 10.0, 25.0, 50.0, 5.0, 10.0, 25.0, 50.0],
    'rocktype': ['ox', 'ox', 'sulph', 'sulph', 'ox', 'ox', 'sulph', 'sulph']
}

# result
resultdata1 = {
    'cog': [2.5, 12.5, 50],
    'tonnage': [600.0, 300.0, 200.0],
    'avg_grade': [25.0, ((125 / 300) * 100), 50.0],
}

cog_grades = [2.5, 12.5, 50]


# write test, no image
def test_grade_tonnage_1():
    data = pd.DataFrame(testdata1)
    output = data.grade_tonnage_plot(grade_col='cu', ton_col='tonnage', cog_grades=cog_grades, show_plot=False)
    result = pd.DataFrame(resultdata1)

    # check result
    check = (output.sum() - result.sum()).sum()
    assert check.sum() < 0.0001, "grade tonnage table not the same"


# test data
testdata2 = {
    'tonnage': [50, 100, 50, 100, 50, 100, 50, 100],
    'cu': [5.0, 10.0, 25.0, 50.0, 5.0, 10.0, 25.0, 50.0],
    'rocktype': ['ox', 'ox', 'sulph', 'sulph', 'ox', 'ox', 'sulph', 'sulph']
}

# result
resultdata2 = {
    'cog': [5.0, 27.5, 50],
    'tonnage': [600.0, 200.0, 200.0],
    'avg_grade': [25.0, 50.0, 50.0]
}

cog_grade_points = 3


# test with image
def test_grade_tonnage_2():
    data = pd.DataFrame(testdata2)

    # make temp jpg, delete=False to test png
    plot_path = tempfile.NamedTemporaryFile(mode="wb", delete=False)

    output = data.grade_tonnage_plot(grade_col='cu', ton_col='tonnage', cog_grade_points=cog_grade_points,
                                     show_plot=False, plot_path=plot_path.name)
    result = pd.DataFrame(resultdata2)

    # check result of grade tonnage table
    check = (output.sum() - result.sum()).sum()
    assert check.sum() < 0.0001, "grade tonnage table not the same"

    # check plot
    path = plot_path.name + '.png'
    assert imghdr.what(path, h=None) == 'png', 'grade tonnage plot error'

    # close temporary file
    plot_path.close()

