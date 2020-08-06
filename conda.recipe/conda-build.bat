REM clean build dir
conda build purge

REM check the recipe
conda-build conda.recipe --check

REM build the recipe
conda-build conda.recipe --no-anaconda-upload --channel conda-forge

REM test the package
conda-build conda.recipe --test