set /p version=<miningpy/VERSION
python setup.py install --single-version-externally-managed --record=record.txt --testversion %version%.{{ DEPLOY_TIME }}