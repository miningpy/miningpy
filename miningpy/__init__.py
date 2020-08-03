import miningpy.core
import miningpy.visualisation
import miningpy.vulcan
import os

DIR = os.path.dirname(__file__)

# versioning
with open(os.path.join(DIR, 'VERSION')) as version_file:
    ver = version_file.read().strip()

__version__ = ver
