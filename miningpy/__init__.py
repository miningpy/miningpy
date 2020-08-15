from miningpy.core import *
from miningpy.visualisation import *
from miningpy.vulcan import *
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# versioning
with open(os.path.join(__location__, 'VERSION')) as version_file:
    ver = version_file.read().strip()

__version__ = ver
