""" Initializations """

from pkg_resources import get_distribution, DistributionNotFound # What's this? 
from .library import *
# from yaconfigobject import Config


# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#     # package is not installed
#     pass

__author__ = """Todd C. Harris"""
__email__ = 'todd.harris@meteoswiss.ch'
__version__ = "0.2.12"

# CONFIG = Config(name='python_tamer.conf')
