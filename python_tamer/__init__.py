""" Initializations """

from pkg_resources import get_distribution, DistributionNotFound

from yaconfigobject import Config


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__author__ = """Todd C. Harris"""
__email__ = 'todd.harris@meteoswiss.ch'

CONFIG = Config(name='python_tamer.conf')
