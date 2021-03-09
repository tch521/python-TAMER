import pytest

from python_tamer.library import *

def test_data_load():

    test = ExposureMap(date_selection=pd.date_range(start="2018-01-01",end="2018-01-01"),
    units = "UVI",
    statistic = "max"
    ).collect_data()


