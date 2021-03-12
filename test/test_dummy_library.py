import pytest

import datetime as dt
from python_tamer.library import *

def test_ExposureMap_max():

    test = ExposureMap(date_selection=pd.date_range(start="2018-07-01",end="2018-07-01"),
    units = "UVI",
    statistic = "max",
    data_directory="test/",
    nc_filename_format="UV_test_data_yyyy.nc",
    bin_width=0.1
    ).collect_data().calculate_map()

    assert test.map[50,50] == 8.05

def test_SpecificDoses():

    test = SpecificDoses(pd.DataFrame({
        "Date" : [dt.date(2018,7,1)],
        "Time_start" : [dt.time(11,0,0)],
        "Time_end" : [dt.time(12,0,0)],
        "Anatomic_zone" : ["Forehead"],
        "Posture" : ["Standing erect arms down"],
        "Latitude" : [46.79166],
        "Longitude" : [6.79167]
    }))
    test.data_directory = "test/"
    test.nc_filename_format = "UV_test_data_yyyy.nc"

    test = test.ER_from_posture().schedule_constant_exposure()

    test = test.calculate_specific_dose()

    assert test['Ambient_dose'][0] == pytest.approx(8.08847 * 0.9, 0.05)

