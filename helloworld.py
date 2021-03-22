import numpy as np
import netCDF4 as nc 
import pandas as pd
import datetime as dt
from python_tamer import *
from test.test_dummy_library import *

# test_ExposureMap_max()

# test_SpecificDoses()

test = SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',header=2,index_col=0,usecols="B:K"))

test = test.standard_column_names()

test['Time_start'] = convert_swiss_time_to_UTC(test,"Time_start")
test['Time_end'] = convert_swiss_time_to_UTC(test,"Time_end")

test = test.schedule_constant_exposure()

test = test.ER_from_posture()

test = test.calculate_specific_dose()

test.to_excel('C:/Data/Dosimetry/doses_test_day_fix.xlsx')
    
"""

testmap = ExposureMap(date_selection=pd.date_range(start="2005-01-01",end="2005-12-31"),
    statistic="std",
    units="UVI",
    exposure_schedule=np.concatenate([np.zeros(10),np.ones(4),np.zeros(10)])
    )


testmap.calculate_pix_hist().calculate_map().plot_map()

data_directory = "C:/Data/UV/"

filename = 'UVery.AS_ch02.lonlat_yyyy01010000.nc'

# dataset=nc.Dataset(data_directory+filename) #the netcdf package can automatically load multiple files at once, but even just one year is slow
# dataset.set_auto_mask(False)

# data = dataset['UV_AS'][:]*40 #the netcdf object has a lot of metadata - useful but wouldn't work if loading from DWH

# data24 = np.reshape(data,(24,8760//24,103,241),order='F') # hard to wrap head around but Fortran (F) order is important here

#potential_dose = np.sum(data24,axis=0)

# sched = np.array([0,0,0,0,0,0 , 0,0,0,0,1,1 , 0,0,0,0,0,0 , 0,0,0,0,0,0])

input_table = pd.read_excel(r'C:/Data/Dosimetry/David_old_UV_dosimetry.xls',parse_dates=[2],index_col=1)
legend_dict_reverse = {'Point' : ['Lieu de mesure'],
    'Date' : ['Date'],
    'Time_start' : ['Heure début'],
    'Time_end' : ['Heure fin'],
    'Measured_dose' : ['Exposition [MED]','Exposure'],
    'Anatomic_zone' : ['Zone anatomique','Body part'],
    'Posture' : ['Posture'],
    'Latitude' : ['lat'],
    'Longitude' : ['lon','lng']}
legend_dict = {keys: old_keys for old_keys, old_values in legend_dict_reverse.items() for keys in old_values}
input_table = input_table.rename(columns=legend_dict)
minimal_input_table = SpecificDoseEstimationTable(input_table[list(legend_dict_reverse.keys())])

minimal_input_table['Time_start'] = convert_swiss_time_to_UTC(input_table,"Time_start")
minimal_input_table['Time_end'] = convert_swiss_time_to_UTC(input_table,"Time_end")

minimal_input_table = minimal_input_table.ER_from_posture()


minimal_input_table = minimal_input_table.schedule_constant_exposure()

minimal_input_table = minimal_input_table.calculate_specific_dose()

minimal_input_table.to_excel('C:/Data/Dosimetry/doses_testnewcode.xlsx')

yearsmonths = pd.date_range(start="2015-01-02",end="2017-12-31")
# test = multi_file_dose_histogram(yearsmonths,schedule=sched)



msg="Hello World!"
print(msg)

"""