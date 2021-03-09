import pandas as pd

default_nc_filename_format = 'UVery.AS_ch02.lonlat_yyyy01010000.nc'
default_data_directory = 'C:/Data/UV/'

default_map_options = {
    "title" : "Test map",
    "size" : [20,15],
    "save" : True,
    "img_dir" : "",
    "img_filename" : "default",
    "img_filetype" : "png",
    "brdr_nation" : True,
    "brdr_nation_rgba" : [0,0,0,0],
    "brdr_state" : False,
    "brdr_state_rgba" : [0,0,0,0.67],
    "cmap" : "jet",
    "cmap_limits" : None,
    "cbar" : True,
    "cbar_limits" : None
}

default_bin_widths = {
    "SED" : 0.1, 
    "J m-2" : 10, 
    "UVI" : 0.1, 
    "W m-2" : 0.0025, 
    "mW m-2" : 2.5
}

# This data should be stored in a separate file so it can be updated if the method is ever extended
Vernez_2015_vis_table = pd.DataFrame.from_records(
    columns=['Seated','Kneeling','Standing erect arms down','Standing erect arms up','Standing bowing'],
    index=['Face','Skull','Forearm','Upper arm','Neck','Top of shoulders','Belly','Upper back','Hand','Shoulder','Upper leg','Lower leg','Lower back'],
    data=[[53.7,28.7,46.6,44.9,19.2],
        [56.2,66.6,61.1,58.4,67.5],
        [62.3,56.5,49.4,53.1,62.1],
        [51.7,60.5,45.9,65.3,61.6],
        [58.3,84.3,67.6,65.2,81.6],
        [35.9,50.3,48.6,45.7,85.3],
        [58.1,45.1,50.3,49.6,15.2],
        [35.9,50.3,48.6,45.7,85.3],
        [59.2,58.8,42.4,55,58.5],
        [68,62,63,67.1,64],
        [65.4,45.4,50.9,51,43.5],
        [32.8,63.4,49.7,50.3,50],
        [44.9,51.6,56.6,53.4,86.9]])

# Below is a dictionary describing a range of synonyms for the anatomical zones defined in the Vis table.
Anatomic_zone_synonyms_reverse = {'Forearm' : ['wrist','Left extern radial','Right extern radial','Left wrist: radius head','Right wrist: radius head','Left wrist','Right wrist'],
    'Face' : ['Forehead'],
    'Upper back' : ['Right trapezoid','Left trapezoid','trapezius'],
    'Belly' : ['Chest'],
    'Shoulder' : ['Left deltoid','Right deltoid','Left shoulder','Right shoulder'],
    'Upper arm' : ['Left elbow','Right elbow','Left biceps','Right biceps'],
    'Upper leg' : ['Left thigh','Right thigh','Left knee','Right knee'],
    'Lower back' : ['Low back']}
# The dictionary is reversed so that the multiple synonyms can be mapped to the few correct terms for the Vis table.
Anatomic_zone_synonyms = {keys: old_keys for old_keys, old_values in Anatomic_zone_synonyms_reverse.items() for keys in old_values}
