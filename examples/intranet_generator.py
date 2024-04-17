# Script used to generate maps and time-trace plots shown on the MeteoSuisse intranet page
# A. Iantchenko 17th april 2024
import python_tamer as pt
import numpy as np
import os

src_directory = '/home/aiantchenk/UVdata/'
example = pt.ExposureMapSequence(src_directory=src_directory,
                                 src_filename_format="UVery.AS_ch02.lonlat_yyyy01010000.nc")

#Define exposure schedule 1 (working day)
ES1 =   exposure_schedule = np.array([0  ,0  ,0  ,0  ,0  ,0  ,
                        0  ,0  ,1.0,1.0  ,1.0,1.0  ,
                        1.0,1.0,1.0  ,1.0  ,1.0,0  ,
                        0  ,0  ,0  ,0  ,0  ,0  ])

#Define exposure schedule 2 (working day with pause)
ES2 =   exposure_schedule = np.array([0  ,0  ,0  ,0  ,0  ,0  ,
                        0  ,0  ,1.0,1.0  ,1.0,0.0  ,
                        0.0,1.0,1.0  ,1.0  ,1.0,0  ,
                        0  ,0  ,0  ,0  ,0  ,0  ])


# Define function to simplify generating plots for intranet page
def automatic_exposure_sequence(example,ES=None,substr="",statistic=["max","min"],unit="SED",match_cmap_limits=None,savedir='/home/aiantchenk/UVdata2'):
    # function to automatically generate and save UV plots
    # No exposure schedule
    if not isinstance(ES,np.ndarray):
        example = example.collect_data(['annual'],year_selection=[0],units=unit)
        img_dir= savedir + '/annual' + substr +'/'
        if not os.path.exists(img_dir): 
            os.makedirs(img_dir) 
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)
        # Yearly averaged over all years
        example = example.collect_data(['annual'],year_selection=np.sort(example.dataset_years),units=unit)
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)
        # then Monthly averaged over all years
        example = example.collect_data('monthly',year_selection=np.sort(example.dataset_years),units=unit)
        img_dir= savedir + '/monthly' + substr +'/'
        if not os.path.exists(img_dir): 
            os.makedirs(img_dir)         
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)
    else: 
        # Repeat with ES1
        example = example.collect_data(['annual'],year_selection=[0],units=unit,exposure_schedule=ES)
        img_dir= savedir + '/annual' + substr +'/'
        if not os.path.exists(img_dir): 
            os.makedirs(img_dir)         
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)
        # Yearly averaged over all years
        example = example.collect_data(['annual'],year_selection=np.sort(example.dataset_years),units=unit,exposure_schedule=ES)
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)
        # then Monthly averaged over all years
        example = example.collect_data('monthly',year_selection=np.sort(example.dataset_years),units=unit,exposure_schedule=ES)
        img_dir= savedir + '/monthly' + substr +'/'
        if not os.path.exists(img_dir): 
            os.makedirs(img_dir)         
        example = example.calculate_maps(statistic=statistic)
        example.save_maps(save=True,show=False,img_dir=img_dir,match_cmap_limits=match_cmap_limits) 
        example.save_trace(save=True,show=False,img_dir=img_dir)

# I include here to generate all the scripts, however sometimes VS code crashes 
# so you might have to do it in parts (comment out what you do not want to run)

# generate for statistic: mean, units: SED
statistic=['mean']
#match_cmap_limits=[0,55] # Deactivate if you do not want to use manual limits (and set next flag to true)
match_cmap_limits=True
substr = '_mean'
unit="SED"

print("1) Generating: mean, SED, no ES ...")
automatic_exposure_sequence(example,substr=substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)  
print("1.1) Generating: mean, SED, with ES1 ...")
automatic_exposure_sequence(example,ES=ES1,substr= "_ES1" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("1.2) Generating: mean, SED, with ES2 ...")
automatic_exposure_sequence(example,ES=ES2,substr= "_ES2" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("Done")

# same for unit: UVI
unit="UVI"
print("2) Generating: mean, UVI, no ES ...")
automatic_exposure_sequence(example,substr=substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)  
print("2.1) Generating: mean, UVI, with ES1 ...")
automatic_exposure_sequence(example,ES=ES1,substr= "_ES1" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("2.2) Generating: mean, UVI, with ES2 ...")
automatic_exposure_sequence(example,ES=ES2,substr= "_ES2" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("Done")

# generate for statistic: max, units: SED
statistic=['max']
#match_cmap_limits=[0,55] # Deactivate if you do not want to use manual limits (and set next flag to true)
match_cmap_limits=True
substr = 'max'
unit="SED"

print("3) Generating: max, SED, no ES ...")
automatic_exposure_sequence(example,substr=substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)  
print("3.1) Generating: max, SED, with ES1 ...")
automatic_exposure_sequence(example,ES=ES1,substr= "_ES1" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("3.2) Generating: max, SED, with ES2 ...")
automatic_exposure_sequence(example,ES=ES2,substr= "_ES2" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("Done")

# same for unit: UVI
unit="UVI"
print("4) Generating: max, UVI, no ES ...")
automatic_exposure_sequence(example,substr=substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)  
print("4.1) Generating: max, UVI, with ES1 ...")
automatic_exposure_sequence(example,ES=ES1,substr= "_ES1" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("4.2) Generating: max, UVI, with ES2 ...")
automatic_exposure_sequence(example,ES=ES2,substr= "_ES2" + substr,statistic=statistic,unit=unit,match_cmap_limits=match_cmap_limits)
print("Done")


print("All Done")
