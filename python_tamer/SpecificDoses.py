import pandas as pd
import numpy as np
import netCDF4 as nc
from .subroutines import *


class SpecificDoses(pd.DataFrame):
    """A class for specific dose estimates akin to dosimetry measurements

    High resolution data allows for personal and ambient dose estimation without the need for
    direct measurement. This class is structured like a table with a set of functions to add 
    columns ultimately leading to dose estimates. Each row of this table represents a specific
    exposure instance, i.e. an individual at a specific location for a specific date and time
    with a specific exposure ratio. See Harris et al. 2021 
    (https://doi.org/10.3390/atmos12020268) for more information on calculations appropriate 
    for this class.


    Parameters
    ----------

    src_filename_format : str
        Describes the filename of the netCDF files containing the UV data with 'yyyy' in place 
        of the year.
    
    src_directory : str
        The directory where the data is stored. Must end with a slash.


    Notes
    -----

    Presently, the class is inherited from a pandas.DataFrame which is somewhat restrictive 
    and will likely be revised in a later update. For the time being, this means that the 
    parameters cannot be set when initialising a `SpecificDoses` object, they must instead
    be adjusted after initialisation, like so::

        ExistingExposureMapObject.src_directory = "/new/data/directory/"

    
    Example
    -------

    In this example, we illustrate the process for calculating the doses in Harris et al. 2021
    (https://doi.org/10.3390/atmos12020268) from the spreadsheet supplied as supplementary 
    data (https://www.mdpi.com/2073-4433/12/2/268/s1). Note that results will differ as the
    spreadsheet contains only local Swiss time and not UTC time. There are four important
    functions as part of this class, three for standardising and preparing the columns,
    and one for actually loading the data and performing the dose calculations. See below::

        import python_tamer as pt
        import pandas as pd
        example = pt.SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',
                                                 header=2,index_col=0,usecols="B:K"))
        example.src_directory = 'C:/enter_the_directory_of_your_dataset_here'
        example = example.standard_column_names() 
        example = example.schedule_constant_exposure().ER_from_posture()
        example = example.calculate_specific_dose()


    """

    # This property ensures that functions return the same subclass
    @property
    def _constructor(self):
        return SpecificDoses
    
    # This adds some useful metadata (self-explanatory)
    _metadata = ["src_filename_format","src_directory"]
    src_filename_format = 'UVery.AS_ch02.lonlat_yyyy01010000.nc'
    src_directory = 'C:/Data/UV/' # TODO: set up __init__ for these options
    # It feels like this should be declared with __init__ as well but idk

    def standard_column_names(self) :
        """Limited function to standardise column names

        When loading tables to use as the basis for a SpecificDoses table, some columns may have
        slightly names to what is expected. This function standardises the names but is very
        limited in terms of what it can recognise. The user is encourages to ensure the columns
        are correctly labelled themselves and not to rely on this function.

        Returns
        -------
        SpecificDoses
            The table has its column names modified.
        """
        legend_dict_reverse = {'Point' : ['Lieu de mesure'],
            'Date' : ['Date'],
            'Time_start' : ['Heure début','Start_time','Start time'],
            'Time_end' : ['Heure fin','End_time','End time'],
            'Measured_dose' : ['Exposition [MED]','Exposure'],
            'Anatomic_zone' : ['Zone anatomique','Body part','Anatomic zone'],
            'Posture' : ['Body posture'],
            'Latitude' : ['lat'],
            'Longitude' : ['lon','lng']}
        legend_dict = {keys: old_keys for old_keys, old_values in legend_dict_reverse.items() for keys in old_values}
        self = self.rename(columns=legend_dict)

        return self

    def schedule_constant_exposure(self) :
        """Generates exposure schedules given start and end times.

        This function generates exposure schedules based on simple continuous exposure, i.e.
        with a start time and an end time. The exposure schedule is a vector with length 24
        with each entry representing the proportion of the corresponding hour of the day that
        the subject is exposed. 


        Returns
        -------

        python_tamer.SpecificDoses
            An exposure_schedule column is created and is appended to the input 
            `SpecificDoses` object or, if that column already exists, it is overwritten.
        

        Notes
        -----

        The input `SpecificDoses` object must contain the following columns:
        * ``Time_start``
        * ``Time_end``


        Example
        -------

        In this example, we illustrate the process for calculating the doses in Harris et al. 2021
        (https://doi.org/10.3390/atmos12020268) from the spreadsheet supplied as supplementary 
        data (https://www.mdpi.com/2073-4433/12/2/268/s1). Note that results will differ as the
        spreadsheet contains only local Swiss time and not UTC time. There are four important
        functions as part of this class, three for standardising and preparing the columns,
        and one for actually loading the data and performing the dose calculations. See below::

            import python_tamer as pt
            import pandas as pd
            example = pt.SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',
                                                    header=2,index_col=0,usecols="B:K"))
            example.src_directory = 'C:/enter_the_directory_of_your_dataset_here'
            example = example.standard_column_names() 
            example = example.schedule_constant_exposure().ER_from_posture()
            example = example.calculate_specific_dose()


        """

        def schedule_constant_exposure_iter(Start_time,End_time) :
            """Iterates through rows of a SpecificDoses table to generate schedules.

            This function is designed to be applied to each row in a datatable to generate an
            exposure schedule based on a start time and end time

            Parameters
            ----------
            Start_time : datetime.time
                UTC time at which exposure period begins
            End_time : datetime.time
                UTC time at which exposure period end

            Returns
            -------
            numpy.array
                24 length vector of values between 0 and 1 indicating proportion 
                of time exposed for that corresponding hour of the day.


            """

            schedule = np.zeros(24)
            schedule[Start_time.hour:End_time.hour] = 1

            # Modify start and end hours according to proportion of time exposed
            if Start_time.minute != 0 :
                schedule[Start_time.hour] = (1 - Start_time.minute/60)

            if End_time.minute != 0 :
                schedule[End_time.hour] = End_time.minute/60 

            return schedule
        # With that function defined, we need just one line to apply it to the whole table
        self["Schedule"] = self.apply(lambda x: schedule_constant_exposure_iter(
            x["Time_start"],x["Time_end"]),axis='columns')
        return self
    
    def ER_from_posture(self,
    Vis_table_path=None,
    Vis_table=None) :
        """ER_from_posture calculates Exposure Ratios for a given anatomic zone, posture, and date.

        This function calculates ER as a percentage between 0 and 100 based on information from an input table.
        The input table must contain certain columns at a minimum. Those are: Date, Anatomic_zone, and Posture.
        This function contains hard-coded synonyms for certain anatomical zones, e.g. 'Forehead" maps to "Face'.
        See Vernez et al., Journal of Exposure Science and Environmental Epidemiology (2015) 25, 113–118 
        (https://doi.org/10.1038/jes.2014.6) for further details on the model used for the calculation.


        Parameters
        ----------

        Vis_table_path : str, optional
            The full path to an alternative table for the Vis parameter. 
                Must be a csv file. Defaults to None.
        Vis_table : str, optional
            An alternative table for the Vis parameter. Defaults to None.


        Returns
        -------

        SpecificDoses
            Returns input table appended with ER column


        Notes
        -----

        The SpecificDoses table used must contain columns for Date, Anatomic_zone, and Posture.
        The Date column should contain DateTime entries. The Anatonic_zone column should contain one string per 
        row describing the exposed body part. The Posture column should contain one string per row describing 
        one of six accepted postures.


        Example
        -------

        In this example, we illustrate the process for calculating the doses in Harris et al. 2021
        (https://doi.org/10.3390/atmos12020268) from the spreadsheet supplied as supplementary 
        data (https://www.mdpi.com/2073-4433/12/2/268/s1). Note that results will differ as the
        spreadsheet contains only local Swiss time and not UTC time. There are four important
        functions as part of this class, three for standardising and preparing the columns,
        and one for actually loading the data and performing the dose calculations. See below::

            import python_tamer as pt
            import pandas as pd
            example = pt.SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',
                                                    header=2,index_col=0,usecols="B:K"))
            example.src_directory = 'C:/enter_the_directory_of_your_dataset_here'
            example = example.standard_column_names() 
            example = example.schedule_constant_exposure().ER_from_posture()
            example = example.calculate_specific_dose()


        """

        # This chunk of code checks if the default Vis table should be used or if the user enters some alternative table.
        if Vis_table is None and Vis_table_path is None :
            Vis_table = pd.DataFrame.from_records(
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
            # The 'standing moving' posture must be dealt with somehow...
            # Vis_table['Standing moving']= (Vis_table['Standing erect arms down'] + Vis_table['Standing bowing']) / 2
            # TODO: add interpeter or force users to conform?
            Vis_table['Standing moving']= Vis_table['Standing erect arms down'] 
        elif Vis_table is None :
            Vis_table = pd.read_csv(Vis_table_path)

        # Below is a dictionary describing a range of synonyms for the anatomical zones defined in the Vis table.
        Anatomic_zone_synonyms_reverse = {
            'Forearm' : ['wrist',
                         'Left extern radial',
                         'Right extern radial',
                         'Left wrist: radius head',
                         'Right wrist: radius head',
                         'Left wrist',
                         'Right wrist'],
            'Face' : ['Forehead'],
            'Upper back' : ['Right trapezoid',
                            'Left trapezoid',
                            'trapezius'],
            'Belly' : ['Chest'],
            'Shoulder' : ['Left deltoid',
                          'Right deltoid',
                          'Left shoulder',
                          'Right shoulder'],
            'Upper arm' : ['Left elbow',
                           'Right elbow',
                           'Left biceps',
                           'Right biceps'],
            'Upper leg' : ['Left thigh',
                           'Right thigh',
                           'Left knee',
                           'Right knee'],
            'Lower back' : ['Low back']
        }
        # The dictionary is reversed so that the multiple synonyms can be mapped to the few correct terms for the Vis table.
        Anatomic_zone_synonyms = {keys: old_keys for old_keys, old_values in Anatomic_zone_synonyms_reverse.items() for keys in old_values}

        self = self.replace({'Anatomic_zone' : Anatomic_zone_synonyms})

        # With the correct anatomic zone names established, we can lookup the Vis values from the table
        # TODO: lookup is being depreciated, must replace with something new
        Vis = Vis_table.lookup(self['Anatomic_zone'],self['Posture'])

        # Next we must calculate the minimal Solar Zenith Angle for the given date
        mSZA = min_solar_zenith_angle(self.Date,self.Latitude)

        # With the Vis value and the SZA, we can calculate the ER according to the Vernez model
        self.loc[:,'ER'] = ER_Vernez_model_equation(Vis,mSZA) / 100

        return self

    def calculate_specific_dose(self) :
        """Calculates doses according to exposure schedule, ER, date, and location.

        This function takes the SpecificDoseEstimationTable and calculates the specific 
        ambient and personal doses according to the exposure schedule and ER. There are
        a few key steps to this function. First it reads the Date column to determine 
        which years of data must be loaded. It then iterates through each year, loading
        only the necessary dates. It applies the exposure schedule and the ER to 
        calculate the ambient and personal doses.


        Returns
        -------

        SpecificDoses
            The input table is appended with a Ambient_dose and Personal_dose column.
        

        Notes
        -----

        The input SpecificDoses object must include Date, Schedule, ER, Latitude,
        and Longitude columns.
        Consult Harris et al. 2021 (https://doi.org/10.3390/atmos12020268) for more 
        information on how this function can be used in the context of mimicking UV
        dosimetry measurements.


        Example
        -------

        In this example, we illustrate the process for calculating the doses in Harris et al. 2021
        (https://doi.org/10.3390/atmos12020268) from the spreadsheet supplied as supplementary 
        data (https://www.mdpi.com/2073-4433/12/2/268/s1). Note that results will differ as the
        spreadsheet contains only local Swiss time and not UTC time. There are four important
        functions as part of this class, three for standardising and preparing the columns,
        and one for actually loading the data and performing the dose calculations. See below::

            import python_tamer as pt
            import pandas as pd
            example = pt.SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',
                                                    header=2,index_col=0,usecols="B:K"))
            example.src_directory = 'C:/enter_the_directory_of_your_dataset_here'
            example = example.standard_column_names() 
            example = example.schedule_constant_exposure().ER_from_posture()
            example = example.calculate_specific_dose()


        """

        # First step is find unique years to avoid loading unnecessary data
        years = pd.DatetimeIndex(self.Date).year
        unique_years = sorted(set(years))

        self['Ambient_dose'] = np.nan 
        self['Personal_dose'] = np.nan

        for year in unique_years :
            # Load netCDF file
            print("Processing year "+str(year)) 
            dataset=nc.Dataset(self.src_directory+self.src_filename_format.replace('yyyy',str(year))) 
            dataset.set_auto_mask(False) # This is important for nans to import correctly

            # Make temporary table for yearly subset
            temp_table = self[years == year].copy()

            # find all unique days in year to be loaded
            unique_days,unique_days_idx = np.unique(pd.DatetimeIndex(temp_table.Date).dayofyear,
                return_inverse=True)
            temp_table['unique_days_idx'] = unique_days_idx

            #pd.DatetimeIndex(nc.num2date(dataset.variables["time"][:],dataset.variables["time"].units,only_use_cftime_datetimes=False))

            if dataset.dimensions['time'].size == 24 :
                # needed if just a single day
                time_subset = [True for i in range(dataset.dimensions['time'].size)]
            else :
                # Next we pull a subset from the netCDF file
                # declare false array with same length of time dimension from netCDF
                time_subset = [False for i in range(dataset.dimensions['time'].size)] 
                # reshape false array to have first dimension 24 (hours in day)
                time_subset = assert_data_shape_24(time_subset) 
                # set the appropriate days as true
                time_subset[:,unique_days-1] = True 
                # flatten time_subset array back to one dimension
                time_subset = time_subset.flatten(order='F')

            data = assert_data_shape_24(dataset['UV_AS'][time_subset,:,:]) 
            # TODO: improve comprehension of raw data units rather than assuming

            # convert lat lon into pixel coordinates
            # TODO: consider is necessary to load entire maps for just a few required pixels
            lat = dataset['lat'][:]
            lon = dataset['lon'][:]
            temp_table['pixel_lat'] = temp_table.apply(lambda x: 
                find_nearest(lat,x['Latitude']),axis='columns')
            temp_table['pixel_lon'] = temp_table.apply(lambda x: 
                find_nearest(lon,x['Longitude']),axis='columns')

            
            # calculate doses
            temp_table['Ambient_dose'] = temp_table.apply(lambda x: 
                np.sum(data[:,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] * 
                    x['Schedule']),axis='columns')
            temp_table['Personal_dose'] = temp_table.apply(lambda x: 
                np.sum(data[:,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] * 
                    (x['Schedule'] * x['ER'])),axis='columns')

            # extra step necessary to ensure correct assignment
            self.loc[temp_table.index,'Ambient_dose'] = temp_table['Ambient_dose'].values
            self.loc[temp_table.index,'Personal_dose'] = temp_table['Personal_dose'].values

        # TODO: improve units options here
        self['Ambient_dose'] = self['Ambient_dose']/40*3600/100 # SED
        self['Personal_dose'] = self['Personal_dose']/40*3600/100 # SED
        return self        





    def analyse_variable(self,
    variable="UV_AS",
    statistic="Mean",
    src_filename_format=None,
    src_directory=None) :
        """Basic calculations for specific exposure instances

        This function is for calculating information other than ambient and personal
        doses that corresponds to specific exposure instances. 


        Parameters
        ----------
        variable : str, optional
            The name of the variable to be analysed. This informs what data should be
            pulled from the source netCDF files. This also informs the name of the column(s)
            that will be created by this function. Defaults to "UV_AS", i.e. the All-Sky
            UV data that is used in the calculate_specific_dose function.

        statistic : str or list, optional
            The statistic to be calculated, options include: mean, median, stdev, variance, 
            min, max, weighted_mean, and sum. Not case sensitive. Can be a single string or
            a list of strings whereby multiple columns will be calculated. Defaults to "Mean".

        src_filename_format : str, optional
            Allows the user to select different source data. This may be useful in cases where
            the user wants to compare doses calculated with one dataset to (say) cloud cover
            from another dataset. Defaults to None, where the function uses the source files
            specified by the object's metadata.

        src_directory : str, optional
            Allows the user to select different source data. This may be useful in cases where
            the user wants to compare doses calculated with one dataset to (say) cloud cover
            from another dataset. Defaults to None, where the function uses the source files
            specified by the object's metadata.


        Returns
        -------
        SpecificDoses
            The table is appended with new columns named [variable]_[statistic].


        Example
        -------

        In this example, we illustrate the process for calculating the doses in Harris et al. 2021
        (https://doi.org/10.3390/atmos12020268) from the spreadsheet supplied as supplementary 
        data (https://www.mdpi.com/2073-4433/12/2/268/s1). Note that results will differ as the
        spreadsheet contains only local Swiss time and not UTC time. Additionally, to demonstrate
        the analyse_variable function, we also calculate the weighted mean CMF assuming it to be
        an additional variable in the source data files. See below::

            import python_tamer as pt
            import pandas as pd
            example = pt.SpecificDoses(pd.read_excel(r'atmosphere-12-00268-s001.xlsx',
                                                    header=2,index_col=0,usecols="B:K"))
            example.src_directory = 'C:/enter_the_directory_of_your_dataset_here'
            example = example.standard_column_names() 
            example = example.schedule_constant_exposure().ER_from_posture()
            example = example.calculate_specific_dose()
            example = example.analyse_variable(variable="CMF",statistic="weighted_mean")

        """

        # users have option to load different files, otherwise defaults to metadata
        if src_filename_format is None :
            src_filename_format = self.src_filename_format
        if src_directory is None :
            src_directory = self.src_directory

        # First step is find unique years to avoid loading unnecessary data
        years = pd.DatetimeIndex(self.Date).year
        unique_years = sorted(set(years))

        if isinstance(statistic,str) :
            self[variable+"_"+statistic] = np.nan
            # convert to list to simplify code later
            statistic = [statistic]
        elif isinstance(statistic,list) :
            for x in statistic :
                self[variable+"_"+x]=np.nan
        else :
            raise TypeError("statistic input must be str or list of str")

        for year in unique_years :
            # Load netCDF file
            print("Processing year "+str(year)) 
            dataset=nc.Dataset(src_directory+src_filename_format.replace('yyyy',str(year))) 
            dataset.set_auto_mask(False) # This is important for nans to import correctly

            # Make temporary table for yearly subset
            temp_table = self[years == year].copy()

            # find all unique days in year to be loaded
            unique_days,unique_days_idx = np.unique(pd.DatetimeIndex(temp_table.Date).dayofyear,
                return_inverse=True)
            temp_table['unique_days_idx'] = unique_days_idx

            #pd.DatetimeIndex(nc.num2date(dataset.variables["time"][:],dataset.variables["time"].units,only_use_cftime_datetimes=False))

            if dataset.dimensions['time'].size == 24 :
                # needed if just a single day
                time_subset = [True for i in range(dataset.dimensions['time'].size)]
            else :
                # Next we pull a subset from the netCDF file
                # declare false array with same length of time dimension from netCDF
                time_subset = [False for i in range(dataset.dimensions['time'].size)] 
                # reshape false array to have first dimension 24 (hours in day)
                time_subset = assert_data_shape_24(time_subset) 
                # set the appropriate days as true
                time_subset[:,unique_days-1] = True 
                # flatten time_subset array back to one dimension
                time_subset = time_subset.flatten(order='F')

            data = assert_data_shape_24(dataset[variable][time_subset,:,:]) 
            # TODO: improve comprehension of raw data units rather than assuming

            # convert lat lon into pixel coordinates
            # TODO: consider is necessary to load entire maps for just a few required pixels
            lat = dataset['lat'][:]
            lon = dataset['lon'][:]
            temp_table['pixel_lat'] = temp_table.apply(lambda x: 
                find_nearest(lat,x['Latitude']),axis='columns')
            temp_table['pixel_lon'] = temp_table.apply(lambda x: 
                find_nearest(lon,x['Longitude']),axis='columns')

            
            # calculate 
            for stat in statistic :
                # mean
                if stat.lower() in ["mean",'average','avg'] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.mean(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # median
                elif stat.lower() in ["median","med"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.median(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # stdev
                elif stat.lower() in ["std","sd","stdev"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.std(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # variance
                elif stat.lower() in ["var","variance"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.var(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # minimum
                elif stat.lower() in ["min",'minimum'] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.amin(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # maximum
                elif stat.lower() in ["max","maximum"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.amax(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')
                # weighted mean
                elif stat.lower() in ["weighted_mean","weighted_average","mean_weighted","average_weighted","avg_weighted"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.average(data[:,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']],weights=x['Schedule']),axis='columns')
                # sum
                elif stat.lower() in ["sum","total"] :
                    temp_table[variable+"_"+stat] = temp_table.apply(lambda x: 
                        np.sum(data[x['Schedule']!=0,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] ),axis='columns')

                # extra step necessary to ensure correct assignment
                self.loc[temp_table.index,variable+"_"+stat] = temp_table[variable+"_"+stat].values

        return self        
