""" python_TAMER """ 

"""
This library contains two classes at present: SpecificDoses and ExposureMap. Each class has
a typical pipeline of operations the user can work through to produce the desired results:
specific ambient and personal dose estimations or maps of some exposure metric respectively.
Currently, the dataset only works with the Vuilleumier et al. erythemal UV dataset for
Switzerland. See https://doi.org/10.1016/j.envint.2020.106177 for more information.
"""

import numpy as np
import netCDF4 as nc
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import datetime as dt
import regex as re
from .subroutines import *
from .initial_reference import *

class SpecificDoses(pd.DataFrame):
    """SpecificDoses is the class for replicating dosimetry measurements

    High resolution data allows for personal and ambient dose estimation without the need for
    direct measurement. This class is structured like a table with a set of functions to add 
    columns ultimately leading to dose estimates. Each row of this table represents a specific
    exposure instance, i.e. an individual at a specific location for a specific date and time
    with a specific exposure ratio. See Harris et al. 2021 
    (https://doi.org/10.3390/atmos12020268) for more information on calculations appropriate 
    for this class.

    Parameters
    ----------
    nc_filename_format : str
        Describes the filename of the netCDF files containing the UV data with 'yyyy' in place 
        of the year.
    
    data_directory : str
        The directory where the data is stored. Must end with a slash.

    """

    # This property ensures that functions return the same subclass
    @property
    def _constructor(self):
        return SpecificDoses
    
    # This adds some useful metadata (self-explanatory)
    _metadata = ["nc_filename_format","data_directory"]
    nc_filename_format = default_nc_filename_format
    data_directory = default_data_directory # TO DO: set up __init__ for these options
    # It feels like this should be declared with __init__ as well but idk

    def schedule_constant_exposure(self) :
        """schedule_constant_exposure generates exposure schedules given start and end times.

        This function generates exposure schedules based on simple continuous exposure, i.e.
        with a start time and an end time. The exposure schedule is a vector with length 24
        with each entry representing the proportion of the corresponding hour of the day that
        the subject is exposed. 
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
        (doi:10.1038/jes.2014.6) for further details on the model used for the calculation.

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

        """

        # This chunk of code checks if the default Vis table should be used or if the user enters some alternative table.
        if Vis_table is None and Vis_table_path is None :
            Vis_table = Vernez_2015_vis_table
            # The 'standing moving' posture must be dealt with somehow...
            # Vis_table['Standing moving']= (Vis_table['Standing erect arms down'] + Vis_table['Standing bowing']) / 2
            Vis_table['Standing moving']= Vis_table['Standing erect arms down'] 
        elif Vis_table is None :
            Vis_table = pd.read_csv(Vis_table_path)

        self = self.replace({'Anatomic_zone' : Anatomic_zone_synonyms})

        # With the correct anatomic zone names established, we can lookup the Vis values from the table
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
        Read Harris et al. 2021 (https://doi.org/10.3390/atmos12020268) for more 
        information on how this function can be used in the context of mimicking UV
        dosimetry measurements.

        """

        # First step is find unique years to avoid loading unnecessary data
        years = pd.DatetimeIndex(self.Date).year
        unique_years = sorted(set(years))

        self['Ambient_dose'] = np.nan 
        self['Personal_dose'] = np.nan

        for year in unique_years :
            # Load netCDF file
            print("Processing year "+str(year)) 
            dataset=nc.Dataset(self.data_directory+self.nc_filename_format.replace('yyyy',str(year))) 
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
                time_subset[:,unique_days] = True 
                # flatten time_subset array back to one dimension
                time_subset = time_subset.flatten(order='F')

            data = assert_data_shape_24(dataset['UV_AS'][time_subset,:,:]*40) 
            # *40 converts it to uv index
            # TO DO: improve comprehension of raw data units rather than assuming

            # convert lat lon into pixel coordinates
            # TO DO: consider is necessary to load entire maps for just a few required pixels
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
                    x['Schedule'] * x['ER']),axis='columns')

            # extra step necessary to ensure correct assignment
            self.loc[temp_table.index,'Ambient_dose'] = temp_table['Ambient_dose'].values
            self.loc[temp_table.index,'Personal_dose'] = temp_table['Personal_dose'].values

        # TO DO: improve units options here
        self['Ambient_dose'] = self['Ambient_dose']/40*3600/100 # SED
        return self        


















class ExposureMap:
    """ is a class for calculating maps based on user specifications

    Each instance of this class contains information required to calculate and illustrate a map
    of exposure information, be that simple averages or more advanced mathematical representations
    of exposure risk. The class is designed to have three core functions run in sequence, with
    room for flexibility should more advanced users desire it. First, the data is read and a pixel
    histogram is calculated. This allows much more data to be stored in memory and is the basis
    for performing this kind of data analysis on personal computers.

    Parameters
    ----------
    units : str
        The units of the quantity to be mapped. Must be "SED", "J m-2" or "UVIh" for doses or "UVI",
        "W m-2" or "mW m-2" for irradiances. Defaults to "SED".

    exposure_schedule : array (length-24)
        A vector of values describing the relative exposure of each hour of the day. 0 indicates no
        exposure, 1 indicates full exposure, and a fractional value such as 0.5 would indicate 
        exposure for a total of 30 minutes within the hour, or a 50% partial exposure for the full
        hour, or anything equivalent. Values greater than 1 are allowed. When not calculating doses,
        hours with any non-zero entry in this vector are included, with the corresponding irradiance
        value being multiplied by the value


    """

    def __init__(self,units="SED",
    exposure_schedule=np.ones(24),
    statistic="mean",
    bin_width = "default",
    date_selection="all",
    map_options=default_map_options,
    nc_filename_format=default_nc_filename_format,
    data_directory=default_data_directory):
        self.units = units
        self.exposure_schedule=exposure_schedule
        self.statistic = statistic
        self.map_options = map_options
        self.nc_filename_format = nc_filename_format
        self.data_directory = data_directory
        self.date_selection = date_selection
        if bin_width == "default" :
            self.bin_width = default_bin_widths[self.units]
        else :
            self.bin_width = bin_width
    
    def collect_data(self, data_directory=None,nc_filename_format=None,
    date_selection=None,units=None,exposure_schedule=None,bin_width=None) :
        """collect_data calculates histograms for each pixel of the underlying data

        In order to handle large amounts of data without exceeding memory limitations, files are
        loaded one at a time and the time dimension is removed, either by calculating daily doses
        or by simply taking the data as is. The resulting information is then stored not as a 
        list of specific values but rather binned into a histogram for each pixel. This process
        is repeated for each file required by the user input, building up the pixel histograms
        with more information that does not require additional memory.

        Parameters
        ----------
        data_directory : str, optional
            Directory containing multi-file dataset.

        nc_filename_format : str, optional
            Filename of multi-file dataset with year replaced by 'yyyy'.

        date_selection : pandas.DateTimeIndex, optional
            The list of dates from which to pull data. Untested, but other datatypes probably 
            acceptable. Currently relies on output from pandas.date_range function.

        units : str, optional
            Name of units of desired output. This also indicates whether daily doses must be 
            calculated or not. Units of "SED", "J m-2", or "UVIh" will produce daily doses,
            units of "UVI", "W m-2" or "mW m-2" will not. 

        exposure_schedule : array, optional
            A length-24 array of values indicating the proportion of the corresponding hour
            spent fully exposed. A value of 1 indicates full exposure, 0 indicates no 
            exposure, and 0.5 could indicate either half an hour of exposure or a full hour 
            of 50% exposure. 

        bin_width : float, optional
            The width of the histogram bins according to the chosen units.

        Returns
        -------
        ExposureMap
            The input ExposureMap object is appended with new fields, ``pix_hist`` contains
            the counts for the histogram, and ``bin_edges``, `bin_centers``, and ``num_bins``
            all serve as metadata for the pixel histograms. ``lat`` and ``lon`` are also 
            added from the multi-file dataset to inform the pixel locations for map making
            further down the typical pipeline.
        
        
        """

        # TO DO: There must be a better way to do this
        if not (data_directory is None) :
            self.data_directory = data_directory
        if not (nc_filename_format is None) :
            self.nc_filename_format = nc_filename_format
        if not (date_selection is None) :
            self.date_selection = date_selection
        if not (units is None) :
            self.units = units
        if not (exposure_schedule is None) :
            self.exposure_schedule = exposure_schedule
        if not (bin_width is None) :
            self.bin_width = bin_width

        # first we read the data_directory to check the total number of unique years available
        data_dir_contents = os.listdir(self.data_directory)
        # TO DO: improve jankiness of this format-matching search for filenames
        char_year = self.nc_filename_format.find('yyyy')
        dataset_years = [ x for x in data_dir_contents if re.findall(self.nc_filename_format.replace("yyyy","[0-9]{4}"),x)]
        dataset_years = [ int(x[char_year:char_year+4]) for x in dataset_years ]

        # Now we can handle default options like "all"
        if type(self.date_selection) == str and self.date_selection == "all" :
            date_selection = pd.date_range(start=str(dataset_years[0])+"-01-01",
                end=str(dataset_years[-1])+"-12-31")
        else :
            date_selection = self.date_selection # TO DO: much more interpretation options here

        #now we find unique years 
        list_of_years = sorted(set(date_selection.year))

        for i in range(len(list_of_years)) :
            year = list_of_years[i]
            print("Processing year "+str(year)) #should use logging, don't yet know how
            dataset=nc.Dataset(self.data_directory+self.nc_filename_format.replace('yyyy',str(year))) 
            dataset.set_auto_mask(False) #to get normal arrays (faster than default masked arrays)

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
                time_subset[:,date_selection[date_selection.year == year].dayofyear-1] = True 
                # flatten time_subset array back to one dimension
                time_subset = time_subset.flatten(order='F')

            # load subset of data
            print("   Slicing netcdf data with time subset")
            data = dataset['UV_AS'][time_subset,:,:]*40 #work in UVI by default because it's easy to read
            # TO DO: check units of dataset files, CF conventions for UVI or W/m2

            # now to calculate doses if requested
            if self.units in ["SED","J m-2","UVIh"] :
                # if calculating doses
                print('   Calculating doses')
                data = assert_data_shape_24(data)
                data = np.sum(np.reshape(self.exposure_schedule,[24,1,1,1]) * data,axis=0)

            elif (self.exposure_schedule != np.ones(24)).any() :
                # assume elsewise calculating intensity (i.e. UV-index) then limit data selection according
                # to schedule (remembering that default schedule is just ones)
                print('   Slicing data with exposure schedule')
                # reshape so first dimension is 24 hours
                data = assert_data_shape_24(data)
                # select only those hours with nonzero entry in exposure schedule
                data = data[self.exposure_schedule != 0,:,:,:]
                # select nonzero values from exposure schedule
                exposure_schedule_nonzero = self.exposure_schedule[self.exposure_schedule != 0]

                # if any nonzero entries aren't 1, multiply data accordingly
                if (exposure_schedule_nonzero != 1).any() :
                    data *= np.reshape(exposure_schedule_nonzero,[len(exposure_schedule_nonzero),1,1,1])

                # recombine first two dimensions (hour and day) back into time ready for histogram
                data = assert_data_shape_24(data,reverse=True) 

            # now multiply data by conversion factor according to desired untis
            # TO DO: Should expand upon this in reference files
            data *= {"SED":0.9, "J m-2":90, "UVIh":1, "UVI":1, "W m-2":0.025, "mW m-2":25}[self.units]

            # if this is the first iteration, declare a hist
            if i == 0 :
                # seems like useful metadata to know bin n and edges
                # TO DO: reconsider where this belongs in the code (__init__?)
                self.num_bins = int(2*np.nanmax(data) // self.bin_width)
                self.bin_edges = np.array(range(self.num_bins+1)) * self.bin_width
                self.bin_centers = self.bin_edges[:-1] + 0.5 * np.diff(self.bin_edges)

                # TO DO: think about possible cases where dimensions could differ
                self.pix_hist=np.empty([self.num_bins,
                    np.shape(data)[-2],np.shape(data)[-1]], dtype=np.int16)

                # TO DO: this should also be done by some initial dataset analysis
                self.lat = dataset['lat'][:]
                self.lon = dataset['lon'][:]

            # TO DO: add check here in case max exceeds current bin edges
            # now put data into hist using apply_along_axis to perform histogram for each pixel
            def hist_raw(x) :
                hist, _ = np.histogram(x,bins=self.bin_edges)
                return hist

            print("   Calculating and adding to pix hist")
            self.pix_hist[:,:,:] += np.apply_along_axis(hist_raw,0,data)

        return self

    def calculate_map(self,pix_hist=None,statistic=None,bin_centers=None) :
        """calculate_map calculates statistical descriptor values for pixel histograms to produce a map

        This function interprets the statistic string, which can either be a simple command
        such as "mean" or a more advanced formula of keywords. The corresponding function is 
        applied to each pixel of the pix_hist object within the ExposureMap class, essentially
        removing the first dimension and resulting in straightforward map to be plotted.

        Parameters
        ----------
        pix_hist : array, optional
            A 3D array with the first dimension containing vectors of counts for histograms
            and the next two dimensions serving as pixel coordinates. See 
            ExposureMap.collect_data for more information.

        stastistic : str, optional
            If this string is equivalent to one of the following (case insensitive), then the
            corresponding function will be applied; "mean", "median" or "med", "sd" or "std" 
            or "stdev", "max" or "maximum", "min" or "minimum".

            *Planned:* the string can otherwise be a formula using any of the keywords above,
            as well at "prct" or "percentile" preceeded by a number between 0 and 100, and 
            basic mathematical operators (+, -, *, /, **) and numeric factors.
            
        bin_centers : array, optional
            The central numeric values corresponding to the bins in pix_hist

        Returns
        -------
        ExposureMap
            The ExposureMap class object is appended with a map field containing a 2D array
        """

        if not (pix_hist is None) :
            self.pix_hist = pix_hist
        if not (statistic is None) :
            self.statistic = statistic
        if not (bin_centers is None) :
            self.bin_centers = bin_centers

        # Begin by defining the easy options that only require two inputs
        basic_descriptor_functions = {
            "mean": hist_mean,
            "median": lambda x,y: hist_percentile(x,y,0.5),
            "med": lambda x,y: hist_percentile(x,y,0.5),
            "sd": hist_stdev,
            "std": hist_stdev,
            "stdev": hist_stdev,
            "max": hist_max,
            "maximum": hist_max,
            "min": hist_min,
            "minimum":hist_min
        }
        # we can check if the chosen statistic is basic or advanced
        if self.statistic.lower() in basic_descriptor_functions.keys() :
            # in this case, we can simply select the basic function from the dict...
            descriptor_function = basic_descriptor_functions[self.statistic.lower()]
            # ...and execute it across the map
            self.map = np.apply_along_axis(lambda x: descriptor_function(x,self.bin_centers),0,self.pix_hist)

        else :
            # TO DO: interpret self.statistic to build advanced functions (y i k e s)
            print("WARNING: ExposureMap.statistic not recognised.")
        

        return self

    def plot_map(self,map_options=None) :
        """plot_map renders and saves a map of the calculated quantity

        This function caps off the typical workflow for the ExposureMap class by rendering the contents
        of the map property. Many aesthetic factors are accounted for, contained within the
        ExposureMap.map_options dictionary.

        Parameters
        ----------
        map_options : dict, optional
            A collection of many typical options such as image and font sizes, colormaps, etc.
        """

        if not (map_options is None) :
            self.map_options = map_options

        # TO DO: Add custom sizing and resolution specifications
        fig = plt.figure(figsize=(self.map_options['size'][0]/2.54,self.map_options['size'][1]/2.54))

        # TO DO: Accept custom projections
        proj = ccrs.Mercator()

        # TO DO: Add support for multiple plots per figure (too complex? consider use cases)
        ax = fig.add_subplot(1,1,1,projection = proj)

        # TO DO: Increase flexibility of borders consideration
        if self.map_options['brdr_nation'] :
            ax.add_feature(cfeat.BORDERS)

        # TO DO: Consider first-last versus min-max - how can we avoid accidentally flipping images
        extents=[self.lon[0],self.lon[-1],self.lat[0],self.lat[-1]]
        ax.set_extent(extents)

        # Confusingly, this code correctly translate the lat/lon limits into the projected coordinates
        extents_proj = proj.transform_points(ccrs.Geodetic(),np.array(extents[:2]),np.array(extents[2:]))
        extents_proj = extents_proj[:,:2].flatten(order='F')

        # TO DO: Custom colormaps, interpolation, cropping
        im = ax.imshow(self.map,extent=extents_proj,transform=proj,origin='lower',
            cmap=self.mapoptions['cmap'],interpolation='bicubic')

        # TO DO: Add support for horizontal
        if self.map_options['cbar'] :
            cb = plt.colorbar(im, ax=ax, orientation='horizontal',pad=0.05,fraction=0.05)
            cb.ax.set_xlabel(self.units)

        # TO DO: Add plot title, small textbox description, copyright from dataset, ticks and gridlines
        if self.map_options['save'] :
            # Generate timestamp filename if relying on default
            if self.map_options['img_filename']=="default" :
                img_filename=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            plt.savefig(self.map_options['img_dir']+img_filename+"."+self.map_options['img_filetype'],
                bbox_inches="tight",dpi=self.map_options['dpi'])

        plt.show()


