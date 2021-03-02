""" TAME-py library """

import numpy as np
import netCDF4 as nc
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import datetime as dt
from .subroutines import *
from .initial_reference import *

class SpecificDoseEstimationTable(pd.DataFrame):
    """SpecificDoseEstimationTable is the class for replicating dosimetry measurements

    High resolution data allows for personal and ambient dose estimation without the need for
    direct measurement. This class is structured like a table with a set of functions to add 
    columns ultimately leading to dose estimates. Each row of this table represents a specific
    exposure instance, i.e. an individual at a specific location for a specific date and time
    with a specific exposure ratio. The open access article at doi.org/10.3390/atmos12020268
    focuses on calculations appropriate for this class.

    Attributes:
        nc_filename_format (str): Describes the filename of the netCDF files containing the UV 
            data with 'yyyy' in place of the year.
        data_directory (str): The directory where the data is stored. Must end with a slash.
    """

    # This property ensures that functions return the same subclass
    @property
    def _constructor(self):
        return SpecificDoseEstimationTable
    
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
            """schedule_constant_exposure_iter iterates through rows of a SpecificDoseEstimation table to generate schedules.

            This function is designed to be applied to each row in a datatable to generate an
            exposure schedule based on a start time and end time

            Args:
                Start_time (datetime.time): UTC time at which exposure period begins
                End_time (datetime.time): UTC time at which exposure period ends

            Returns:
                numpy.array: 24 length vector of values between 0 and 1 indicating proportion 
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
        self["Schedule"] = self.apply(lambda x: schedule_constant_exposure_iter(x["Time_start"],x["Time_end"]),axis='columns')
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

        Args:
            input_table (pandas.DataFrame): A Pandas DataFrame containing columns for Date, Anatomic_zone, and Posture.
                The Date column should contain DateTime entries. The Anatonic_zone column should contain one string per 
                row describing the exposed body part. The Posture column should contain one string per row describing 
                one of six accepted postures.
            Vis_table_path (str, optional): The full path to an alternative table for the Vis parameter. 
                Must be a csv file. Defaults to None.
            Vis_table (pandas.DataFrame, optional): An alternative table for the Vis parameter. Defaults to None.

        Returns:
            SpecificDoseEstimation: Returns input table appended with ER column
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
        """calculate_specific_dose calculates doses according to exposure schedule, ER, date, and location.

        This function takes the SpecificDoseEstimationTable and calculates the specific ambient and 
        personal doses according to the exposure schedule and ER.

        Returns:
            [type]: [description]
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
            unique_days,unique_days_idx = np.unique(pd.DatetimeIndex(temp_table.Date).dayofyear,return_inverse=True)
            temp_table['unique_days_idx'] = unique_days_idx
            time_subset = [False for i in range(dataset.dimensions['time'].size)] 
            time_subset = assert_data_shape_24(time_subset) 
            time_subset[:,unique_days] = True 
            time_subset = time_subset.flatten(order='F') # This is 'Fortran' ordering
            data = assert_data_shape_24(dataset['UV_AS'][time_subset,:,:]*40) # *40 converts it to uv index
            # convert lat lon into pixel coordinates
            lat = dataset['lat'][:]
            lon = dataset['lon'][:]
            temp_table['pixel_lat'] = temp_table.apply(lambda x: find_nearest(lat,x['Latitude']),axis='columns')
            temp_table['pixel_lon'] = temp_table.apply(lambda x: find_nearest(lon,x['Longitude']),axis='columns')
            # calculate doses
            temp_table['Ambient_dose'] = temp_table.apply(lambda x: 
                np.sum(data[:,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] * x['Schedule']),axis='columns')
            temp_table['Personal_dose'] = temp_table.apply(lambda x: 
                np.sum(data[:,x['unique_days_idx'],x['pixel_lat'],x['pixel_lon']] * x['Schedule'] * x['ER']),axis='columns')
            # extra step necessary to ensure correct assignment
            self.loc[temp_table.index,'Ambient_dose'] = temp_table['Ambient_dose'].values
            self.loc[temp_table.index,'Personal_dose'] = temp_table['Personal_dose'].values
        self['Ambient_dose'] = self['Ambient_dose']/40*3600/100 # SED
        return self        


















class ExposureMap:
    """ ExposureMap is a class for calculating maps based on user specifications

    Each instance of this class contains information required to calculate and illustrate a map
    of exposure information, be that simple averages or more advanced mathematical representations
    of exposure risk.
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
            self.bin_wdith = bin_width
    
    def calculate_pix_hist(self) :
        # first we read the data_directory to check the total number of unique years available
        data_dir_contents = os.listdir(self.data_directory)
        char_year = self.nc_filename_format.find('yyyy')
        dataset_years = [ int(x[char_year:char_year+4]) for x in data_dir_contents ]

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
            # TO DO: check units of dataset, CF conventions for UVI or W/m2
            # now to calculate doses if requested
            if self.units in ["SED","J m-2","UVIh"] :
                # if calculating doses
                print('   Calculating doses')
                data = effective_dose(self.exposure_schedule,data)
            elif self.exposure_schedule != np.ones(24) :
                # assume elsewise calculating intensity (i.e. UV-index) then limit data selection according
                # to schedule (remembering that default schedule is just ones)
                print('   Slicing data with exposure schedule')
                data = assert_data_shape_24(data)
                data *= np.reshape(self.exposure_schedule,[24,1,1,1])
                data = assert_data_shape_24(data,reverse=True)
            # now multiply data by conversion factor according to desired untis
            # TO DO: Should expand upon this in reference files
            data *= {"SED":0.9, "J m-2":90, "UVIh":1, "UVI":1, "W m-2":0.025, "mW m-2":25}[self.units]
            # if this is the first iteration, declare a hist
            if i == 0 :
                # seems like useful metadata to know bin n and edges
                self.num_bins = int(2*np.nanmax(data) // self.bin_width)
                # TO DO: reconsider where this belongs in the code (__init__?)
                self.bin_edges = np.array(range(self.num_bins+1)) * self.bin_width
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

    def calculate_map(self) :
        """ This function must determine how to calculate statistics based on histograms to make a map
        """

        self.map = np.apply_along_axis(lambda x: hist_mean(x,self.bin_edges),0,self.pix_hist)

        return self

    def plot_map(self,img_dir = "", img_filename = "default",img_filetype = "png") :
        """plot_map renders and saves a map of the calculated quantity

        This function caps off the typical workflow for the ExposureMap class by rendering the contents
        of the map property. Many aesthetic factors are accounted for, mostly contained within the
        ExposureMap.map_options dictionary
        """

        # TO DO: Add custom sizing and resolution specifications
        fig = plt.figure(figsize=(20/2.54,15/2.54))
        # TO DO: Accept custom projections
        proj = ccrs.Mercator()
        # TO DO: Add support for multiple plots per figure (too complex? consider use cases)
        ax = fig.add_subplot(1,1,1,projection = proj)
        # TO DO: Increase flexibility of borders consideration
        ax.add_feature(cfeat.BORDERS)
        # TO DO: Consider first-last versus min-max - how can we avoid accidentally flipping images
        extents=[self.lon[0],self.lon[-1],self.lat[0],self.lat[-1]]
        ax.set_extent(extents)
        # Confusingly, this code correctly translate the lat/lon limits into the projected coordinates
        extents_proj = proj.transform_points(ccrs.Geodetic(),np.array(extents[:2]),np.array(extents[2:]))
        extents_proj = extents_proj[:,:2].flatten(order='F')
        # TO DO: Custom colormaps, interpolation, cropping
        im = ax.imshow(self.map,extent=extents_proj,transform=proj,origin='lower',cmap='jet',interpolation='bicubic')
        # TO DO: Add support for horizontal
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',pad=0.05,fraction=0.05)
        cb.ax.set_xlabel(self.units)
        # TO DO: Add plot title, small textbox description, copyright from dataset, ticks and gridlines
        if img_filename=="default" :
            img_filename=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        plt.savefig(img_dir+img_filename+"."+img_filetype,bbox_inches="tight",dpi=300)
        plt.show()














def hist_mean(counts,bin_centers) :
    """hist_mean calculates the mean of a histogram using numpy functions

    This function is designed to calculate a histogram mean as efficiently as possible

    Args:
        counts (array): 1d array (length n) of integers describing counts of histogram
        bin_edges (array): 1d array (length n+1) describing bin edges of histogram
    """

    n = np.sum(counts,0)

    mean = np.dot(counts, bin_centers) / n

    return mean


def hist_percentile(counts,bin_centers,prct) :
    """hist_percentile calculates percentiles of histogram data

    This function takes discretised data, typical for histograms, and calculates
    the user-specified percentile quantity. 

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers ([type]): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.
        prct (float): A fraction betwee 0.0 and 1.0 for the desired percentile.

    Returns:
        float: The desired percentile value. In cases where the quantity falls 
            between two bins, their respective central values are averaged.
    """

    n = np.sum(counts,0)
    cumcounts = np.cumsum(counts)
    i = len(cumcounts[cumcounts<=n*prct]) 
    #this feels like cheating but damn it's quick, faster than np.sortedsearch
    j = len(cumcounts[cumcounts<n*prct]) 
    if i==j :
        percentile = bin_centers[i]
    else :
        percentile = (bin_centers[i] + bin_centers[j])/2
    return percentile   
















class HistogramsTable(pd.DataFrame) :
    """DoseHistogramsTable is a class for generating tables of histograms of doses and UVI values

    This class is designed generate tables of histograms of UV doses or UVI values according to
    conditions set by the user. This includes both point and area location specifications, and 
    flexible date and time specifications. Exposure schedules and ER calculations are also
    possible.
    """

    # This property ensures that functions return the same subclass
    @property
    def _constructor(self):
        return HistogramsTable
    
    # This adds some useful metadata (self-explanatory)
    _metadata = ["nc_filename_format","data_directory","bin_edges"]
    nc_filename_format = 'UVery.AS_ch02.lonlat_yyyy01010000.nc'
    data_directory = 'C:/Data/UV/' 
    bin_edges = np.linspace(0,20,21)



















def effective_dose(schedule, data) :
    """Calculate the effective dose based on a 24 hour exposure schedule

    Args:
        schedule (array): a vector of 24 exposure ratios (typically between 0 and 1) corresponding 
            to each hour of the day (UTC)

        data (array): Data for which to integrate exposure - first dimension must be either length
            24 or divisible by 24 to be reshaped as such
    
    Returns:
        EDD: Effective Daily Dose, an array of equivalent size to the input data without the first 
            dimension
    """

    # TO DO: further investigate how nans are handled
    #        if nan outside of schedule, it should be fine but probably isn't
    data = assert_data_shape_24(data)

    EDD = np.sum(np.reshape(schedule,[24,1,1,1]) * data,axis=0)

    return EDD

def multi_file_dose_histogram(yearsmonths,
schedule = np.ones(24),
data_directory = 'C:/Data/UV/',
filename_format = 'UVery.AS_ch02.lonlat_yyyy01010000.nc',
bin_edges = range(0,51)) :
    """Calculate a histogram of the effective dose across multiple years
    
    Args:
        yearsmonths (pandas DateTimeIndex): 

    Returns:
        histogram
    """

    # Find the unique years in the "yearsmonths" datetime object
    list_of_years = sorted(set(yearsmonths.year))
    # Declare empty histogram
    hist=np.empty([len(list_of_years), len(bin_edges)-1], dtype=int)

    for i in range(3):
        year = list_of_years[i]
        print("Processing year "+str(year)) #should use logging, don't yet know how
        dataset=nc.Dataset(data_directory+filename_format.replace('yyyy',str(year))) 
        dataset.set_auto_mask(False) #to get normal arrays (faster than default masked arrays)
        # dimension_names = dataset.variables['UV_AS'].dimensions # Need to introduce way to read metadata and determine time, x,y dimensions
        # Next we pull a subset from the netCDF file
        # declare false array with same length of time dimension from netCDF
        time_subset = [False for i in range(dataset.dimensions['time'].size)] 
        # reshape false array to have first dimension 24 (hours in day)
        time_subset = assert_data_shape_24(time_subset) 
        # set the appropriate days as true
        time_subset[:,yearsmonths[yearsmonths.year == year].dayofyear-1] = True 
        # flatten time_subset array back to one dimension
        time_subset = time_subset.flatten(order='F')
        # load subset of data
        data = dataset['UV_AS'][time_subset,:,:]*40 # need to generalise for different variables and units
        # apply schedule calculation to data
        EDD = effective_dose(schedule,data)
        # put data into histogram
        partial_hist, _ = np.histogram(EDD,bins=bin_edges)
        # put partial hist into overall hist
        hist[i,:]=partial_hist

    return hist

   
