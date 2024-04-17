from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import datetime as dt
import os
import re
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from cartopy.io import shapereader
from scipy.interpolate import interp2d
from scipy.ndimage import zoom
from .subroutines import *
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import shapefile as shp
from pyproj import Transformer
import xarray as xr 




class ExposureMap:
    """ A class for calculating maps based on user specifications

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

    exposure_schedule : array
        A vector of values describing the relative exposure of each hour of the day. 0 indicates no
        exposure, 1 indicates full exposure, and a fractional value such as 0.5 would indicate 
        exposure for a total of 30 minutes within the hour, or a 50% partial exposure for the full
        hour, or anything equivalent. Values greater than 1 are allowed. Must have a length of 24 (for 
        each hour of the day) although a length of 1 is also allowed, in which case that number will 
        be immediately replicated across a 24-length vector. When not calculating doses, hours with 
        any non-zero entry in this vector are included, with the irradiance values being 
        multiplied by the corresponding non-zero value in the exposure schedule.

    bin_width : float
        The width of each bin in the pixel histogram. Value assumed to be in the same units as 
        defined by the units parameter. *Making bin_width excessively small can lead to high
        memory usage,* consider the underlying accuracy of the source data and be sure not to
        substantially exceed its precision with this parameter.

    statistic : str
        The statistical descriptor to be calculated from the pixel histograms to be later 
        represented on the rendered map. Must contain at least one of these keywords:
        "mean", "median" or "med", "sd" or "std" or "stdev", "max" or "maximum", "min" or 
        "minimum". 

        *Planned:* the string can be a formula using any of the keywords above,
        as well at "prct" or "percentile" preceeded by a number between 0 and 100, and 
        basic mathematical operators (+, -, *, /, **) and numeric factors.

    date_selection : list of dates
        The dates for which the irradiances are retrieved or the daily doses are calculated. 
        Defaults to None whereby the program selects all data within the src_directory that
        matches the src_filename_format.

    src_filename_format : str
        Describes the filename of the netCDF files containing the data with 'yyyy' in place 
        of the year.
    
    src_directory : str
        The directory where the data is stored. Must end with a slash.

    box : list
        A box defining the min and max latitude-longitude of a canton.
        Used to select a spatial cut in the UV data.
        If not passed as argument, consider a box enclosing all Switzerland.

    Example
    -------

    The code below shows a typical use case for the ExposureMap class. The long-term average daily doses
    (i.e. the chronic doses) for typical school children are calculated across Switzerland asssuming certain
    hours of exposure for journeying to and from school and having breaks for morning tea and lunch time. ::

        import python_tamer as pt
        import pandas as pd
        import numpy as np
        src_directory = 'C:/enter_your_src_directory_here'
        ER = pt.ER_Vernez_2015("Forehead","Standing") # Long-term average ER for foreheads in standing posture
        kt_folder = '/home/lmartinell/Desktop/Tamer-git-project/Meteoswiss/python-TAMER/test/'
        kt_box, kt_lon, kt_lat = pt.select_box_canton('VS', kt_folder)
        map = pt.ExposureMap(
            src_directory=src_directory,
            units = "J m-2",
            exposure_schedule = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.5, 0.0, 0.5, 0.0,
                                          0.5, 0.5, 0.0, 0.0, 0.5, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0])*ER,
            bin_width = 25,
            date_selection = ["2010-01-01", "2012-12-31"],
            statistic = "mean",
            map_options={"title": "Chronic daily UV dose for typical school children, 2005-2014",
                        "save": False},
            box = kt_box)
        map = map.collect_data().calculate_map()
        map.plot_map()


    """

    def __init__(self,units= "SED",
    exposure_schedule      = 1,
    statistic              = "mean",
    bin_width              = None,
    date_selection         = None,
    map_options            = None,
    box                    = [], # option to implement spatial selection of data
    src_filename_format    = 'UVery.AS_ch02.lonlat_yyyy01010000.nc',
    src_directory          = 'C:/Data/UV/'):
        # assigning options to fields in class with a few basic checks
        self.units = units

        self.exposure_schedule=np.array(exposure_schedule)
        if len(np.atleast_1d(self.exposure_schedule)) == 1 :
            self.exposure_schedule = np.repeat(self.exposure_schedule,24)

        self.statistic = statistic

        self.map_options = {
            "title"            : "Test map",
            "save"             : True,
            "img_size"         : [20,15],
            "img_dpi"          : 300,
            "img_dir"          : "",
            "img_filename"     : "timestamp",
            "img_filetype"     : "png",
            "brdr_nation"      : True,
            "brdr_nation_rgba" : [0,0,0,0],
            "brdr_state"       : False,
            "brdr_state_rgba"  : [0,0,0,0.67],
            "cmap"             : "jet",
            "cmap_limits"      : None,
            "cbar"             : True,
            "cbar_limits"      : None
        }
        if map_options is not None :
            self.map_options.update(map_options)

        self.src_filename_format = src_filename_format
        self.src_directory = src_directory

        self.date_selection = date_selection

        if bin_width is None :
            self.bin_width = {
                "SED"    : 0.1, 
                "J m-2"  : 10, 
                "UVI"    : 0.1, 
                "W m-2"  : 0.0025, 
                "mW m-2" : 2.5
            }[self.units]
        else :
            self.bin_width = bin_width

        self.box = box
        
    
    def collect_data(self, 
                     src_directory       = None,
                     src_filename_format = None,
                     date_selection      = None,
                     units               = None,
                     exposure_schedule   = None,
                     bin_width           = None,
                     box                 = None) :
        """Loads and manipulates data into histograms for each pixel of the underlying data

        In order to handle large amounts of data without exceeding memory limitations, files are
        loaded one at a time and the time dimension is removed, either by calculating daily doses
        or by simply taking the data as is. The resulting information is then stored not as a 
        list of specific values but rather binned into a histogram for each pixel. This process
        is repeated for each file required by the user input, building up the pixel histograms
        with more information that does not require additional memory.


        Parameters
        ----------

        src_filename_format : str
            Describes the filename of the netCDF files containing the data with 'yyyy' in place 
            of the year.Nonef dates
            The dates for which the irradiances are retrieved or the daily doses are calculated. 
            Defaults to None whereby the program selects all data within the src_directory that
            matches the src_filename_format.

        units : str
            Name of units of desired output. This also indicates whether daily doses must be 
            calculated or not. Units of "SED", "J m-2", or "UVIh" will produce daily doses,
            units of "UVI", "W m-2" or "mW m-2" will not. 

        exposure_schedule : array
            A vector of values describing the relative exposure of each hour of the day. 0 indicates no
            exposure, 1 indicates full exposure, and a fractional value such as 0.5 would indicate 
            exposure for a total of 30 minutes within the hour, or a 50% partial exposure for the full
            hour, or anything equivalent. Values greater than 1 are allowed. Must have a length of 24 (for 
            each hour of the day) although a length of 1 is also allowed, in which case that number will 
            be immediately replicated across a 24-length vector. When not calculating doses, hours with 
            any non-zero entry in this vector are included, with the irradiance values being 
            multiplied by the corresponding non-zero value in the exposure schedule.

        bin_width : float
            The width of each bin in the pixel histogram. Value assumed to be in the same units as 
            defined by the units parameter. *Making bin_width excessively small can lead to high
            memory usage,* consider the underlying accuracy of the source data and be sure not to
            substantially exceed its precision with this parameter.
        
        box : array
            Defines the minimum and maximum latitude and longitudes enclosing a selected territory 
            (i.e a canton) in order to perform a spatial cut of the UV data


        Returns
        -------

        python_tamer.ExposureMap
            The input ExposureMap object is appended with new fields, `pix_hist` contains
            the counts for the histogram, and `bin_edges`, `bin_centers`, and `num_bins`
            all serve as metadata for the pixel histograms. `lat` and `lon` are also 
            added from the multi-file dataset to inform the pixel locations for map making
            further down the typical pipeline.
        
        
        Example
        -------

        The example code below shows how an ExposureMap class can be declared with the default parameters that
        can then be later redefined by collect_data() and the other class functions. ::

            import python_tamer as pt
            import pandas as pd
            import numpy as np
            src_directory = 'C:/enter_your_src_directory_here'
            ER = pt.ER_Vernez_2015("Forehead","Standing") # Long-term average ER for foreheads in standing posture
            map = pt.ExposureMap()
            map = map.collect_data(
                src_directory=src_directory,
                units = "J m-2",
                exposure_schedule = np.array([0  ,0  ,0  ,0  ,0  ,0  ,
                                     0  ,0  ,0.5,0  ,0.5,0  ,
                                     0.5,0.5,0  ,0  ,0.5,0  ,
                                     0  ,0  ,0  ,0  ,0  ,0  ])*ER,
                bin_width = 25,
                date_selection = pd.date_range(start="2005-01-01",end="2014-12-31")
            )
            map = map.calculate_map(statistic = "mean")
            map.plot_map(map_options={"title": "Chronic daily UV dose for typical school children, 2005-2014",
                                      "save": False})


        """
        print("I'm using the git version of Exposure map py")
        params = {'src_directory'      : src_directory,
                  'src_filename_format': src_filename_format,
                  'date_selection'     : date_selection,
                  'units'              : units,
                  'exposure_schedule'  : exposure_schedule,
                  'bin_width'          : bin_width,
                  'box'                : box}

        for param, value in params.items():
            if value is not None:
                setattr(self, param, value)

        # first we read the src_directory to check the total number of unique years available
        data_dir_contents = os.listdir(self.src_directory)
        # TODO: improve jankiness of this format-matching search for filenames
        char_year = self.src_filename_format.find('yyyy')
        dataset_years = [ x for x in data_dir_contents if re.findall(self.src_filename_format.replace("yyyy","[0-9]{4}"),x)]
        dataset_years = [ int(x[char_year:char_year+4]) for x in dataset_years ]

        

        # Now we can handle default options like "all"
        if type(self.date_selection) == str and self.date_selection == "all" :
            date_selection = pd.date_range(start=str(dataset_years[0])+"-01-01",
                end=str(dataset_years[-1])+"-12-31")
        else :
            date_selection = self.date_selection # TODO: much more interpretation options here

        #now we find unique years 
        list_of_years = sorted(set(date_selection.year))

        # Condition on regional box for spatial slicing of data
        if box is not None:
            # use box to define minimum values of latitude and longitude
            # enclosing the selected territory  
            min_lon = box[0]  
            min_lat = box[1]
            max_lon = box[2]
            max_lat = box[3]
        else:
            min_lon = 0


        for i in range(len(list_of_years)) :
            year = list_of_years[i]
            print("Processing year "+str(year)) #should use logging, don't yet know how
            filename = self.src_directory+self.src_filename_format.replace('yyyy',str(year))
            # xarray facilititate the data slicing, 
            dataset = xr.open_dataset(filename)
            '''
            # deprecated method for time slicing
            dataset=nc.Dataset(filename) 
            dataset.set_auto_mask(False) #to get normal arrays (faster than default masked arrays)
            if len(dataset['time']) == 24 :
                # needed if just a single day
                time_subset = [True for i in range(len(dataset['time']))]
            else :
                # Next we pull a subset from the netCDF file
                # declare false array with same length of time dimension from netCDF
                time_subset = [False for i in range(len(dataset['time']))] 
                # reshape false array to have first dimension 24 (hours in day)
                time_subset = assert_data_shape_24(time_subset) 
                # set the appropriate days as true
                time_subset[:,date_selection[date_selection.year == year].dayofyear-1] = True 
                # flatten time_subset array back to one dimension
                time_subset = time_subset.flatten(order='F')
            '''
             
            # load subset of data
            print("Slicing netcdf data with time subset and -if selected- also spatial subset")
            #work in UVI by default because it's easy to read
            if box is not None:   
                data = dataset.sel(time = slice(date_selection[0], date_selection[1]),
                                   lat  = slice(min_lat, max_lat),
                                   lon  = slice(min_lon, max_lon)).UV_AS.to_numpy()
                
            else:
                data = dataset.sel(time = slice(date_selection[0], date_selection[1])).UV_AS.to_numpy()

            # TODO: check units of dataset files, CF conventions for UVI or W/m2

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
            # TODO: Should expand upon this in reference files
            data *= {"SED":0.9, "J m-2":90, "UVIh":1, "UVI":1, "W m-2":0.025, "mW m-2":25}[self.units]

            # if this is the first iteration, declare a hist
            if i == 0 :
                # seems like useful metadata to know bin n and edges
                # TODO: reconsider where this belongs in the code (__init__?)
                self.num_bins = int(np.nanmax(data) // self.bin_width ) + 2
                self.bin_edges = (np.array(range(self.num_bins+1)) - 0.5) * self.bin_width 
                # this form allows for weird custom bin edges, but probably will never use that
                self.bin_centers = self.bin_edges[:-1] + 0.5 * np.diff(self.bin_edges)

                # TODO: think about possible cases where dimensions could differ
                self.pix_hist=np.zeros([self.num_bins,
                    np.shape(data)[-2],np.shape(data)[-1]], dtype=np.int16)

                # TODO: this should also be done by some initial dataset analysis, but that's a drastic
                # design overhaul
                if min_lon > 0:
                    self.lat = dataset['lat'][np.argmin(np.absolute(dataset.variables['lat'][:] - min_lat)):\
                                              np.argmin(np.absolute(dataset.variables['lat'][:] - max_lat))]
                    self.lon = dataset['lon'][np.argmin(np.absolute(dataset.variables['lon'][:] - min_lon)):\
                                              np.argmin(np.absolute(dataset.variables['lon'][:] - max_lon))]
                else:
                    self.lat = dataset['lat'][:]
                    self.lon = dataset['lon'][:]
            else :
                new_num_bins = int(np.nanmax(data) // self.bin_width) + 2 - self.num_bins
                # check if new data requires extra bins in pix_hist
                if new_num_bins > 0 :
                    # append zeros to pix hist to make room for larger values
                    self.pix_hist = np.concatenate((self.pix_hist,np.zeros(
                        [new_num_bins,np.shape(self.pix_hist)[-2],np.shape(self.pix_hist)[-1]],
                        dtype=np.int16)),axis=0)
                    # update bin information
                    self.num_bins = self.num_bins + new_num_bins
                    self.bin_edges = (np.array(range(self.num_bins+1)) - 0.5) * self.bin_width 
                    self.bin_centers = self.bin_edges[:-1] + 0.5 * np.diff(self.bin_edges)

            # TODO: Add check in case bins get "full" (i.e. approach int16 max value)
            # now put data into hist using apply_along_axis to perform histogram for each pixel
            print("   Calculating and adding to pixel histograms")
            self.pix_hist[:,:,:] += np.apply_along_axis(lambda x: 
                np.histogram(x,bins=self.bin_edges)[0],0,data)

        return self

    def calculate_map(self,pix_hist=None,statistic=None,bin_centers=None) :
        """Calculates statistical descriptor values for pixel histograms to produce a map

        This function interprets the statistic string, which can either be a simple command
        such as "mean" or a more advanced formula of keywords. The corresponding function is 
        applied to each pixel of the pix_hist object within the ExposureMap class, essentially
        removing the first dimension and resulting in straightforward map to be plotted.


        Parameters
        ----------

        pix_hist : array
            A 3D array with the first dimension containing vectors of counts for histograms
            and the next two dimensions serving as pixel coordinates. See 
            `ExposureMap.collect_data()` for more information.

        statistic : str
            The statistical descriptor to be calculated from the pixel histograms to be later 
            represented on the rendered map. Must contain at least one of these keywords:
            "mean", "median" or "med", "sd" or "std" or "stdev", "max" or "maximum", "min" or 
            "minimum". 

            *Planned:* the string can be a formula using any of the keywords above,
            as well at "prct" or "percentile" preceeded by a number between 0 and 100, and 
            basic mathematical operators (+, -, *, /, **) and numeric factors.
            
        bin_centers : array
            The central numeric values corresponding to the bins in pix_hist. The 
            `ExposureMap.collect_data` function typically calculates these values from the
            given `bin_width` input.


        Returns
        -------
        python_tamer.ExposureMap
            The ExposureMap class object is appended with a map field containing a 2D array


        Example
        -------

        In the example below, the user imports some pre-calculated pixel histograms, thereby
        completing the ExposureMap workflow without using the `ExposureMap.ExposureMap.collect_data()`
        function. This can be useful if the data collection is timely and the user wants to
        produce multiple different maps. Note that the "custom data" used in this example is not
        included in the python-TAMER package, this simply illustrates a unique use-case. ::

            import python_tamer as pt
            # load custom data from an external file (not included)
            from custom_user_data import pix_hist, bin_centers, map_options 
            map = pt.ExposureMap(map_options = map_options)
            map = map.calculate_map(
                statistic = "median", 
                pix_hist = data, 
                bin_centers = bin_centers
            ).plot_map(save = False)
            map.calculate_map(statistic = "max").plot_map(map_options={"save" = False})
            map.calculate_map(statistic = "std").plot_map(map_options={"save" = False})


        """

        if not (pix_hist is None) :
            self.pix_hist = pix_hist
        if not (statistic is None) :
            self.statistic = statistic
        if not (bin_centers is None) :
            self.bin_centers = bin_centers

        # Begin by defining the easy options that only require two inputs
        basic_descriptor_functions = {
            "mean"   : hist_mean,
            "median" : lambda x,y: hist_percentile(x,y,0.5),
            "med"    : lambda x,y: hist_percentile(x,y,0.5),
            "sd"     : hist_stdev,
            "std"    : hist_stdev,
            "stdev"  : hist_stdev,
            "max"    : hist_max,
            "maximum": hist_max,
            "min"    : hist_min,
            "minimum":hist_min
        }
        # we can check if the chosen statistic is basic or advanced
        if self.statistic.lower() in basic_descriptor_functions.keys() :
            # in this case, we can simply select the basic function from the dict...
            descriptor_function = basic_descriptor_functions[self.statistic.lower()]
            # ...and execute it across the map
            self.map = np.apply_along_axis(lambda x: descriptor_function(x,self.bin_centers),0,self.pix_hist)
        # TODO: a loose space could ruin this, need shunting yard algorithm of sorts
        elif self.statistic.lower()[2:] == "prct" or self.statistic.lower()[2:] == "percentile" :
            prct = int(self.statistic[0:1]) / 100
            self.map = np.apply_along_axis(lambda x: hist_percentile(x,self.bin_centers,prct),0,self.pix_hist)
        else :
            # TODO: interpret self.statistic to build advanced functions (y i k e s)
            print("WARNING: ExposureMap.statistic not recognised.")
        

        return self

    def plot_map(self,map_options=None) :
        """Renders and optionally saves a map of the ``map`` field in an ExposureMap object

        This function caps off the typical workflow for the ExposureMap class by rendering the contents
        of the map field. Many aesthetic factors are accounted for, contained within the
        `ExposureMap.map_options` dictionary.


        Parameters
        ----------

        map_options : dict, optional
            A collection of many typical options such as image and font sizes, colormaps, etc.
            The full range of options is listed below with their default values.

            "title" : "Test map" 
            The title to be rendered above the map. Can be left blank for no title. Can be 
            used to inform img_filename

            "save" : True 
            Boolean to declare whether the map should be saved as an image file or not.
            
            "img_size" : [20,15] 
            The size [width,height] of the image in cm.

            "img_dpi" : 300 
            The dots per inch of the saved image.

            "img_dir" : "" 
            The directory for the image to be saved in, leaving it blank should result
            in images being saved in the working directory.

            "img_filename" : "timestamp" 
            The image filename as a string. The default value of "timestamp" is a keyword
            indicating that the function should generate a filename based on the time at
            the moment of the calculation, specified with the format %Y%m%d_%H%M%S_%f 
            which includes millisecond precision.

            "img_filetype" : "png" 
            The image filetype, must be acceptable to `matplotlib.pyplot.savefig()`.

            "brdr_nation" : True 
            Boolean for drawing national borders on the map.

            "brdr_nation_rgba" : [0,0,0,0] 
            The red, green, blue, and alpha values for the national borders.

            "brdr_state" : False 
            Boolean for drawing state borders as defined by Natural Earth dataset.

            "brdr_state_rgba" : [0,0,0,0.67] 
            The red, green, blue, and alpha values for the national borders.

            "cmap" : "jet" 
            The name of the colourmap to be used when rendering the map.

            "cmap_limits" : None 
            The numeric limits of the colourmap. Defaults to None, where the lower
            and upper limits of the plotted data are used as the colourmap limits.

            "cbar" : True 
            Boolean for rendering a colourbar.

            "cbar_limits" : None 
            The numeric limits of the colourbar. Defaults to None, where the lower
            and upper limits of the plotted data are used as the colourbar limits.
        

        Returns
        -------

        python_tamer.ExposureMap
            Returns the ExposureMap object that was input with an updated map_options
            field (if the user has specificied any changes to the default map_options).


        """

        if map_options is not None :
            self.map_options.update(map_options)

        # TODO: Add custom sizing and resolution specifications
        fig = plt.figure(figsize=(self.map_options['img_size'][0]/2.54,
            self.map_options['img_size'][1]/2.54))

        # TODO: Accept custom projections
        proj = ccrs.Mercator()

        # TODO: Add support for multiple plots per figure (too complex? consider use cases)
        ax = fig.add_subplot(1,1,1,projection = proj)

        # TODO: Increase flexibility of borders consideration
        if self.map_options['brdr_nation'] :
            ax.add_feature(cfeat.BORDERS)

        # TODO: Consider first-last versus min-max - how can we avoid accidentally flipping images
        extents=[self.lon[0],self.lon[-1],self.lat[0],self.lat[-1]]
        ax.set_extent(extents)

        # Confusingly, this code correctly translate the lat/lon limits into the projected coordinates
        extents_proj = proj.transform_points(ccrs.Geodetic(),np.array(extents[:2]),np.array(extents[2:]))
        extents_proj = extents_proj[:,:2].flatten(order='F')

        # TODO: Custom colormaps, interpolation, cropping
        im = ax.imshow(self.map,extent=extents_proj,transform=proj,origin='lower',
            cmap=self.map_options['cmap'],interpolation='bicubic')

        # TODO: Add more advanced title interpretation (i.e. smart date placeholder)
        if self.map_options['title'] is not None :
            ax.set_title(self.map_options['title'])

        # TODO: Add support for horizontal
        if self.map_options['cbar'] :
            cb = plt.colorbar(im, ax=ax, orientation='horizontal',pad=0.05,fraction=0.05)
            cb.ax.set_xlabel(self.units)

        # TODO: Add plot title, small textbox description, copyright from dataset, ticks and gridlines
        if self.map_options['save'] :
            # Generate timestamp filename if relying on default
            if self.map_options['img_filename'] == "timestamp" :
                img_filename=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            plt.savefig(self.map_options['img_dir']+img_filename+"."+self.map_options['img_filetype'],
                bbox_inches="tight",dpi=self.map_options['img_dpi'])

        plt.show()

        return self












class ExposureMapSequence :
    """ Class for generating multiple Exposure Maps in a single operation

    The ExposureMapSequence class is a framework for generating multiple maps following
    a given sequence. The basic workflow begins by declaring an object of this class and
    collecting the data from the source NetCDF files. The process is designed with 
    memory efficiency in mind, so data is loaded one year at a time and put into pixel
    histograms akin to the ExposureMap class behaviour. However, in this class we allow for
    multiple histograms to be stored within a single ExposureMapSequence object. Next,
    the maps are calculated by the calculate_maps function. Multiple maps can be calculated
    for each histogram if the user has specified multiple statistics that they want to calculate. 
    Lastly, the maps are rendered and saved by the save_maps function.


    Parameters
    ----------

    src_filename_format : str
        Describes the filename of the netCDF files containing the data with 'yyyy' in place 
        of the year.
    
    src_directory : str
        The directory where the data is stored. Must end with a slash.


    Example
    -------

    In this example, we produce a basic sequence of monthly average doses for 2020::

        example = ExposureMapSequence()
        example = example.collect_data('monthly',year_selection=[2020],units=["SED"])
        example = example.calculate_maps(statistic='Mean')
        example.save_maps(save=True,show=True)

    In this example, we produce a basic sequence of annual average doses for each year
    of the dataset::

        example = ExposureMapSequence()
        example = example.collect_data(['annual'],year_selection=[0],units=["SED"])
        example = example.calculate_maps(statistic='Mean')
        example.save_maps(save=True,show=True)        

    """

    def __init__(self,
    src_filename_format = 'UVery.AS_ch02.lonlat_yyyy01010000.nc',
    src_directory       = 'C:/Data/UV/',
    units               = None,
    bin_width           = None,
    map_options         = None,
    box                 = None
    ):
        # start with data location to quickly get some metadata
        self.src_filename_format = src_filename_format
        self.src_directory = src_directory

        # first we read the src_directory to check the total number of unique years available
        data_dir_contents = os.listdir(self.src_directory)

        # match filename format to find years
        dataset_years = [ x for x in data_dir_contents 
            if re.findall(self.src_filename_format.replace("yyyy","[1-2][0-9]{3}"),x)]

        char_year = self.src_filename_format.find('yyyy')
        self.dataset_years = [ int(x[char_year:char_year+4]) for x in dataset_years ]

        self.bin_width = bin_width
        self.units = units

        # declare an empty dictionary for map options
        self.map_options={}
        # if any input, update dictionary
        if map_options is not None :
            self.map_options = self.map_options.update(map_options)


    def interpret_parameters(self) :
        """Interprets some parameter inputs and adjusts for consistency

        This function will check that parameters are correctly entered and do some 
        basic interpretation.
        It checks the exposure_schedule, year_selection, units, and bin_width input. 
        All input is converted to lists as required.
        """

        if hasattr(self,'exposure_schedule') and self.exposure_schedule is not None :
            if isinstance(self.exposure_schedule,float) :
                self.exposure_schedule = [np.repeat(self.exposure_schedule,24)]

            elif isinstance(self.exposure_schedule,int) :
                temp = self.exposure_schedule
                self.exposure_schedule = [np.zeros(24)]
                self.exposure_schedule[0][temp] = 1

            elif isinstance(self.exposure_schedule,dict) :
                temp = self.exposure_schedule
                self.exposure_schedule = [np.zeros(24)]
                for x in temp.items() :
                    self.exposure_schedule[0][int(x[0])] = x[1]                

            elif isinstance(self.exposure_schedule,np.ndarray) :
                if len(np.shape(self.exposure_schedule)) == 1 and np.shape(self.exposure_schedule)[0] == 24 :
                    self.exposure_schedule = [self.exposure_schedule]
                elif len(np.shape(self.exposure_schedule)) == 2 and np.shape(self.exposure_schedule)[1] == 24 :
                    # split an array of multiple schedules into a list of single schedule arrays
                    self.exposure_schedule = np.split(self.exposure_schedule,np.shape(self.exposure_schedule)[0])
                else :
                    raise ValueError("Exposure schedule not a comprehensible numpy array, " +
                                    "must be length 24 in first or second dimension")

            elif isinstance(self.exposure_schedule,list) :
                if len(self.exposure_schedule) == 24 and all(isinstance(x,(int,float)) for x in self.exposure_schedule) :
                    self.exposure_schedule = [np.array(self.exposure_schedule)]
                
                for i in range(len(self.exposure_schedule)) :
                    if isinstance(self.exposure_schedule[i],float) :
                        self.exposure_schedule[i] = np.repeat(self.exposure_schedule[i],24)

                    elif isinstance(self.exposure_schedule[i],int) :
                        temp = self.exposure_schedule[i]
                        self.exposure_schedule[i] = np.zeros(24)
                        self.exposure_schedule[i][temp] = 1

                    elif isinstance(self.exposure_schedule[i],dict) :
                        temp = self.exposure_schedule[i]
                        self.exposure_schedule[i] = np.zeros(24)
                        for x in temp.items() :
                            self.exposure_schedule[i][int(x[0])] = x[1]   

                    elif isinstance(self.exposure_schedule[i],np.ndarray) :
                        if not (len(np.shape(self.exposure_schedule[i])) == 1 
                                and np.shape(self.exposure_schedule[i])[0] == 24 ):
                            raise ValueError("Exposure schedule list contains an incomprehensible entry, " + 
                                            "a numpy array that is not length 24")
                    
                    elif isinstance(self.exposure_schedule[i],list) :
                        if len(self.exposure_schedule[i]) == 24 :
                            self.exposure_schedule[i] = np.array(self.exposure_schedule[i])
                        else :
                            raise ValueError("Exposure schedule list contains an incomprehensible entry, " + 
                                            "a list that is not length 24")
                    
                    else :
                        raise TypeError("Exposure schedule list contains an incomprehensible entry")

            else :
                raise TypeError("Exposure schedule must be a list of length-24 numpy arrays or similar")
        ######################################################################################################            
        if hasattr(self,'year_selection') and self.year_selection is not None :
            if isinstance(self.year_selection,int) :
                if self.year_selection==0:
                    self.year_selection = [np.array([x]) for x in self.dataset_years]
                else:
                    self.year_selection = [np.array([self.year_selection])]
            elif isinstance(self.year_selection,np.ndarray) :
                if len(np.shape(self.year_selection)) == 1 :
                    self.year_selection = [self.year_selection]
                else :
                    raise ValueError("Year selection should be a list of numpy arrays, " +
                                    "provided numpy array has incomprehensible shape")
            elif isinstance(self.year_selection,list) :
                if all([isinstance(x,int) for x in self.year_selection]) and all(x!=0 for x in self.year_selection) :
                    self.year_selection = [np.array(self.year_selection)]
                else :
                    i=0
                    for k in range(len(self.year_selection)) :
                        if isinstance(self.year_selection[i],int) :
                            if self.year_selection[i] == 0 :
                                temp = self.year_selection[0:i] + [np.array([x]) for x in self.dataset_years]
                                if i != len(self.year_selection)-1 : 
                                    temp = temp + self.year_selection[i+1:]
                                self.year_selection = temp
                                i = i + len(self.dataset_years) - 1
                            else :
                                self.year_selection[i] = np.array([self.year_selection[i]])
                        elif isinstance(self.year_selection[i],list) :
                            self.year_selection[i] = np.array(self.year_selection[i])
                        elif not isinstance(self.year_selection[i],np.ndarray) :
                            raise TypeError("Year selection list must contain ints, lists, or numpy arrays")
                        i=i+1
            else :
                raise TypeError("Year selection must be an int, numpy array, or list of numpy arrays")

            for i in range(len(self.year_selection)) :
                if all(self.year_selection[i] == 0) :
                    self.year_selection[i] = np.array(self.dataset_years)
        #####################################################################################################
        if hasattr(self,'units') and self.units is not None :
            if isinstance(self.units,str) :
                self.units = [self.units]
            elif isinstance(self.units,list) :
                if not all(isinstance(x,str) for x in self.units) :
                    raise TypeError("Units input must be a list of strings")
            else :
                raise TypeError("Units input must be a list of strings")

            for i in range(len(self.units)) :
                if not isinstance(self.units[i],str) :
                    raise TypeError("Units input must be a list of strings")
                if self.units[i] not in ["SED","UVIh","UVI","J m-2","W m-2","mW m-2"] :
                    raise ValueError("Units input must be list of accepted unit strings, " +
                                     "those being SED, UVIh, J m-2, UVI, W m-2, or mW m-2")


            if hasattr(self,'bin_width') :
                if self.bin_width is None :
                    self.bin_width = []
                    for unit in self.units :
                        self.bin_width.append({
                            "SED" : 0.1, 
                            "J m-2" : 10, 
                            "UVI" : 0.1, 
                            "W m-2" : 0.0025, 
                            "mW m-2" : 2.5
                            }[unit])
                elif isinstance(self.bin_width,(int,float)) :
                    self.bin_width = [self.bin_width]


        return self
        



    def collect_data(self,
                     day_selection,
                     exposure_schedule = [1.0],
                     year_selection    = [0],
                     units             = ["SED"],
                     bin_width         = None,
                     box               = None):
        """Loads data into multiple pixel histograms

        This function loads all of the necessary data and compiles it into one or
        multiple histograms. All parameters are designed to be interpreted as lists
        of arrays or lists of strings, where each list entry corresponding to a 
        different histogram in the sequence. So to create a sequence of maps 
        corresponding to the months of the year, the day_selection input would be a
        list of 12 arrays, the first containing numbers from 1 to 31, the second
        containing numbers from 32 to 59, and so on.

        The user specifies the day_selection and the year_selection as two separate 
        numerical inputs, rather than specifying dates. This make the interpretation
        of the sequence simpler. However, to simplify the user experience, the 
        day_selection input can include keywords to be automatically interpreted as
        days of the year.


        Parameters
        ----------
        day_selection : list, str, array
            A list of arrays and/or strings. Keywords interpretable in such a list
            include the (english) names of the 12 months (at least the first three
            letters), the names of the four seasons (fall or autumn is accepted),
            or the words "year" or "annual" to indicate the full year. These 
            keywords are replaced by the corresponding array of days in the year.
            Note that the 29th of February is removed from consideration should it 
            arise. Note also that the seasons are the meteorological seasons, i.e.
            the three month blocks JJA, SON, DJF, and MAM for summer, autumn,
            winter, and spring respectively. 
            
            The user can alternatively enter a special string instead of a list. 
            The string "monthly" generates a list of 12 arrays according to the 
            months whereas the string "seasons" generates a list of 4 arrays 
            according to the four seasons. 

        exposure_schedule : list, float, int, dict, array
            If the user enters a float, this float value will be repeated across a
            length-24 array to make the exposure schedule. For example, entering
            1.0 (not 1) will generate an array of 24 ones.

            If the user enters an int, i, a length-24 array of zeroes will be 
            generated with the ith entry being set to 1. For example, entering 1 
            (not 1.0) will generate an array that reads [0,1,0,0...] (length 24).

            If the user enters a dict, they can specify the values of a few 
            particular entries in a length-24 array where unspecified entries have
            a value of zero. For example, entering {0:0.5, 2:0.8, 3:1} will 
            generate and array the reads [0.5, 0, 0.8, 1, 0...] (length 24).

            If the user enters an array, it must be 1 dimensional with length 24
            or 2 dimensional with the second dimension having length 24 (allowing
            the user to specify multiple schedules).

            If the user enters a list, each entry of that list is interpreted using
            the rules listed above, with the caveat that arrays within a list cannot
            be 2 dimensional.
            

        year_selection : list, array, int
            The years across which the data should be pulled. Input should be a list
            of arrays of ints corresponding to years available in the dataset. Each
            list entry corresponds to a pixel histogram. The user can enter 0 as a
            shortcut for using all available years. For example, an input might be
            [numpy.arange(2010,2020),[0],0]. The first list entry is an array of a
            decade of years, straightforward enough. The second list entry is [0]. 
            This is equivalent to writing numpy.arange(2004,2021) i.e. it produces 
            an array of the available years of the dataset. The last entry is 0,
            this produces a sequence of individual years. So the input could be
            equivalently written as [numpy.arange(2010,2020),numpy.arange(2004,2021),
            numpy.array([2004]),numpy.array([2005]),numpy.array([2006])...] and so
            on until 2020.

        units : list, optional
            The list of units for each pixel histogram. Acceptable strings are "SED",
            "J m-2", "UVIh", "UVI", "W m-2", and "mW m-2". Defaults to SED.

        bin_width : list, optional
            The bin width for each histogram. By default, these values are defined
            automatically according to the units input. 

        box : list
        A box defining the min and max latitude-longitude of a canton.
        Used to select a spatial cut in the UV data.
        If not passed as argument, consider a box enclosing all Switzerland.


        Returns
        -------
        ExposureMapSequence
            The object has the hist_specs and hists fields added detailing the pixel
            histograms.


        Example
        -------

        In this example, we produce a basic sequence of monthly average doses for 2020::

            example = ExposureMapSequence()
            kt_folder = '/home/lmartinell/Desktop/Tamer-git-project/Meteoswiss/python-TAMER/test/'
            kt_box, kt_lon, kt_lat = pt.select_box_canton('VS', kt_folder)
            example = example.collect_data('monthly',year_selection=[2020],units=["SED"], box = kt_box)
            example = example.calculate_maps(statistic='Mean')
            example.save_maps(save=True,show=True)

        In this example, we produce a basic sequence of annual average doses for each year
        of the dataset::

            example = ExposureMapSequence()
            example = example.collect_data(['annual'],year_selection=[0],units=["SED"])
            example = example.calculate_maps(statistic='Mean')
            example.save_maps(save=True,show=True)   

        """

        # this subroutine handles keyword inputs (monthly, seasonal, etc)
        # i.e., if argument is 'monthly', write the days of the year in self.day_selection
        # grouped in 12 arrays   
        self.day_selection, self.day_input_flt, self.day_nonstring = str2daysofyear(day_selection)

        self.exposure_schedule = exposure_schedule # set to [1.0]  
        self.year_selection    = year_selection    # argument of this function 

        if units is not None :
            self.units = units
        
        if bin_width is not None :
            self.bin_width = bin_width

        self = self.interpret_parameters()

        ############################################################################
        # April 2024  
        # Condition on regional box for spatial slicing of data
        if box is not None:
            # use box to define minimum values of latitude and longitude
            # enclosing the selected region of interest  
            min_lon = box[0]  
            min_lat = box[1]
            max_lon = box[2]
            max_lat = box[3]
        else:
            min_lon = 0

        # set number of histograms equal to the max required 
        # i.e. if day_selection = 'monthly' and only 1 year is selected -> num hist = 12  
        lengths = {'day_selection'     : len(self.day_selection),
                   'exposure_schedule' : len(self.exposure_schedule),
                   'year_selection'    : len(self.year_selection),
                   'units'             : len(self.units),
                   'bin_width'         : len(self.bin_width)}
        
        self.num_hists = max(lengths.items(), key=lambda x: x[1])[1]   
        # check if condition is true, else print message 
        assert all(x == self.num_hists or x == 1 for x in lengths.values()), (
            "Inputs must be lists of length 1 or num_hists")
        # iterate over item defining the value of num_hist
        self.iterators = [x[0] for x in lengths.items() if x[1]==self.num_hists]

        # define dictionary with histogram specifications 
        self.hist_specs = []
        for i in range(self.num_hists) :
            hist_spec = {
                'day_selection'     : self.day_selection[0],
                'exposure_schedule' : self.exposure_schedule[0],
                'year_selection'    : self.year_selection[0],
                'units'             : self.units[0],
                'bin_width'         : self.bin_width[0]}
            for x in self.iterators :
                hist_spec[x] = self.__dict__[x][i]
            self.hist_specs = self.hist_specs + [hist_spec]
                  
        # find unique years to be loaded (probably all years but have to check)
        unique_years = set(self.year_selection[0])
        if len(self.year_selection) > 1 :
            for i in range(1,len(self.year_selection)) :
                unique_years.update(self.year_selection[i])
        unique_years = sorted(unique_years)

        # declare empty hists
        self.hists      = [ None for x in range(self.num_hists)]
        # Declare empty vectors to save time-traces with average over lat and long
        self.trace_UV   = [[None for x in range(len(unique_years))] for j in range(self.num_hists)]
        self.trace_days = [ None for x in range(self.num_hists)]

        for i in range(len(unique_years)) :
            # load data year by year 
            year = unique_years[i]
            print("Processing year "+str(year)) #should use logging, don't yet know how
            # dataset=nc.Dataset(self.src_directory+self.src_filename_format.replace('yyyy',str(year))) 
            # dataset.set_auto_mask(False) #to get normal arrays (faster than default masked arrays)

            # use xarray to load data, facilitating data slicing
            filename = self.src_directory+self.src_filename_format.replace('yyyy',str(year))
            dataset  = xr.open_dataset(filename) 

            if i == 0 :
                if box is not None :
                    print('selecting data in the region of interest...')
                    self.lat = dataset.sel(lat  = slice(min_lat, max_lat)).lat.to_numpy()
                    self.lon = dataset.sel(lon  = slice(min_lon, max_lon)).lon.to_numpy()
                else:
                    # old method for netcdf arrays 
                    # self.lat = dataset['lat'][:]
                    # self.lon = dataset['lon'][:]
                    self.lat = dataset.lat.to_numpy()
                    self.lon = dataset.lon.to_numpy()
                

            # now to determine the unique days for the specific year
            unique_days = set()
            for j in range(self.num_hists) :
                if year in self.hist_specs[j]['year_selection'] :
                    unique_days.update(self.hist_specs[j]['day_selection'])
            unique_days = sorted(unique_days)

            # TODO: when metadata fixed, update this to actually interpret dates (cftime)
            # reformat to index for netCDF
            nc_day_sel = [False for i in range(365*24)] 
            # reshape false array to have first dimension 24 (hours in day)
            nc_day_sel = assert_data_shape_24(nc_day_sel) 
            # set the appropriate days as true
            nc_day_sel[:,np.array(unique_days)-1] = True 
            # correct for leap years (skip feb 29)
            if year % 4 == 0 :
                nc_day_sel = np.concatenate(
                    (nc_day_sel[:,0:59],np.full((24,1),False),nc_day_sel[:,59:]),axis=1)
            # flatten time_subset array back to one dimension
            nc_day_sel = nc_day_sel.flatten(order='F')

            # load data :
            # netcdf version 
            # data_year = assert_data_shape_24(dataset['UV_AS'][nc_day_sel,:,:])
            # xarray version
            # define first and last day of interest for data slicing, then convert to numpy arrays
            date_start = DayNumber_to_Date(str(min(unique_days)), str(year))
            if year % 4 == 0 :
                date_end   = DayNumber_to_Date(str(max(unique_days)+1), str(year))
            else :
                date_end   = DayNumber_to_Date(str(max(unique_days)), str(year))

            if box is not None:   
                data_year = assert_data_shape_24(dataset.sel(time = slice(date_start, date_end),
                                                             lat  = slice(min_lat, max_lat),
                                                             lon  = slice(min_lon, max_lon)).UV_AS.to_numpy())
            else: 
                 data_year = assert_data_shape_24(dataset.sel(time = slice(date_start, date_end)).UV_AS.to_numpy())   
            
            #sort data into histograms
            for j in range(self.num_hists) :
                if year in self.hist_specs[j]['year_selection'] :
                    # select days corresponding to the selected month  
                    sub_day_sel = [ True if x in self.hist_specs[j]['day_selection'] 
                        else False for x in unique_days ]
                    # consider data for the given month (each 24 hours) 
                    temp_data = data_year[:,sub_day_sel,:,:]

                    # Apply the exposure schedule, differently for doses vs intensity
                    if self.hist_specs[j]['units'] in ["SED","J m-2","UVIh"] :
                        # if calculating doses
                        print('   Calculating doses')
                        temp_data = np.sum(np.reshape(
                            self.hist_specs[j]['exposure_schedule'],[24,1,1,1]) * temp_data,axis=0)
                    # more complex when doing intensity
                    else :
                        # assume elsewise calculating intensity (i.e. UV-index) then limit data selection
                        # to schedule (remembering that default schedule is just ones)
                        print('   Slicing data with exposure schedule')
                        # select only those hours with nonzero entry in exposure schedule
                        temp_data = temp_data[self.hist_specs[j]['exposure_schedule'] != 0,:,:,:]
                        # select nonzero values from exposure schedule
                        exposure_schedule_nonzero = self.hist_specs[j]['exposure_schedule'][
                            self.hist_specs[j]['exposure_schedule'] != 0]
                        # if any nonzero entries aren't 1, multiply data accordingly
                        if (exposure_schedule_nonzero != 1).any() :
                            temp_data *= np.reshape(exposure_schedule_nonzero,
                                                    [len(exposure_schedule_nonzero),1,1,1])
                        # recombine first two dimensions (hour and day) back into time ready for histogram
                        temp_data = assert_data_shape_24(temp_data,reverse=True) 

                    # now multiply data by conversion factor according to desired untis
                    # TODO: Should expand upon this in reference files
                    temp_data *= {"SED":0.9, "J m-2":90, "UVIh":1, "UVI":1, "W m-2":0.025,
                                  "mW m-2":25}[self.hist_specs[j]['units']]

                    # AI feb 14th 2024 take mean over all latitudes and longitudes
                    self.trace_UV[j][i] = np.nanmean(temp_data,(1,2))
                    self.trace_days[j]  = self.hist_specs[j]['day_selection']
                    
                    # if this is the first iteration, declare a hist
                    if 'num_bins' not in self.hist_specs[j] :
                        # seems like useful metadata to know bin n and edges
                        self.hist_specs[j]['num_bins']  = int(np.nanmax(temp_data) // 
                                                              self.hist_specs[j]['bin_width'] ) + 2
                        self.hist_specs[j]['bin_edges'] = (np.array(range(self.hist_specs[j]['num_bins']+1))
                            - 0.5) * self.hist_specs[j]['bin_width'] 
                        # this form allows for weird custom bin edges, but probably will never use that
                        self.hist_specs[j]['bin_centers'] = (self.hist_specs[j]['bin_edges'][:-1] 
                            + 0.5 * np.diff(self.hist_specs[j]['bin_edges']))

                        # TODO: think about possible cases where dimensions could differ
                        self.hists[j]=np.zeros([self.hist_specs[j]['num_bins'],
                                                np.shape(temp_data)[-2],
                                                np.shape(temp_data)[-1]], 
                                                dtype=np.int16)

                    else :
                        new_num_bins = int(np.nanmax(temp_data) // self.hist_specs[j]['bin_width']) + 2 - self.hist_specs[j]['num_bins']
                        # check if new data requires extra bins in pix_hist
                        if new_num_bins > 0 :
                            # append zeros to pix hist to make room for larger values
                            self.hists[j] = np.concatenate((self.hists[j],np.zeros(
                                [new_num_bins,np.shape(self.hists[j])[-2],np.shape(self.hists[j])[-1]],
                                dtype=np.int16)),axis=0)
                            # update bin information
                            self.hist_specs[j]['num_bins'] = self.hist_specs[j]['num_bins'] + new_num_bins
                            self.hist_specs[j]['bin_edges'] = (np.array(range(self.hist_specs[j]['num_bins']+1))
                                - 0.5) * self.hist_specs[j]['bin_width'] 
                            self.hist_specs[j]['bin_centers'] = (self.hist_specs[j]['bin_edges'][:-1] 
                                + 0.5 * np.diff(self.hist_specs[j]['bin_edges']))

                    # TODO: Add check in case bins get "full" (i.e. approach int16 max value)
                    # now put data into hist using apply_along_axis to perform histogram for each pixel
                    print("   Calculating and adding to pixel histograms")
                    self.hists[j][:,:,:] += np.apply_along_axis(lambda x: 
                        np.histogram(x,bins=self.hist_specs[j]['bin_edges'])[0],0,temp_data)

        return self                    





    def calculate_maps(self,statistic=None,titles=None,filenames="auto") : 
        """Calcualte the maps from the pixel histograms 

        This function calculates maps from the pixel histograms and generates
        titles and filenames for each map. Note that the number of maps can
        be greater than the number of pixel histograms if more than one 
        statistic is specified.


        Parameters
        ----------
        statistic : list, str
            The statistical descriptor to be calculated from the pixel histograms to be later 
            represented on the rendered map. Must contain at least one of these keywords:
            "mean", "median" or "med", "sd" or "std" or "stdev", "max" or "maximum", "min" or 
            "minimum". The keyword "prct" or "percentile" is also accepted so long as it is
            preceded by a two-digit integer specifying the desired percentile from 01 to 99.

            *Planned:* the string can be a formula using any of the keywords above, and 
            basic mathematical operators (+, -, *, /, **) and numeric factors.            

        titles : list, optional
            If the user does not wish to use the automatically generated map titles,
            they can enter them with this parameter. This must be a list of strings
            with a length equal to the number of maps produced.

        filenames : str, optional
            Filenames are generated to match the titles by default, but the user can
            alternatively enter them manually with this parameter.


        Returns
        -------
        ExposureMapSequence
            The object is appended with maps, map_specs, and num_maps fields.


        Example
        -------

        In this example, we produce a basic sequence of annual average doses for each year
        of the dataset::

            example = ExposureMapSequence()
            example = example.collect_data(['annual'],year_selection=[0],units=["SED"])
            example = example.calculate_maps(statistic='Mean')
            example.save_maps(save=True,show=True)   

        """

        if statistic is not None :
            self.statistic = statistic

        if isinstance(self.statistic,str) :
            self.statistic = [self.statistic]
        
        # declare array of nans to fill with maps
        self.maps = np.full([self.num_hists * len(self.statistic)] + 
            list(np.shape(self.hists[0])[1:]),np.nan)

        if titles is not None :
            self.titles = titles
            self.titles_trace = titles

        else :
            self.titles = [str(x) for x in range(self.num_hists * len(self.statistic))]
            self.titles_trace = [str(x) for x in range(self.num_hists * len(self.statistic))]


        if isinstance(filenames,str) and filenames == "auto" :
            self.filenames = [str(x) for x in range(self.num_hists * len(self.statistic))]
            self.filenames_trace = [str(x) for x in range(self.num_hists * len(self.statistic))]

        else :
            self.filenames = filenames
            self.filenames_trace = filenames


        mapnum = 0
        hist_inds = []
        stat_inds = []
        for i in range(len(self.statistic)) :
            for j in range(self.num_hists) :
                
                self.maps[mapnum,:,:] = calculate_map_from_hists(
                    self.hists[j],self.statistic[i],self.hist_specs[j]['bin_centers'])

                if titles is None :
                    if filenames == "auto" :
                        self.titles[mapnum], self.titles_trace[mapnum],self.filenames[mapnum],self.filenames_trace[mapnum] = gen_map_title(**{
                            **self.hist_specs[j],
                            'statistic':self.statistic[i]},filename=True)
                    else :
                        self.titles[mapnum],self.titles_trace[mapnum] = gen_map_title(**{
                            **self.hist_specs[j],
                            'statistic':self.statistic[i]},filename=False)

                hist_inds = hist_inds + [j]
                stat_inds = stat_inds + [i]

                mapnum += 1

        self.num_maps = mapnum

        self.map_specs = {'hist' : hist_inds, 'statistic' : stat_inds}

        return self


    def save_trace(self,save=False,show=True,img_dir='',img_size=[20,15]) : 
        """Shows or saves a plot of the spatially averaged time evolution of UV radiation


        Parameters
        ----------
        save : bool, optional
            Save or just show the image

        show : bool, optional
            An option to show the maps in a python figure window or not.

        img_dir : string, optional
            Directory where image will be saved. Per default saved in the current directory

        img_size : list, optional
            Size of figure

            
         Example
        -------     

        In this example, we produce a spatially averaged time evolution of 
        the annual dose averaged over all available years
            example = ExposureMapSequence()
            example = example.collect_data(['annual'],year_selection=[0],units=["SED"])
            example = example.calculate_maps(statistic='Mean')
            example.save_trace(save=False,show=False)
        """  

        for i in range(self.num_hists):
            plt.figure(figsize=(img_size[0]/2.54,img_size[1]/2.54))
            
            # Convert days to dates
            X = daysofyear2date(self.trace_days[i],self.units[0],self.exposure_schedule)
            
            # If units == UVI we show the daily may and mean
            if self.units[0] == "UVI": 
                # the UVI averaged over all latitudes and longitudes for each hour
                UV_averaged = np.nanmean(list(filter(lambda item: item is not None,self.trace_UV[i])),axis=0)
                nrow = np.size(self.exposure_schedule[0][self.exposure_schedule[0]!=0])
                ncol = int(np.size(UV_averaged)/nrow)
                UV_averaged = np.reshape(UV_averaged,[nrow,ncol],order='F')
                X = np.reshape(X,[nrow,ncol],order='F')

                plt.plot(X[0,:], np.nanmax(UV_averaged,axis=0), 'o',color='green',label="Daily maximum")
                plt.plot(X[0,:], np.nanmean(UV_averaged,axis=0), 'o',color='red',label="Daily mean")
                plt.legend(loc="upper left",frameon=False,fontsize=11) 

            else:
                # If units == SED show averaged over all latitudes and longitudes
                plt.plot(X, np.nanmean(list(filter(lambda item: item is not None,self.trace_UV[i])),axis=0), 'o',color='blue')

            plt.title(self.titles_trace[i])
            plt.ylabel(self.units[0], fontsize=12)  # Add y-axis label with font size
            
            # Limit number of x ticks
            plt.gca().xaxis.set_major_locator(MaxNLocator(20))
            plt.grid(True)
            # Limit date str to month name and day
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            # Rotate x tick labels
            plt.xticks(rotation=45,ha='right')
            plt.ylim([0,1.1*np.nanmax(UV_averaged)])

            if show: 
                plt.tight_layout()
                plt.show()
            
            if save:
                img_filename = self.filenames_trace[i]
                img_filetype = 'png'
                img_dpi = 300
                plt.savefig(img_dir+img_filename+"."+img_filetype,
                        bbox_inches="tight",dpi=img_dpi)        
      
        

    def save_maps(self,map_options=None,save=None,show=True,match_cmap_limits=True,schedule_diagram=True,img_dir = '') :
        """Renders and saves the pre-calculated maps stored in the object

        With the maps calculated, this function renders the maps with broad flexibility on aesthetic 
        options. It is mostly a wrapper for the render_map function.


        Parameters
        ----------
        map_options : dict, optional
            A dictionary containing options for the render map function.

        save : bool, optional
            Although technically contained within map_options, this option is here so users can
            more easily say whether they want the images to be saved or not.

        show : bool, optional
            An option to show the maps in a python figure window or not.

        match_cmap_limits : bool, list, optional
            When producing multiple maps, it can sometimes be desirable for the colormap limits
            to be consistent across the set of images. This boolean enables that.

            can also be a list to fix limits match_cmap_limits[0],match_cmap_limits[1]

        schedule_diagram : bool, optional
            If true, a circular diagram is rendered on the map illustrating the schedule that
            generated the map.

        """

        if map_options is not None :
            self.map_options.update(map_options)

        if save is not None and isinstance(save,bool) :
            self.map_options['save'] = save
        if type(match_cmap_limits) == list:
             self.map_options['cmap_limits'] = [match_cmap_limits[0],match_cmap_limits[1]]
        elif match_cmap_limits:
            self.map_options['cmap_limits'] = [np.nanmin(self.maps),np.nanmax(self.maps)]
            if self.map_options['cmap_limits'][0] < 0.1 * self.map_options['cmap_limits'][1] :
                self.map_options['cmap_limits'][0] = 0

        for i in range(self.num_maps) :
            opts = self.map_options
            opts['title'] = self.titles[i]
            if self.filenames is not None :
                opts['img_filename'] = self.filenames[i]
            if schedule_diagram :
                opts['schedule'] = self.hist_specs[self.map_specs['hist'][i]]['exposure_schedule']
            render_map(
                self.maps[i,:,:],
                lat=self.lat,
                lon=self.lon,
                cbar_label=self.hist_specs[self.map_specs['hist'][i]]['units'],
                show=show,
                img_dir=img_dir,
                **opts)



def render_map(map,
lat=None,
lon=None,
title=None,
save=True,
show=True,
schedule=None,
schedule_bbox=(-0.03,0,1,0.91),
img_filename=None,
img_dir="",
img_size=[20,15],
img_dpi=300,
img_filetype="png",
brdr_nation=True,
brdr_nation_rgba=[0,0,0,1],
brdr_state=True,
brdr_state_rgba=[0,0,0,0.75],
cmap="gist_ncar",
cmap_limits=None,
cbar=True,
cbar_limits=None,
cbar_label=None,
country_focus="CHE",
gridlines=True,
gridlines_dms=True,
mch_logo=False) :
    """Renders and saves maps

    Renders and saves maps with a wide variety of aesthetic options.

    Parameters
    ----------
    map : array
        The map to be rendered
    lat : [type], optional
        [description], by default None
    lon : [type], optional
        [description], by default None
    title : [type], optional
        [description], by default None
    save : bool, optional
        [description], by default True
    show : bool, optional
        [description], by default True
    schedule : [type], optional
        [description], by default None
    schedule_bbox : tuple, optional
        [description], by default (-0.03,0,1,0.91)
    img_filename : [type], optional
        [description], by default None
    img_dir : str, optional
        [description], by default ""
    img_size : list, optional
        [description], by default [20,15]
    img_dpi : int, optional
        [description], by default 300
    img_filetype : str, optional
        [description], by default "png"
    brdr_nation : bool, optional
        [description], by default True
    brdr_nation_rgba : list, optional
        [description], by default [0,0,0,1]
    brdr_state : bool, optional
        [description], by default True
    brdr_state_rgba : list, optional
        [description], by default [0,0,0,0.75]
    cmap : str, optional
        [description], by default "gist_ncar"
    cmap_limits : [type], optional
        [description], by default None
    cbar : bool, optional
        [description], by default True
    cbar_limits : [type], optional
        [description], by default None
    cbar_label : [type], optional
        [description], by default None
    country_focus : str, optional
        [description], by default "CHE"
    gridlines : bool, optional
        [description], by default True
    gridlines_dms : bool, optional
        [description], by default False
    mch_logo : bool, optional
        [description], by default True
    """

    # TODO: Add custom sizing and resolution specifications
    fig = plt.figure(figsize=(img_size[0]/2.54,img_size[1]/2.54))

    # TODO: Accept custom projections
    # proj = ccrs.Mercator()
    proj = ccrs.Orthographic(central_longitude=(lon[0]+lon[-1])/2, central_latitude=(lat[0]+lat[-1])/2)

    # TODO: Add support for multiple plots per figure (too complex? consider use cases)
    ax = fig.add_subplot(1,1,1,projection = proj)

    # TODO: Increase flexibility of borders consideration
    if brdr_state :
        state_brdrs = cfeat.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        ax.add_feature(state_brdrs,linestyle="--",edgecolor=tuple(brdr_state_rgba),linewidth=0.5)
    if brdr_nation :
        ax.add_feature(cfeat.BORDERS,edgecolor=tuple(brdr_nation_rgba))

    if country_focus is not None :
        shpfilename = shapereader.natural_earth(resolution='10m',
            category='cultural',name='admin_0_countries')
        reader = shapereader.Reader(shpfilename)
        countries = reader.records()    
        # this is a very janky search for Switzerland, but it's ultimately simpler than
        # making geopandas a requirement for the library
        for country in countries :
            if country.attributes['ADM0_A3'] == country_focus :
                break
        assert country.attributes['ADM0_A3'] == country_focus, "country_focus input not recognised"
        poly = country.geometry

        msk_proj  = proj.project_geometry (poly, ccrs.Geodetic())  # project geometry to the projection used by stamen

        # plot the mask using semi-transparency (alpha=0.65) on the masked-out portion
        ax.add_geometries( msk_proj, proj, facecolor='white', edgecolor='none', alpha=0.8)

    # TODO: Consider first-last versus min-max - how can we avoid accidentally flipping images
    extents=[lon[0],lon[-1],lat[0],lat[-1]]
    ax.set_extent(extents,crs=ccrs.Geodetic())

    # this code correctly translate the lat/lon limits into the projected coordinates
    extents_proj = proj.transform_points(ccrs.Geodetic(),np.array(extents[:2]),np.array(extents[2:]))
    extents_proj = extents_proj[:,:2].flatten(order='F')

    if gridlines :
        ax.gridlines(draw_labels=True, dms=gridlines_dms, x_inline=False, y_inline=False,linewidth=0.25,
        ylocs=[46,46.5,47,47.5])

    # TODO: Custom colormaps, interpolation, cropping

    # Upscale matrix for better reprojection
    # f = interp2d(lon, lat, map, kind='linear')
    # latnew = np.linspace(lat[0], lat[-1], (len(lat)-1)*3+1)
    # lonnew = np.linspace(lon[0], lon[-1], (len(lon)-1)*3+1)
    # mapnew = f(lonnew, latnew)

    # Upscale matrix for better reprojection
    mapnew = zoom(map,3)

    # show map with given cmap and set cmap limits
    im = ax.imshow(mapnew,extent=extents,transform=ccrs.PlateCarree(),
        origin='lower',cmap=cmap)
    if cmap_limits is not None :
        im.set_clim(cmap_limits[0],cmap_limits[1])

    # colorbar
    # TODO: Add support for horizontal vertical option
    if cbar :
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',pad=0.05,fraction=0.05)
        cb.ax.set_xlabel(cbar_label)

    # show schedule diagram
    if schedule is not None :
        ax2 = inset_axes(ax, width="25%", height="25%", loc=2,
            axes_class = get_projection_class('polar'),
            bbox_to_anchor=tuple(schedule_bbox),
            bbox_transform=ax.transAxes)
        schedule_clock(ax2,schedule,title="Exposure schedule")

    # TODO: Add more advanced title interpretation (i.e. smart date placeholder)
    if title is not None :
        ax.set_title(title)

    if mch_logo :
        ex = ax.get_extent()
        mch_logo_img = plt.imread('python_tamer/mch_logo.png')
        mch_logo_width = 0.15
        mch_logo_pad = 0
        # some maths to work out position, note image aspect ratio 5:1
        mch_extents = [ex[1]-(ex[1]-ex[0])*mch_logo_width-(ex[1]-ex[0])*mch_logo_pad,
            ex[1]-(ex[1]-ex[0])*mch_logo_pad,
            ex[2]+(ex[3]-ex[2])*mch_logo_pad,
            ex[2]+0.2*(ex[1]-ex[0])*mch_logo_width+(ex[3]-ex[2])*mch_logo_pad]
        # zorder puts image on top (behind mask otherwise for some reason)
        ax.imshow(mch_logo_img,extent=mch_extents,zorder=12)

    # TODO: Add plot title, small textbox description, copyright from dataset, ticks and gridlines
    if save :
        # Generate timestamp filename if relying on default
        if img_filename is None :
            if title is not None :
                img_filename = format_filename(title)
            else :
                img_filename=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        elif img_filename == "timestamp" :
            img_filename=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        plt.savefig(img_dir+img_filename+"."+img_filetype,
            bbox_inches="tight",dpi=img_dpi)

    if show :
        plt.show()


def schedule_clock(axes,schedule,title=None,title_size=9,center=0.25,rmax=1) :
    """Generates a clock-style representation of an exposure schedule

    [extended_summary]

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The polar axes upon which the clock will be plotted
    schedule : list or numpy.ndarray
        The exposure schedule - a length-24 vector of hourly exposure proportions
    title : str, optional
        Title of the exposure clock
    title_size : int, optional
        [description], by default 9
    center : float, optional
        [description], by default 0.25
    rmax : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    axes.bar(
        np.arange(24)/24*2*np.pi, 
        schedule,
        width=2*np.pi/24,
        align='edge',
        bottom=center) 
    axes.bar(0,0.25,width=2*np.pi,color='k')      
    # Set the circumference labels
    axes.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    axes.set_xticklabels(np.linspace(0, 24, 8, endpoint=False,dtype=int),fontsize=8) 
    axes.tick_params(axis='both',which='major', pad=-3)
    axes.set_yticks([0.5+center])
    axes.set_yticklabels(['0.5'],fontsize=5)
    axes.grid(True,color='black',linewidth=0.25)
    # Make the labels go clockwise
    axes.set_theta_direction(-1)       
    # Place 0 at the top
    axes.set_theta_offset(np.pi/2)   
    # format grid and fake ticks
    for t  in np.linspace(0, 2*np.pi, 24, endpoint=False):
        axes.plot([t,t], np.array([0.95,1.1])*rmax+center, lw=0.5, color="k")
    for t  in np.linspace(0, 2*np.pi, 8, endpoint=False):
        axes.plot([t,t], np.array([0.9,1.1])*rmax+center, color="k")   
    if title is not None :
        axes.set_title(title,fontsize=title_size) 
    axes.set_rmax(rmax+center)

    return axes


def gen_map_title(
statistic = None,
exposure_schedule=None,
hour=None,
units=None,
year_selection=None,
day_selection=None,
filename=False,
**kwargs) :
       # AI added trace option Feb 21 20204

    if units in ['SED','J m-2','UVIh'] :
        if all(exposure_schedule == np.ones(24)) :
            title = 'UV daily doses'
        else :
            title = 'UV doses'
    elif units in ['W m-2','UVI','mW m-2'] :
        if np.sum(exposure_schedule)==1 and all(x in [0,1] for x in exposure_schedule) :
            #user chosen just one hour
            hour=np.where(exposure_schedule)[0][0]
            title = 'UV intensity between ' + str(hour) + 'h-' + str(hour+1) + 'h'
        else :
            title = 'UV intensity'
    else :
        raise ValueError('Units must be SED, J m-2, UVIh, UVI, W m-2, or mW m-2')
    

    ayear = pd.date_range(start="2010-01-01",end="2010-12-31")
    ds = {'year' : ayear.dayofyear.values.tolist()}
    ds['winter (DJF)'] = [x for i in [12,1,2] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
    ds['spring (MAM)'] = [x for i in [3,4,5] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
    ds['summer (JJA)'] = [x for i in [6,7,8] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
    ds['autumn (SON)'] = [x for i in [9,10,11] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
    ds['January'] = ayear[ayear.month == 1].dayofyear.values.tolist()
    ds["February"] = ayear[ayear.month == 2].dayofyear.values.tolist()
    ds["March"] = ayear[ayear.month == 3].dayofyear.values.tolist()
    ds["April"] = ayear[ayear.month == 4].dayofyear.values.tolist()
    ds["May"] = ayear[ayear.month == 5].dayofyear.values.tolist()
    ds["June"] = ayear[ayear.month == 6].dayofyear.values.tolist()
    ds["July"] = ayear[ayear.month == 7].dayofyear.values.tolist()
    ds["August"] = ayear[ayear.month == 8].dayofyear.values.tolist()
    ds["September"] = ayear[ayear.month == 9].dayofyear.values.tolist()
    ds["October"] = ayear[ayear.month == 10].dayofyear.values.tolist()
    ds["November"] = ayear[ayear.month == 11].dayofyear.values.tolist()
    ds["December"] = ayear[ayear.month == 12].dayofyear.values.tolist()

    day_str = None

    for item in ds.items() :
        if set(day_selection) == set(item[1]) :
            day_str = item[0]
            break

    
    if day_str == 'year' :
        if len(year_selection) == 1 :
            title = title + ' for the year of ' + str(year_selection[0])
        elif all(np.diff(year_selection)==1) :
            title = title + ' for the years ' + str(np.min(year_selection)) + '-' + str(np.max(year_selection))
        else :
            title = title + ' for the years: ' + np.array2string(year_selection,separator=', ')

    elif day_str is not None :
        title = title + ' for ' + day_str 
        if len(year_selection) == 1 :
            title = title + ' ' + str(year_selection[0])
        elif all(np.diff(year_selection)==1) :
            title = title + ', ' + str(np.min(year_selection)) + '-' + str(np.max(year_selection))
        else :
            title = title + ' ' + np.array2string(year_selection,separator=', ')

    else :
        # TODO: potentially make this workable with "custom day selection" placeholder in title
        raise ValueError("Day selection not recognised, auto-title cannot proceed")

    title_trace = 'Spatially averaged ' + title
    title = statistic + ' of ' + title

    if filename :
        custom = False
        filename = "UV." + units + '.' + statistic + '.'
        if len(year_selection) == 1 :
            filename = filename + str(year_selection[0]) + '.'
        elif all(np.diff(year_selection)==1) :
            filename = filename + str(np.min(year_selection)) + '-' + str(np.max(year_selection)) + '.'
        else :
            filename = filename + str(year_selection[0]) + '-custom' + '.'
            custom = True
        day_str_filenamer = {
            "January"   : "01",
            "February"  : "02",
            "March"     : "03",
            "April"     : "04",
            "May"       : "05",
            "June"      : "06",
            "July"      : "07",
            "August"    : "08",
            "September" : "09",
            "October"   : "10",
            "November"  : "11",
            "December"  : "12",
            "winter (DJF)"  : "s-",
            "spring (MAM)"  : "s-",
            "summer (JJA)"  : "s-",
            "autumn (SON)"  : "s-",
            "year"      : "year"}
        filename = filename + day_str_filenamer[day_str] 
        if hour is not None :
            filename = filename + '.' + str(hour) + 'h'
        if custom :
            filename = filename + '.created_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = format_filename(filename)
            
        filename_trace='trace_' + filename
        return title,title_trace,filename,filename_trace
    else :
        return title, title_trace
    





def calculate_map_from_hists(pix_hist,statistic,bin_centers) :

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
    if statistic.lower() in basic_descriptor_functions.keys() :
        # in this case, we can simply select the basic function from the dict...
        descriptor_function = basic_descriptor_functions[statistic.lower()]
        # ...and execute it across the map
        map = np.apply_along_axis(lambda x: descriptor_function(x,bin_centers),0,pix_hist)
    # TODO: a loose space could ruin this, need shunting yard algorithm of sorts
    elif statistic.lower()[3:] == "prct" or statistic.lower()[3:] == "percentile" :
        prct = int(statistic[0:2]) / 100
        map = np.apply_along_axis(lambda x: hist_percentile(x,bin_centers,prct),0,pix_hist)
    else :
        # TODO: interpret self.statistic to build advanced functions (y i k e s)
        raise ValueError("Statistic string not recognised")
    

    return map