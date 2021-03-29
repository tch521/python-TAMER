import pandas as pd
import numpy as np
import datetime as dt
import os
import re
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from .subroutines import *


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
        Defaults to None whereby the program selects all data within the data_directory that
        matches the src_filename_format.

    src_filename_format : str
        Describes the filename of the netCDF files containing the data with 'yyyy' in place 
        of the year.
    
    data_directory : str
        The directory where the data is stored. Must end with a slash.


    Example
    -------

    The code below shows a typical use case for the ExposureMap class. The long-term average daily doses
    (i.e. the chronic doses) for typical school children are calculated across Switzerland asssuming certain
    hours of exposure for journeying to and from school and having breaks for morning tea and lunch time. ::

        import python_tamer as pt
        import pandas as pd
        data_directory = 'C:/enter_your_data_directory_here'
        ER = pt.ER_Vernez_2015("Forehead","Standing") # Long-term average ER for foreheads in standing posture
        map = pt.ExposureMap(
            data_directory=data_directory,
            units = "J m-2",
            exposure_schedule = [0  ,0  ,0  ,0  ,0  ,0  ,
                                0  ,0  ,0.5,0  ,0.5,0  ,
                                0.5,0.5,0  ,0  ,0.5,0  ,
                                0  ,0  ,0  ,0  ,0  ,0  ]*ER,
            bin_width = 25,
            date_selection = pd.date_range(start="2005-01-01",end="2014-12-31"),
            statistic = "mean",
            map_options={"title": "Chronic daily UV dose for typical school children, 2005-2014",
                        "save": False})
        map = map.collect_data().calculate_map()
        map.plot_map()


    """

    def __init__(self,units="SED",
    exposure_schedule=1,
    statistic="mean",
    bin_width = None,
    date_selection=None,
    map_options=None,
    src_filename_format='UVery.AS_ch02.lonlat_yyyy01010000.nc',
    data_directory='C:/Data/UV/'):
        # assigning options to fields in class with a few basic checks
        self.units = units

        self.exposure_schedule=np.array(exposure_schedule)
        if len(np.atleast_1d(self.exposure_schedule)) == 1 :
            self.exposure_schedule = np.repeat(self.exposure_schedule,24)

        self.statistic = statistic

        self.map_options = {
            "title" : "Test map",
            "save" : True,
            "img_size" : [20,15],
            "img_dpi" : 300,
            "img_dir" : "",
            "img_filename" : "timestamp",
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
        if map_options is not None :
            self.map_options.update(map_options)

        self.src_filename_format = src_filename_format
        self.data_directory = data_directory

        self.date_selection = date_selection

        if bin_width is None :
            self.bin_width = {
                "SED" : 0.1, 
                "J m-2" : 10, 
                "UVI" : 0.1, 
                "W m-2" : 0.0025, 
                "mW m-2" : 2.5
            }[self.units]
        else :
            self.bin_width = bin_width
        
    
    def collect_data(self, data_directory=None,src_filename_format=None,
    date_selection=None,units=None,exposure_schedule=None,bin_width=None) :
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
            of the year.
        
        data_directory : str
            The directory where the data is stored. Must end with a slash.

        date_selection : list of dates
            The dates for which the irradiances are retrieved or the daily doses are calculated. 
            Defaults to None whereby the program selects all data within the data_directory that
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
            import pandas 
            data_directory = 'C:/enter_your_data_directory_here'
            ER = pt.ER_Vernez_2015("Forehead","Standing") # Long-term average ER for foreheads in standing posture
            map = pt.ExposureMap()
            map = map.collect_data(
                data_directory=data_directory,
                units = "J m-2",
                exposure_schedule = [0  ,0  ,0  ,0  ,0  ,0  ,
                                     0  ,0  ,0.5,0  ,0.5,0  ,
                                     0.5,0.5,0  ,0  ,0.5,0  ,
                                     0  ,0  ,0  ,0  ,0  ,0  ]*ER,
                bin_width = 25,
                date_selection = pandas.date_range(start="2005-01-01",end="2014-12-31")
            )
            map = map.calculate_map(statistic = "mean")
            map.plot_map(map_options={"title": "Chronic daily UV dose for typical school children, 2005-2014",
                                      "save": False})


        """

        # TODO: There must be a better way to do this
        if not (data_directory is None) :
            self.data_directory = data_directory
        if not (src_filename_format is None) :
            self.src_filename_format = src_filename_format
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

        for i in range(len(list_of_years)) :
            year = list_of_years[i]
            print("Processing year "+str(year)) #should use logging, don't yet know how
            dataset=nc.Dataset(self.data_directory+self.src_filename_format.replace('yyyy',str(year))) 
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
            data = dataset['UV_AS'][time_subset,:,:] #work in UVI by default because it's easy to read
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
                self.num_bins = int(2*np.nanmax(data) // self.bin_width)
                self.bin_edges = np.array(range(self.num_bins+1)) * self.bin_width
                self.bin_centers = self.bin_edges[:-1] + 0.5 * np.diff(self.bin_edges)

                # TODO: think about possible cases where dimensions could differ
                self.pix_hist=np.empty([self.num_bins,
                    np.shape(data)[-2],np.shape(data)[-1]], dtype=np.int16)

                # TODO: this should also be done by some initial dataset analysis, but that's a drastic
                # design overhaul
                self.lat = dataset['lat'][:]
                self.lon = dataset['lon'][:]

            # TODO: add check here in case max exceeds current bin edges
            # now put data into hist using apply_along_axis to perform histogram for each pixel
            def hist_raw(x) :
                hist, _ = np.histogram(x,bins=self.bin_edges)
                return hist

            print("   Calculating and adding to pix hist")
            self.pix_hist[:,:,:] += np.apply_along_axis(hist_raw,0,data)

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
        produce multiple different maps. ::

            import python_tamer as pt
            import numpy as np
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

            "title" : "Test map", 
            The title to be rendered above the map. Can be left blank for no title. Can be 
            used to inform img_filename

            "save" : True, 
            Boolean to declare whether the map should be saved as an image file or not.
            
            "img_size" : [20,15], 
            The size [width,height] of the image in cm.

            "img_dpi" : 300, 
            The dots per inch of the saved image.

            "img_dir" : "", 
            The directory for the image to be saved in, leaving it blank should result
            in images being saved in the working directory.

            "img_filename" : "timestamp", 
            The image filename as a string. The default value of "timestamp" is a keyword
            indicating that the function should generate a filename based on the time at
            the moment of the calculation, specified with the format %Y%m%d_%H%M%S_%f 
            which includes millisecond precision.

            "img_filetype" : "png", 
            The image filetype, must be acceptable to `matplotlib.pyplot.savefig()`.

            "brdr_nation" : True, 
            Boolean for drawing national borders on the map.

            "brdr_nation_rgba" : [0,0,0,0], 
            The red, green, blue, and alpha values for the national borders.

            "brdr_state" : False, 
            Boolean for drawing state borders as defined by Natural Earth dataset.

            "brdr_state_rgba" : [0,0,0,0.67], 
            The red, green, blue, and alpha values for the national borders.

            "cmap" : "jet", 
            The name of the colourmap to be used when rendering the map.

            "cmap_limits" : None, 
            The numeric limits of the colourmap. Defaults to None, where the lower
            and upper limits of the plotted data are used as the colourmap limits.

            "cbar" : True, 
            Boolean for rendering a colourbar.

            "cbar_limits" : None, 
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


class ExposureMapBatch:
    
    def __init__(self,
    quick_selection=None,
    units="SED",
    exposure_schedule=1,
    statistic="mean",
    bin_width = None,
    date_selection=None,
    climatology_years=0,
    map_options=None,
    src_filename_format='UVery.AS_ch02.lonlat_yyyy01010000.nc',
    data_directory='C:/Data/UV/'):
        # start with data location to quickly get some metadata
        self.src_filename_format = src_filename_format
        self.data_directory = data_directory
        # first we read the data_directory to check the total number of unique years available
        data_dir_contents = os.listdir(self.data_directory)
        # TODO: improve jankiness of this format-matching search for filenames
        char_year = self.src_filename_format.find('yyyy')
        dataset_years = [ x for x in data_dir_contents if re.findall(self.src_filename_format.replace("yyyy","[0-9]{4}"),x)]
        dataset_years = [ int(x[char_year:char_year+4]) for x in dataset_years ]


        # if quick_selection is not None :
        #     if quick_selection.lower() == "monthly" :
        #         date_selection = 

        # assigning options to fields in class with a few basic checks
        self.units = units

        self.exposure_schedule=np.array(exposure_schedule)
        if len(np.atleast_1d(self.exposure_schedule)) == 1 :
            self.exposure_schedule = np.repeat(self.exposure_schedule,24)

        self.statistic = statistic

        self.map_options = {
            "title" : "Test map",
            "save" : True,
            "img_size" : [20,15],
            "img_dpi" : 300,
            "img_dir" : "",
            "img_filename" : "timestamp",
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
        if map_options is not None :
            self.map_options = self.map_options.update(map_options)

        self.date_selection = date_selection

        if bin_width is None :
            self.bin_width = {
                "SED" : 0.1, 
                "J m-2" : 10, 
                "UVI" : 0.1, 
                "W m-2" : 0.0025, 
                "mW m-2" : 2.5
            }[self.units]
        else :
            self.bin_width = bin_width
    