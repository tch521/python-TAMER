import numpy as np
import pandas as pd
import datetime as dt
import string
import shapefile as shp
from pyproj import Transformer

def assert_data_shape_24(data,reverse=False,force_second_dim=True) :
    """Simple function to check if first dimension is 24 hours and, if not, reshapes accordingly
    """
    datashape = np.shape(data)
    # TODO: Could be a big job, but this F ordering is weird and I should reconsider
    if datashape[0] != 24 and not reverse: # Checks that first dimension is length 24 (hours in a day) and reshapes if not
        new_shape = (24, datashape[0]//24) + datashape[1:]
    elif reverse :
        new_shape = [datashape[0] * datashape[1]] + list(datashape[2:])
    elif force_second_dim :
        new_shape = (24, 1) + datashape[1:]
    else :
        # option in case no reshaping necessary
        return data
    data = np.reshape(data,new_shape,order='F')
    return data


def ER_Vernez_model_equation(Vis,mSZA) :
    """ER_Vernez_model_equation calculates the Exposure Ratio according to the Vis parameter and the Solar Zenith Angle.

    See Vernez et al., Journal of Exposure Science and Environmental Epidemiology (2015) 25, 113–118 
    (doi:10.1038/jes.2014.6) for further details on the model used for the calculation.

    Args:
        Vis (pandas.DataFrame): Values for the Vis parameter (percentages between 0 and 100)
        mSZA (pandas.DataFrame): Values of the minimal Solar Zenith Angle in degrees for the 
            given date and latitude. Can be calculated using the min_solar_zenith_angle function

    Returns:
        pandas.DataFrame: A single column DataFrame containing the calculated Exposure Ratios.
    """

    Vis_cent = Vis / 10 - 5.800
    lnVis_cent = np.log(Vis / 10) - 1.758
    cosSZA3_cent = np.cos(np.radians(mSZA))**3 - 0.315
    ER = -3.396 * lnVis_cent + 10.714 * Vis_cent - 9.199 * cosSZA3_cent + 56.991
    return ER
    

def min_solar_zenith_angle(date,lat) :
    """min_solar_zenith_angle calculates the minimal Solar Zenith Angle for a given date and latitude.

    This function is adapted from the SACRaM_astr MATLAB function written by Laurent Vuilleumier for MeteoSwiss.

    Args:
        date (pandas.DataFrame): A datetime column describing the specific day of exposure
        lat (panda.DataFrame): A column of latitude values in decimal degrees

    Returns:
        pandas.DataFrame: A column of minimal SZA values in degrees.
    """

    if type(date) is pd.core.series.Series :
        TrTim = date.apply(lambda x: x.toordinal() + 366).to_numpy() * 2.73785151e-05 - 18.9996356
    else : # adds support for single date input
        TrTim = (date.toordinal() +366) * 2.73785151e-05 - 18.9996356
        TrTim = np.array(TrTim)
    G  = np.radians(np.mod( 358.475833 + 35999.04975   * TrTim - 0.000150 * TrTim**2 , 360))
    SL = np.radians(np.mod( 279.696678 + 36000.76892   * TrTim + 0.000303 * TrTim**2 , 360))
    SJ = np.radians(np.mod( 225.444651 + 3034.906654   * TrTim , 360))
    SN = np.radians(np.mod( 259.183275 - 1934.142008   * TrTim + 0.002078 * TrTim**2 , 360))
    SV = np.radians(np.mod( 212.603219 + 58517.803875  * TrTim + 0.001286 * TrTim**2 , 360))
    theta = (-( 0.00001 * TrTim * np.sin( G + SL ) + 0.00001 * np.cos( G - SL - SJ ) ) -
        0.000014 * np.sin( 2*G - SL ) - 0.00003 * TrTim * np.sin( G - SL ) -
        0.000039 * np.sin( SN - SL )  - 0.00004 * np.cos( SL ) +
        0.000042 * np.sin( 2*G + SL ) - 0.000208 * TrTim * np.sin( SL ) +
        0.003334 * np.sin( G+SL ) +
        0.009999 * np.sin( G-SL ) +
        0.39793  * np.sin( SL ))
    rho = (0.000027 * np.sin( 2*G - 2*SV ) - 0.000033 * np.sin( G - SJ )
        + 0.000084 * TrTim * np.cos( G ) - 0.00014 * np.cos( 2*G ) - 0.033503 * np.cos( G )
        + 1.000421)
    declination = np.arcsin(theta / np.sqrt(rho))
    return lat - np.degrees(declination)    


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def convert_swiss_time_to_UTC(input_table,name) :
    # TODO: Need a replacement for this, but argument the responsibility of the user?
    def convert_swiss_time_to_UTC_iter(time_in,Date) :
        if Date.month > 3 and Date.month < 11 :
            time_out = time_in 
            time_out = time_out.replace(hour=time_in.hour - 2)
        else :
            time_out = time_in 
            time_out = time_out.replace(hour=time_in.hour - 1)
        return time_out
    new_time = input_table.apply(lambda x: convert_swiss_time_to_UTC_iter(x[name],x["Date"]),axis='columns')
    return new_time


def hist_mean(counts,bin_centers) :
    """hist_mean calculates the mean of a histogram using numpy functions

    This function is designed to calculate a histogram mean as efficiently as possible

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.

    Returns:
        float: the mean value of the histogram
    """

    mean = np.dot(counts, bin_centers) / np.sum(counts)
    return mean


def hist_var(counts,bin_centers) :
    """hist_var calculates the variance of a histogram

    This function calculates the variance of a histogram as E[X^2] - E[X]^2 using the
    hist_mean function to efficiently calculate expectation values.

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.

    Returns:
        float: the variance of the histogram
    """

    # Not really essential seeing as this would break the units
    var = hist_mean(counts,bin_centers**2) - hist_mean(counts,bin_centers)**2
    return var


def hist_stdev(counts,bin_centers) :
    """hist_stdev calculates the standard deviation of a histogram

    This function calculates the variance of a histogram as E[X^2] - E[X]^2 using the
    hist_mean function to efficiently calculate expectation values. It then returns the
    square root of the variance.

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.

    Returns:
        float: the standard deviation of the histogram
    """

    var = hist_mean(counts,bin_centers**2) - hist_mean(counts,bin_centers)**2
    return var**0.5


def hist_percentile(counts,bin_centers,prct) :
    """hist_percentile calculates percentiles of histogram data

    This function takes discretised data, typical for histograms, and calculates
    the user-specified percentile quantity. 

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.
        prct (float): A fraction betwee 0.0 and 1.0 for the desired percentile.

    Returns:
        float: The desired percentile value. In cases where the quantity falls 
            between two bins, their respective central values are averaged.
    """

    n = np.sum(counts)
    cumcounts = np.cumsum(counts)
    # TODO: Possibly unnecessary, but could probably improve efficiency of
    # this if statement (e.g. if i==j no need to take average)
    if prct == 0 :
        # special case: searching for min
        j = np.searchsorted(cumcounts,n*prct,side='right')
        percentile = bin_centers[j]
    elif prct == 1 : 
        # special case: searching for max
        i = np.searchsorted(cumcounts,n*prct)
        percentile = bin_centers[i]
    else :
        i = np.searchsorted(cumcounts,n*prct) 
        j = np.searchsorted(cumcounts,n*prct,side='right') 
        percentile = (bin_centers[i] + bin_centers[j])/2

    return percentile   


def hist_min(counts,bin_centers) :
    """hist_min calculates the minimum value of a histogram

    This function finds the minimum value of a histogram.
    It is built on the some basic functionality as hist_percentile.

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.

    Returns:
        float: The minimum value of the histogram
    """

    cumcounts = np.cumsum(counts)
    j = np.searchsorted(cumcounts,0,side='right')
    min = bin_centers[j]

    return min


def hist_max(counts,bin_centers) :
    """hist_max calculates the maximum value of a histogram

    This function finds the maximum value of a histogram.
    It is built on the some basic functionality as hist_percentile.

    Args:
        counts (array): The quantity of numbers within each histogram bin
        bin_centers (array): The central value of each histogram bin. Note that
            this can be calculated as bin_edges[:-1]+0.5*np.diff(bin_edges) but
            we removed this calculation to optimise this function further.

    Returns:
        float: The maximum value of the histogram
    """

    n = np.sum(counts)
    cumcounts = np.cumsum(counts)
    i = np.searchsorted(cumcounts,n)
    max = bin_centers[i]

    return max


def ER_Vernez_2015(Anatomic_zone,
Posture,
Date=None,
Latitude=None,
Vis_table_path=None,
Vis_table=None) :
    """Calculates Exposure Ratios for a given anatomic zone, posture, and date.

    This function calculates ER as a percentage between 0 and 100 based on Anatomic_zone, Posture, Date, and Latitude
    information. This function contains hard-coded synonyms for certain anatomical zones, e.g. 'Forehead" 
    maps to "Face'. See Vernez et al., Journal of Exposure Science and Environmental Epidemiology (2015) 
    25, 113–118 (https://doi.org/10.1038/jes.2014.6) for further details on the model used for the calculation.


    Parameters
    ----------

    Anatomic_zone : list
        String or list of strings describing the anatomic zone for which the ER is to be calculated. 

    Posture : list
        String or list of strings describing the posture for which the ER is to be calculated.

    Date : list, optional
        The date for which the ER is to be calculated. The date affects the minimum solar zenith
        angle in the Vernez et al. 2015 ER model. The specific year is not relevant. Defaults to
        March 20, the equinox.

    Latitude : list, optional
        The latitude is important for calculating the ER. Defaults to None, wherein the latitude
        of the centroid of Switzerland (46.8 degrees) is used.

    Vis_table_path : str, optional
        The full path to an alternative table for the Vis parameter. 
        Must be a csv file. Defaults to None.

    Vis_table : str, optional
        An alternative table for the Vis parameter. Defaults to None.


    Returns
    -------

    list
        Returns ER values as a list


    """

    # In case of single input rather than lists
    if isinstance(Anatomic_zone, str): Anatomic_zone = [Anatomic_zone]
    if isinstance(Posture,str): Posture = [Posture]

    if Latitude is None:
        Latitude = [46.8] #Switzerland centroid
    
    if Date is None:
        Date = [dt.date(2015,3,20)] # equinox

    if not isinstance(Latitude,list): Latitude = [Latitude]
    if not isinstance(Date,list): Date = [Date]

    d = {'Anatomic_zone': Anatomic_zone,
         'Posture': Posture,
         'Latitude': Latitude,
         'Date': Date}

    lengths = [len(x) for x in d.values()]
    max_length = max(lengths)
    for key in list(d.keys()) :
        if len(d[key]) != max_length :
            d[key] = d[key] * (max_length//len(d[key]))

    self = pd.DataFrame(d)

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
        Vis_table['Standing']=Vis_table['Standing erect arms down'] 
    elif Vis_table is None :
        Vis_table = pd.read_csv(Vis_table_path)

    # Below is a dictionary describing a range of synonyms for the anatomical zones defined in the Vis table.
    Anatomic_zone_synonyms_reverse = {
        'Forearm'    : ['wrist',
                        'Left extern radial',
                        'Right extern radial',
                        'Left wrist: radius head',
                        'Right wrist: radius head',
                        'Left wrist',
                        'Right wrist'],
        'Face'       : ['Forehead'],
        'Upper back' : ['Right trapezoid',
                        'Left trapezoid',
                        'trapezius'],
        'Belly'      : ['Chest'],
        'Shoulder'   : ['Left deltoid',
                        'Right deltoid',
                        'Left shoulder',
                        'Right shoulder'],
        'Upper arm'  : ['Left elbow',
                        'Right elbow',
                        'Left biceps',
                        'Right biceps'],
        'Upper leg'  : ['Left thigh',
                        'Right thigh',
                        'Left knee',
                        'Right knee'],
        'Lower back' : ['Low back']
    }
    
    # The dictionary is reversed so that the multiple synonyms can be mapped to the few correct terms for the Vis table.
    Anatomic_zone_synonyms = {keys: old_keys for old_keys, old_values in Anatomic_zone_synonyms_reverse.items() for keys in old_values}

    self = self.replace({'Anatomic_zone' : Anatomic_zone_synonyms})

    # With the correct anatomic zone names established, we can lookup the Vis values from the table
    Vis = Vis_table.lookup(self['Anatomic_zone'],self['Posture'])

    # Next we must calculate the minimal Solar Zenith Angle for the given date
    mSZA = min_solar_zenith_angle(self.Date,self.Latitude)

    # With the Vis value and the SZA, we can calculate the ER according to the Vernez model
    ER = ER_Vernez_model_equation(Vis,mSZA) / 100

    return ER.to_numpy()

def format_filename(inp):
    """Takes a string and return a valid filename constructed from the string.
    
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.
    
    Note: this method may produce invalid filenames such as ``, `.` or `..`
    When using this method, prepend a date string like '2009_01_15_19_46_32_'
    and append a file extension like '.txt', so to avoid the potential of using
    an invalid filename.


    Parameters
    ----------
    s : str
        Input string to be converted to valid filename


    Returns
    -------
    str
        
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    inp_rpl = inp.replace(' ','_').replace(':','-')
    filename = ''.join(c for c in inp_rpl if c in valid_chars)
    return filename

def str2daysofyear(inp) :
    """Interprets a string, list, or array into a list of arrays for days of the year

    An ExposureMapSequence object requires of a list of arrays describing the days of
    the year to be used in the creation of each histogram. This function simplifies the
    process of entering this information. The user can enter keywords to automatically
    generate the appropriate list of days.


    Parameters
    ----------
    inp : str or list or numpy.array
        The input to be interpreted. Numeric entries should be included in the output
        unmodified, while string entries should be replaced by numeric arrays.

    Returns
    -------
    list
        Produces a list of arrays that is interpretable by the ExposureMapSequence code.
    """

    def str2daysofyear_raw(inp) :

        ayear = pd.date_range(start="2010-01-01",end="2010-12-31")
        winter = [x for i in [12,1,2] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
        spring = [x for i in [3,4,5] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
        summer = [x for i in [6,7,8] for x in ayear[ayear.month == i].dayofyear.values.tolist()]
        autumn = [x for i in [9,10,11] for x in ayear[ayear.month == i].dayofyear.values.tolist()]

        keys_ref = {
            "month"  : [ayear[ayear.month == i].dayofyear.values.tolist() for i in range(1,13)],
            "season" : [spring,summer,autumn,winter],
            "quarter": [spring,summer,autumn,winter],
            "year"   : ayear.dayofyear.values.tolist(),
            "annual" : ayear.dayofyear.values.tolist(),
            "jan" : ayear[ayear.month == 1].dayofyear.values.tolist(),
            "feb" : ayear[ayear.month == 2].dayofyear.values.tolist(),
            "mar" : ayear[ayear.month == 3].dayofyear.values.tolist(),
            "apr" : ayear[ayear.month == 4].dayofyear.values.tolist(),
            "may" : ayear[ayear.month == 5].dayofyear.values.tolist(),
            "jun" : ayear[ayear.month == 6].dayofyear.values.tolist(),
            "jul" : ayear[ayear.month == 7].dayofyear.values.tolist(),
            "aug" : ayear[ayear.month == 8].dayofyear.values.tolist(),
            "sep" : ayear[ayear.month == 9].dayofyear.values.tolist(),
            "oct" : ayear[ayear.month == 10].dayofyear.values.tolist(),
            "nov" : ayear[ayear.month == 11].dayofyear.values.tolist(),
            "dec" : ayear[ayear.month == 12].dayofyear.values.tolist(),
            "winter" : winter,
            "autumn" : autumn,
            "fall"   : autumn,
            "spring" : spring,
            "summer" : summer,
        }

        return keys_ref[inp]
    
    
    keys = ["month","season","quarter","year","annual",
            "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
            "winter","autumn","fall","spring","summer"]

    if isinstance(inp,str) :
        # simple case, user has entered "monthly" or some such

        # there should be only one result from the filter anyway
        inp_flt = list(filter(lambda x: x in inp.lower(), keys))

        out = str2daysofyear_raw(inp_flt[0])

        # in case user hasn't selected one of the nice advanced options, must convert to list
        if inp_flt[0] not in ["month","season","quarter","year","annual"] :
            out = [out]

        # TODO: rewrite this function to do this first and then pass filtered input to raw
        if inp_flt[0] == 'month' :
            inp_flt = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        elif inp_flt[0] in ['season','quarter'] :
            inp_flt = ['spring','summer','autumn','winter']
        elif inp_flt[0] == 'year' :
            inp_flt[0] = 'annual'

        nonstrings = [False]

    elif isinstance(inp,list) :
        # complex case, user has entered list 
        # ["june","july","august","summer"] or some such
        out = []
        inp_flt = []
        nonstrings = []
        for inpx in inp :
            if isinstance(inpx,str) : 
                inp_flt_temp = list(filter(lambda x: x in inpx.lower(), keys))[0]
                inp_flt.append(inp_flt_temp)
                out.append(str2daysofyear_raw(inp_flt_temp))
                nonstrings.append(False)
            else :
                inp_flt.append(inpx)
                out.append(inpx)
                nonstrings.append(True)
    
    # convert list of lists to list of arrays for consistency
    for i in range(len(out)) :
        out[i] = np.array(out[i])

    return out, inp_flt, nonstrings

def select_box_canton(canton_abbr, canton_folder):
    ''' Return the latitude-longitude box containing the selected canton.
        Return also longitude and latitude coordinates for plotting purposes.
        
        Canton names input are in each canton's initials:
    '''
    # set path to file containing cantons data and shapes
    canton_file = canton_folder + 'g2k16vz.shp'
    # open file
    sf   = shp.Reader(canton_file)
    # Dictionary for canton assignment
    canton_dict = {'ZH' : 'Zurich',
                   'BE' : 'Bern / Berne',
                   'LU' : 'Luzern',
                   'UR' : 'Uri',
                   'OW' : 'Obwalden',
                   'NW' : 'Nidwalden',
                   'SW' : 'Scwyz',
                   'GL' : 'Glarus',
                   'ZG' : 'Zug',
                   'FR' : 'Fribourg / Freiburg', 
                   'SO' : 'Solothurn', 
                   'BS' : 'Basel-Stadt',
                   'BL' : 'Basel-Landschaft',
                   'SH' : 'Schaffhausen',
                   'AR' : 'Appenzell Ausserrhoden',
                   'AI' : 'Appenzell Innerrhoden',
                   'SG' : 'St. Gallen',
                   'GR' : 'Graubünden / Grigioni / Grischun',
                   'AG' : 'Aargau',
                   'TG' : 'Thurgau',
                   'TI' : 'Ticino',
                   'VD' : 'Vaud',
                   'VS' : 'Valais / Wallis',
                   'NE' : 'Neuchâtel',
                   'GE' : 'Genève',
                   'JU' : 'Jura'} 
                    
    for initials in canton_dict:
        if initials == canton_abbr:
            canton_name =  canton_dict[initials] 

    # iterate over shapes and records - it should be the optimised iterator for shapefiles
    for shapeRec in sf.iterShapeRecords():
        if canton_name in shapeRec.record['KTNAME']:
            coords    = shapeRec.shape.points 
            # coord is a list of tuples
            # access first and second element of all tuples in the list
            s_longitude = list(zip(*coords))[0]
            s_latitude  = list(zip(*coords))[1]
    # convert swiss coordinates to latitude and longitude
    # using Transformer from pyproj 
    transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326")
    # transformer.transform(s_longitude, s_latitude)
    latitude  = list(transformer.transform(s_longitude, s_latitude))[0]
    longitude = list(transformer.transform(s_longitude, s_latitude))[1]
    # define the box enclosing the canton 
    box       = [min(longitude), min(latitude), max(longitude), max(latitude)] 
    return box, longitude, latitude


def DayNumber_to_Date(day_num, year):
    day_num.rjust(3 + len(day_num), '0')
 
    # converting to date
    date = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%m-%d-%Y")
 
    # printing result
    print("Resolved date : " + str(res))
    return date