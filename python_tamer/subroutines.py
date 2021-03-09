import numpy as np

def assert_data_shape_24(data,reverse=False) :
    """Simple function to check if first dimension is 24 hours and, if not, reshapes accordingly
    """
    datashape = np.shape(data)
    # TO DO: Could be a big job, but this F ordering is weird and I should reconsider
    if datashape[0] != 24 and not reverse: # Checks that first dimension is length 24 (hours in a day) and reshapes if not
        new_shape = (24, datashape[0]//24) + datashape[1:]
    elif reverse :
        new_shape = [datashape[0] * datashape[1]] + list(datashape[2:])
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

    TrTim = date.apply(lambda x: x.toordinal() + 366).to_numpy() * 2.73785151e-05 - 18.9996356
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
    # TO DO: Need a replacement for this, but argument the responsibility of the user?
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
    # TO DO: Possibly unnecessary, but could probably improve efficiency of
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

