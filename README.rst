============================
Introduction to python-TAMER
============================

Advanced environmental exposure calculations made easy! 
Official documentation hosted at https://tch521.github.io/python-TAMER

The Toolkit for Analysis and Maps of Exposure Risk (TAMER) is designed to calculate estimates of individual and
population exposure across a geographic area. Currently, the project is focused on erythemal UV radiation 
exposure, but the tools provided by TAMER could be used in a variety of contexts provided there is appropriate
source data. In addition to providing a simple methods for basic exposure calculations, (such as mean, median,
and maximum intensities over certain time periods,) TAMER allows users to calculate daily doses by integrating
the exposure over time. TAMER deals with very large volumes of data, but is designed with memory efficiency in
mind so that such data can be processed on even modest personal comptuters.

In the context of UV, the dose received by an exposed individual is far more relevant to their corresponding 
health risk than the ambient level of UV. Usually, these doses are measured by wearable devices. Harris et al.
2021 (https://doi.org/10.3390/atmos12020268) explains the various benefits of instead estimating such doses
based on satellite-derived data with sufficiently high spatial and temporal resolution. The Swiss UV 
climatology provided by Vuilleumier et al. 2020 (https://doi.org/10.1016/j.envint.2020.106177) is Currently
the most appropriate source data for TAMER as it provides erythemal UV at an approximately 1.5km spatial
resolution and an hourly temporal resolution. Harris et al. 2021 shows that with location, date, and time
information, reasonable ambient doses can be calculated. However, to calculate personal doses, the Exposure
Ratio (ER) must also be known, that being the ratio between the ambient dose and the personal dose received
by a certain body part. Different body parts have varying ERs which also depend on body posture, for example
the ER of the forehead is lower when bowing down than when standing normally. TAMER includes a model from
Vernez et al. 2015 (https://doi.org/10.1038/jes.2014.6) to calculate ERs according to anatomic zone, posture,
and time of year. python-tamer.SpecificDoses is a table-like class designed to take location, date, time, 
posture, and anatomic zone information to calculate specific ambient and personal doses of the described
individuals. 

A large part of TAMER is dedicated to producing high quality maps of a variety of exposure metrics. Maps of UV
exposure often show the mean, median, or max irradiance for a given time period. TAMER includes the option to 
calculate such maps, but also offers more advanced alternatives. The TAMER approach balances versatility with
memory efficiency by calculating histograms for each pixel as a first step. These histograms can describe the
irradiance or the daily doses for any time selection and exposure schedule. They can be built up iteratively, 
processing one year at a time to ensure only moderate memory usage. With the pixel histograms calculated, the
user then has to choose a statistical descriptor to condense the distribution into a single number to be 
plotted on the map. This can be basic statistics such as mean, median, or max, however we include some more
advanced options such as a custom percentile and the standard deviation. In a forthcoming release, we shall
also include the option to define one's own formula for a custom descriptor, allowing for metrics like the
difference between the 95th percentile and the median divided by the standard deviation which would be 
indicative of the severity of acute exposure instances. The simple and novel approaches to exposure estimation
provided by the combined release of high resolution UV data (https://doi.org/10.1016/j.envint.2020.106177) and
the simple and novel exposure calculations provided by TAMER give opportunity to epidemiologists and public 
health experts to study UV exposure with higher detail than has ever been possible before.


* Free software: BSD-3-Clause license


Features
^^^^^^^^

* Calculate daily doses rapidly with custom exposure schedules
* Analyse exposure distributions per pixel
* Produce maps to represent chronic and acute exposure using standard or custom metrics
* Replicate dosimetry measurements using Exposure Ratio modelling

In Development
^^^^^^^^^^^^^^

* Improved support for custom statistical descriptors
* Custom area selection for the SpecificDoses class
* Improved aesthetic options for ExposureMap class

Future work
^^^^^^^^^^^

* Improved support for different source files (new units, temporal resolutions, etc.)
* Integrate support for cross multiplication of ExposureMap with population distribution data