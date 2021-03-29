=====
Usage
=====

Currently, python-TAMER is only compatiable with the `Vuilleumier et al. 2020 
<https://doi.org/10.1016/j.envint.2020.106177>`_ UV climatology for Switzerland. This dataset is 
currently only availabe on request, but will soon be released publicly. 

To use python-TAMER in a project::

    import python_tamer

There are currently two major classes in python-TAMER, ``SpecificDoses`` and ``ExposureMap``.
Each class has an associated workflow designed to produce tablular and map outputs respectively.
Every object created by one of these classes has two critical properties, ``data_directory``
and ``src_filename_format``. These describe the location and filename format of the Vuilleumier
et al. UV climatology. Users must ensure these properties are correct for each class object 
they create. Below we describe in general terms the approach taken by these classes. For
examples of raw code, see the :ref:`Code Reference`.


Calculating specific doses
--------------------------

The ``SpecificDoses`` class is essentially a table listing the key information to perform a dose
estimation. In the context of UV, this can be used in tandem with or as a substitute for 
tradition dosimetry measurements. See `Harris et al. 2021 <https://doi.org/10.3390/atmos12020268>`_
for more information on this approach.

To get started with the ``SpecificDoses`` class, prepare a pandas DataFrame (i.e. a table) with 
columns for ``Latitude``, ``Longitude``, ``Date``, ``ER``, and ``Schedule``. Each row of this 
table represents an exposed individual and the columns provide the corresponding details needed
to estimate their total daily UV dose. The ``ER`` column refers to the Exposure Ratio of the 
exposed individual. This information is not always readily available, and so the ``SpecificDoses``
class includes the ``ER_from_posture()`` function that calculates an ``ER`` column based on
``Posture`` and ``Anatomic_zone`` columns. Similarly, the ``Schedule`` column refers to the 
Exposure Schedule, a 24-length vector describing the proportion of each hour spent outdoors.
Often it is easier to simply list and start time and end time for a given exposure period,
so we have included the ``schedule_constant_exposure()`` function which generates a ``Schedule``
column based on ``Time_start`` and ``Time_end`` columns.

Once the SpecificDoses table is prepared, the ``calculate_specific_doses()`` function can be
applied to append ``Ambient_dose`` and ``Personal_dose`` columns to the table. 


Generating maps of exposure information
---------------------------------------

The ``ExposureMap`` class is designed to calculate more general information about doses or 
UV radiation intensity. The idea is to generate what we refer to as "pixel histograms", 
those are histograms of UV intensity or UV doses for each spatial pixel available in the 
dataset. By using histograms, python-TAMER is able to collect large distributions of data
per pixel with relatively low memory requirements, ensuring that these calculations can be
performed on personal computers with as little as 4GB of RAM. The data is loaded using the
``collect_data()`` function, which goes through the dataset one file at a time, loading 
the relevant data, performing any necessary calculations (such as applying an exposure
schedule to calculate a dose or performing unit conversions), and then discretising the
results and adding that information to the pixel histograms. 

Once the relevant data has been collected into pixel histograms, the ``calculate_map()``
function can be used to calculate a descriptor quantity for each histogram. These
descriptors can be simple, such as the mean or median of the histograms. They can also
be more advanced, such as percentiles or the standard deviation. This gives a single 
number per pixel, stored as an array, that can then be rendered as a colour map using
the ``plot_map()`` function.
