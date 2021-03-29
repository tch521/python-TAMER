============
Installation
============

Python-TAMER was built using Python 3.8. It may still work on older versions
of python, but this has not been tested. 


Anaconda Users
--------------

The easiest way to install python-TAMER is with `Anaconda`_. 
First, ensure the `Dependencies`_ (listed below) are installed and up-to-date. 
This is most easily done using the `Anaconda Navigator`_.
Users opting to use conda commands in a terminal will first want to add the 
conda-forge channel to their environment::

    conda config --append channels conda-forge

They should then be able to install the the dependencies with the command::

    conda install numpy pandas matplotlib netcdf4 cartopy

.. _Anaconda: https://www.anaconda.com/
.. _Anaconda Navigator: https://docs.anaconda.com/anaconda/navigator/

Finally, python-TAMER can be installed by using pip within the conda environment::

    pip install python-tamer


Pip Users
---------

Pip users will have to ensure `Cartopy`_ is installed which is not a trivial operation
due to some dependencies it has. However, once that is done, installing python-TAMER 
*should* be as simple as the command::

    pip install python-tamer


Dependencies
------------

* `Cartopy`_
* `Matplotlib`_
* `NetCDF4`_
* `Numpy`_
* `Pandas`_


.. _Cartopy: https://scitools.org.uk/cartopy/docs/latest/
.. _Matplotlib: https://matplotlib.org/stable/users/installing.html
.. _NetCDF4: https://unidata.github.io/netcdf4-python/
.. _Numpy: https://numpy.org/install/
.. _Pandas: https://pandas.pydata.org/getting_started.html
