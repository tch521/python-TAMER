=======
History
=======

0.3.1 Open Alpha (2021-03-29)
---------------------------------

* Fixed errors in example code in documentation
* Fixed ``ER_Vernez_2015()`` function 
* Updated assumptions about units to reflect new dataset from Vuilleumier in UVI rather than W m-2
* Fixed issue where modifying ``map_options`` property would result in it being set to None


0.3.0 Open Alpha (2021-03-23)
---------------------------------

* Compiled and added to PyPI for easy public access
* Added standalone function for calculating ER using the Vernez 2015 method: ``ER_Vernez_2015()``
* Significantly expanded and standardised docstrings, adding examples
* Fixed error involving day selection in ``SpecificDoses`` class being one day late
* Added ``SpecificDoses.standard_column_names()`` function for standardising column names to ensure functionality 

0.2.0 Alpha (2021-03-11)
-----------------------------------

* Added documentation
* Added basic unit tests for each class (``SpecificDoses`` and ``ExposureMap``)
* Added histogram descriptor calculator functions to subroutines.
* Added map making functionality for ``ExposureMap`` class, limited consideration of ``map_options`` at this stage
* Fixed errors when working with single day test data (but anticipate further issues with this, to be fixed in a later release)


0.1.0 Pre-Alpha (2021-03-02)
--------------------------------------

* Alpha release on github only, no documentation and limited functionality
