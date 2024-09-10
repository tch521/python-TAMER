import datetime
import os
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import pandas as pd
import re
import processing
import subprocess
import time
from   osgeo            import gdal
from   matplotlib       import pyplot as plt
from   qgis.PyQt.QtCore import QDate, QTime
from   qgis.PyQt.QtGui  import QColor
from   qgis.core        import (QgsProject, 
                                QgsRasterLayer, 
                                QgsVectorLayer, 
                                QgsCoordinateReferenceSystem, 
                                QgsDataSourceUri,
                                QgsFeatureRequest, 
                                QgsGeometry,
                                QgsRectangle,
                                QgsPointCloudLayer,
                                QgsRaster, 
                                QgsPointXY,
                                QgsGradientColorRamp,
                                QgsColorRampShader,
                                QgsSingleBandPseudoColorRenderer,
                                QgsGradientColorRamp,
                                QgsRasterShader)


'''
Launch this script within QGIS console to avoid issues with processing tools libraries.

Choose the corrisponding paths to your raster folders and to your output folders.

This script requires 3 ingredients to be present in the input raster folder:
    - ground + buildings digital surface model (dsm file)
    - ground + trees and ground only digital surface model (for cdsm calculation)
    - ground - buildings digital surface model for mask calculation
These files are created from point cloud lidar data provided by SwissTopo and processed by the
RasterisePointCloud(pc_dir, raster_dir, crs, project) function
    '''

# Define the coordinate reference system (CRS) and create a QGIS project
def set_crs(epsg_code="EPSG:2056"):
    """
    Set the coordinate reference system (CRS) for the QGIS project.

    This function defines the CRS that will be used throughout the project. 
    The EPSG code is a standard code that represents specific coordinate reference systems. 
    For example, EPSG:2056 corresponds to the Swiss Coordinate Reference System (CH1903+ / LV95),
    which is commonly used in Switzerland for accurate geospatial data representation.

    Args:
        epsg_code (str): EPSG code for the CRS. Default is "EPSG:2056" for SwissTopo data.
    
    Returns:
        crs (QgsCoordinateReferenceSystem): The coordinate reference system object.
        project (QgsProject): The QGIS project instance.
    
    Explanation:
        - The function creates a `QgsCoordinateReferenceSystem` object using the provided EPSG code.
        - It then retrieves the current QGIS project instance using `QgsProject.instance()`.
        - The CRS is applied to the project by setting it with `project.setCrs(crs)`.
        - Finally, the function returns both the `crs` and `project` objects.
    
    Example Usage:
        crs, project = set_crs("EPSG:4326")
        This would set the CRS to WGS84, a commonly used global coordinate reference system.
    """
    # Create the CRS object using the specified EPSG code
    crs = QgsCoordinateReferenceSystem(epsg_code)
    
    # Retrieve the current QGIS project instance
    project = QgsProject.instance()
    
    # Set the project CRS to the specified CRS
    project.setCrs(crs)
    
    # Return both the CRS and the project instance for later use
    return crs, project

# Load a raster layer
def load_raster_layer(file_path, crs, project):
    """
    Load a raster layer into the QGIS project.

    This function loads a raster file (e.g., a GeoTIFF) into the current QGIS project. 
    It assigns the specified coordinate reference system (CRS) and adds the layer to the QGIS project. 
    Raster layers are commonly used for representing continuous data such as elevation, 
    temperature, or satellite imagery.

    Args:
        file_path (str): The full path to the raster file on disk (e.g., "/path/to/raster.tif").
        crs (QgsCoordinateReferenceSystem): The CRS object that defines the spatial 
        reference of the raster layer.
        project (QgsProject): The QGIS project instance where the layer will be added.

    Returns:
        rlayer (QgsRasterLayer): The loaded raster layer object, or None if loading fails.

    Explanation:
        - The function extracts the base name of the raster file (without the ".tif" extension) 
          to use as the layer's display name in QGIS.
        - It then creates a `QgsRasterLayer` object using the provided file path and layer name.
        - The function checks if the layer is valid (i.e., it was successfully loaded). 
          If the layer is not valid, an error message is printed.
        - If the layer is valid, it is added to the QGIS project using `project.addMapLayer()`, 
          and the CRS is set with `rlayer.setCrs(crs)`.
        - Finally, the function returns the `rlayer` object, which represents the loaded raster layer.

    Example Usage:
        rlayer = load_raster_layer("/path/to/dem.tif", crs, project)
        This would load the DEM (Digital Elevation Model) raster into the QGIS project using the specified CRS.
    """
    # Extract the base name of the file to use as the layer name in QGIS
    layer_name = os.path.basename(file_path).replace(".tif", "")
    
    # Create a QgsRasterLayer object using the file path and extracted layer name
    rlayer = QgsRasterLayer(file_path, layer_name)
    
    # Check if the layer was successfully loaded
    if not rlayer.isValid():
        print(f"Failed to load layer: {layer_name}")
        return None
    else:
        # If valid, add the layer to the QGIS project
        project.addMapLayer(rlayer)

        # Set the specified CRS for the layer
        rlayer.setCrs(crs)
        print(f"Layer loaded: {rlayer.name()}")
    
    # Return the loaded raster layer, or None if loading failed
    return rlayer

# Create a directory if it doesn't exist
def create_directory(path):
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

# create raster layers from point cloud file 
def RasterisePointCloud(pc_dir, raster_dir, crs, project):
    """
    Converts a directory of LiDAR point cloud files into various raster layers representing 
    different surface models.

    Args:
        pc_dir (str): The directory containing point cloud (.las) files.
                      Example: '/home/lmartinell/uv/data/GeoData/Lausanne/LidarData/'
        raster_dir (str): The directory to store the generated raster files.
                          Example: '/home/lmartinell/uv/data/GeoData/Lausanne/Rasters/'
        crs (QgsCoordinateReferenceSystem): The coordinate reference system to assign to 
        the point cloud and raster layers.
        project (QgsProject): The QGIS project instance where layers are loaded.

    Returns:
        None

    Workflow:
        1. The function scans the `pc_dir` directory for `.las` point cloud files.
        2. Each point cloud file is loaded as a QgsPointCloudLayer and assigned the specified CRS.
        3. The function filters the point cloud data based on classification 
           (e.g., ground, buildings, vegetation) to create raster layers:
            - DSM (Digital Surface Model) including ground and buildings.
            - Ground + vegetation model.
            - Ground model excluding buildings.
            - DTM (Digital Terrain Model) representing only the ground.
        4. The `make_cdsm` function is called to generate the Canopy Digital Surface Model (CDSM) 
        based on the generated rasters.

    Example Usage:
        RasterisePointCloud('/path/to/lidar/', '/path/to/output_rasters/', crs, QgsProject.instance())
    """

    # Retrieve all file names in the specified point cloud directory
    names = os.listdir(pc_dir)
    # Sort file names alphabetically to process them in a consistent order
    names.sort()

    # Loop through each file in the directory
    for name in names:
        # Process only .las files (LiDAR data format)
        if name.endswith(".las"):
            # Construct the full file path
            file = os.path.join(pc_dir, name)
            # Load the point cloud file as a QGIS point cloud layer
            pc_layer = QgsPointCloudLayer(file, name, "pdal")

            # Check if the layer loaded successfully
            if not pc_layer.isValid():
                print("Layer failed to load!")
                return
            
            # Assign the specified CRS to the point cloud layer
            pc_layer.setCrs(crs)
            # Extract the tile identifier from the file name (excluding the extension)
            tile_ind = str(pc_layer.name().split('.')[0])
            print("%s layer loaded" % tile_ind)

            # Classification filters based on SwissTopo data:
            ground_building_filter = 'Classification = 2 OR Classification = 6'  # Ground and buildings
            ground_trees_filter    = 'Classification = 2 OR Classification = 3'  # Ground and small vegetation
            ground_filter          = 'Classification = 2'                        # Ground only

            # Create an output directory for the raster files, if it doesn't exist
            output_dir = os.path.join(raster_dir, tile_ind)
            create_directory(output_dir)

            # Generate a Digital Surface Model (DSM) raster including ground and buildings
            filter_params = {
                'INPUT'            : 'pdal://' + file,        # PDAL input requires the 'pdal://' prefix
                'RESOLUTION'       : 1,                       # Set the raster resolution (1 meter per pixel)
                'TILE_SIZE'        : 1000,                    # Tile size for processing
                'FILTER_EXPRESSION': ground_building_filter,  # Filter ground and building points
                'FILTER_EXTENT'    : None,                    # No extent filtering; process the full dataset
                'ORIGIN_X'         : None,                    # No specific origin
                'ORIGIN_Y'         : None,                    # No specific origin
                'OUTPUT'           : os.path.join(output_dir, tile_ind + '_dsm.tif')  # Output DSM file path
            }
            # Run the processing algorithm to export the DSM raster
            processing.run("pdal:exportrastertin", filter_params)
            load_raster_layer(filter_params['OUTPUT'], crs, project)

            # Update parameters to create a raster combining ground and vegetation
            filter_params['FILTER_EXPRESSION'] = ground_trees_filter
            filter_params['OUTPUT']            = os.path.join(output_dir, tile_ind + '_grd_trs.tif')
            processing.run("pdal:exportrastertin", filter_params)
            load_raster_layer(filter_params['OUTPUT'], crs, project)

            # Generate a raster of the ground without buildings, using a different PDAL function
            # that doesn't interpolate between points. 
            # In this way, in place of buildings we will have holes. 
            filter_params['OUTPUT'] = os.path.join(output_dir, tile_ind + '_grd_nob.tif')
            processing.run("pdal:exportraster", filter_params)
            load_raster_layer(filter_params['OUTPUT'], crs, project)

            # Update parameters to create a Digital Terrain Model (DTM) raster using only ground points
            filter_params['FILTER_EXPRESSION'] = ground_filter
            filter_params['OUTPUT']            = os.path.join(output_dir, tile_ind + '_grd.tif')
            processing.run("pdal:exportrastertin", filter_params)
            load_raster_layer(filter_params['OUTPUT'], crs, project)

            # Generate the Canopy Digital Surface Model (CDSM) using the `make_cdsm` function
            make_cdsm(raster_dir, tile_ind, crs, project)
    
    # End of function
    return

def make_cdsm(raster_dir, tile_index, crs, project):
    """
    Create a Canopy Digital Surface Model (CDSM) for a specific tile 
    by subtracting the Digital Elevation Model (DEM) from the Digital Surface Model (DSM).

    The CDSM represents the relative height of vegetation above ground level, 
    which is a key input for environmental and urban climate models like UMEP.

    Args:
        raster_dir (str): Directory containing the input raster files.
        tile_index (str): Index of the tile to be processed.
        crs (QgsCoordinateReferenceSystem): The coordinate reference system for the rasters.
        project (QgsProject): The QGIS project instance to which the layers will be added.

    Returns:
        str: Path to the generated CDSM raster file, or None if the operation fails.

    Workflow:
        1. The function constructs the file paths for the vegetation DSM and ground DEM rasters.
        2. It checks if both raster files exist and loads them into the QGIS project.
        3. If both files are available, the function calculates the CDSM by subtracting the 
           DEM from the DSM using the GDAL raster calculator.
        4. The resulting CDSM raster is loaded into the QGIS project and its file path is returned.
    """

    # Construct the file path for the vegetation DSM (ground + trees) raster
    vegetation_file = os.path.join(raster_dir, tile_index, tile_index + '_grd_trs.tif')
    if os.path.isfile(vegetation_file):
        # Load the vegetation DSM raster into the QGIS project
        load_raster_layer(vegetation_file, crs, project)
    else:
        print(f'No DSM file with vegetation for tile {tile_index}')
        return None  # Exit the function if the vegetation DSM file is missing
    
    # Construct the file path for the ground DEM (Digital Elevation Model) raster
    ground_file = os.path.join(raster_dir, tile_index, tile_index + '_grd.tif')
    if os.path.isfile(ground_file):
        # Load the ground DEM raster into the QGIS project
        load_raster_layer(ground_file, crs, project)
    else:
        print(f'No DEM file for tile {tile_index}')
        return None  # Exit the function if the ground DEM file is missing
    
    # Define the output file path for the CDSM (Canopy Digital Surface Model)
    output_cdsm = os.path.join(raster_dir, tile_index, f"{tile_index}_cdsm.tif")
    
    # Use the GDAL raster calculator to subtract the ground DEM from the vegetation DSM
    proc = processing.run("gdal:rastercalculator", 
                          {'INPUT_A'   : vegetation_file, # DSM (ground + vegetation)
                           'BAND_A'    : 1,               # Use the first band of the DSM
                           'INPUT_B'   : ground_file,     # DEM (ground only)
                           'BAND_B'    : 1,               # Use the first band of the DEM
                           'FORMULA'   : 'A-B',           # Subtract DEM from DSM
                           'NO_DATA'   : None,            # Keep the original NoData values
                           'EXTENT_OPT': 0,               # Use the extent of the first input raster
                           'PROJWIN'   : None,            # Process the full extent of the rasters
                           'RTYPE'     : 5,               # Output as Float32
                           'OPTIONS'   : '',              # No additional options
                           'EXTRA'     : '',              # No extra GDAL options
                           'OUTPUT'    : output_cdsm})    # Path to save the output CDSM
    
    # Load the generated CDSM raster into the QGIS project
    load_raster_layer(proc['OUTPUT'], crs, project)
    
    # Return the path to the CDSM raster
    return proc['OUTPUT']

def apply_custom_pseudocolor_renderer(raster_layer, color_ramp="green", max_height=25):
    """
    Apply a custom Singleband Pseudocolor rendering with a specified color ramp to the given raster layer.
    The color ramp can be green, red, or custom-defined. The maximum height value controls the upper limit of the color ramp.

    Args:
        raster_layer (QgsRasterLayer): The raster layer to apply the renderer to.
        color_ramp (str): The color ramp to use ("green", "red", or "custom").
        max_height (int): The maximum height value to use in the color ramp.

    Returns:
        bool: True if the renderer was successfully applied, False otherwise.
    """

    if not isinstance(raster_layer, QgsRasterLayer) or not raster_layer.isValid():
        print("Invalid raster layer provided.")
        return False

    # Define color ramp items based on the selected color ramp
    if color_ramp == "green":
        color_ramp_items = [
            QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 255), "0m"),  # White at 0m
            QgsColorRampShader.ColorRampItem(max_height * 0.5, QColor(144, 238, 144), f"{max_height * 0.5}m"),  # Light Green at mid-height
            QgsColorRampShader.ColorRampItem(max_height, QColor(0, 100, 0), f"{max_height}m")  # Dark Green at max height
        ]
    elif color_ramp == "red":
        color_ramp_items = [
            QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 255), "0m"),  # White at 0m
            QgsColorRampShader.ColorRampItem(max_height * 0.5, QColor(255, 160, 122), f"{max_height * 0.5}m"),  # Light Red at mid-height
            QgsColorRampShader.ColorRampItem(max_height, QColor(139, 0, 0), f"{max_height}m")  # Dark Red at max height
        ]
    else:
        print(f"Unsupported color ramp: {color_ramp}. Defaulting to green.")
        return apply_custom_pseudocolor_renderer(raster_layer, "green", max_height)

    # Create a QgsColorRampShader and set the color ramp items
    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Interpolated)
    color_ramp_shader.setColorRampItemList(color_ramp_items)

    # Wrap the color ramp shader in a QgsRasterShader
    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(color_ramp_shader)

    # Set the raster band (usually the first band)
    raster_band = 1

    # Create a renderer
    renderer = QgsSingleBandPseudoColorRenderer(
        raster_layer.dataProvider(), raster_band, raster_shader
    )

    # Apply the renderer to the raster layer
    raster_layer.setRenderer(renderer)

    # Refresh the layer to apply the changes
    raster_layer.triggerRepaint()

    print(f"Custom {color_ramp} renderer applied successfully.")
    return True

def merge_and_clip_rasters(tile_list, lat, lon, output_path, crs, project):
    """
    Merge and clip a 3x3 grid of adjacent raster tiles centered on a specified tile.

    The function ensures that shadows cast by large buildings or trees just outside the central tile 
    are accurately accounted for by expanding the clipped area by 100 meters in all directions. 
    This prevents edge effects from influencing the analysis within the central tile.

    Args:
        tile_list (list)                  : List of file paths for the tiles to merge.
        lat (int)                         : Latitude identifier for the central tile.
        lon (int)                         : Longitude identifier for the central tile.
        output_path (str)                 : File path where the merged and clipped raster will be saved.
        crs (QgsCoordinateReferenceSystem): The coordinate reference system for the output raster.
        project (QgsProject)              : The QGIS project instance where the layers will be added.

    Returns:
        str: The path to the merged and clipped raster, or None if the operation fails.

    Workflow:
        1. Check if there are any tiles to merge. If not, print a message and return None.
        2. Merge the tiles using GDAL’s merge tool.
        3. Clip the merged raster around the central tile, with a buffer of 100m on each side.
        4. Load both the merged and the clipped raster into the QGIS project.
        5. Return the path to the final raster, or None if the operation fails at any step.
    """

    # Step 1: Validate that there are tiles to merge
    if not tile_list:
        print(f"No tiles to merge for {output_path}.")
        return None
    
    # Step 2: Merge the tiles using the GDAL merge tool
    print('Merging the following tiles:')
    print(*tile_list, sep="\n")
    
    merge_res = processing.run("gdal:merge",
                               {'INPUT'        : tile_list,    # List of input raster files
                                'PCT'          : False,        # Preserve color tables (not needed here)
                                'SEPARATE'     : False,        # Merge bands together
                                'NODATA_INPUT' : None,         # NoData value for input (can be set if needed)
                                'NODATA_OUTPUT': None,         # NoData value for output (can be set if needed)
                                'OPTIONS'      : '',           # Additional GDAL options (none required here)
                                'EXTRA'        : '',           # Extra command-line options
                                'DATA_TYPE'    : 1,            # Byte data type
                                'OUTPUT'       : 'TEMPORARY_OUTPUT'})  # Temporary output file
    
    # Step 3: Validate that merging was successful
    if 'OUTPUT' not in merge_res:
        print(f"Merge failed for {output_path}.")
        return None
    
    # Load the merged raster into the QGIS project for validation
    load_raster_layer(merge_res['OUTPUT'], crs, project)
    
    # Step 4: Define the extent for clipping the merged raster
    print('Clipping the merged raster output.')
    clip_extent = (f"{lat * 1000 - 100},{lat * 1000 + 1100},"  # Latitude  range with 100m buffer
                   f"{lon * 1000 - 100},{lon * 1000 + 1100}"   # Longitude range with 100m buffer
                   "[EPSG:2056]")
    
    # Step 5: Clip the merged raster to the defined extent
    clip_res = processing.run("gdal:cliprasterbyextent", 
                              {'INPUT'    : merge_res['OUTPUT'],  # Merged raster file
                               'PROJWIN'  : clip_extent,          # Clipping extent
                               'OVERCRS'  : False,                # No CRS transformation
                               'NODATA'   : None,                 # NoData value (optional)
                               'OPTIONS'  : '',                   # Additional GDAL options
                               'DATA_TYPE': 0,                    # Same data type as input
                               'EXTRA'    : '',                   # Extra command-line options
                               'OUTPUT'   : output_path})         # Output file path
    
    # Step 6: Validate that clipping was successful
    if 'OUTPUT' not in clip_res:
        print(f"Clip failed for {output_path}.")
        return None
    
    # Load the clipped raster into the QGIS project
    load_raster_layer(clip_res['OUTPUT'], crs, project)
    
    # Return the path to the final clipped raster
    return output_path

def process_tiles(tile_index, raster_dir, crs, project):
    """
    Process and merge adjacent raster tiles based on the given tile index.
    If the merged and clipped DSM and CDSM tiles already exist, they are loaded directly.
    If not, the function merges and clips the necessary tiles to create them.

    Args:
        tile_index (str): The tile index in the format 'lat_lon' (e.g., '2534_1158').
        raster_dir (str): The directory containing the raster tiles.
        crs (QgsCoordinateReferenceSystem): The coordinate reference system for the rasters.
        project (QgsProject): The QGIS project instance.

    Returns:
        tuple: Paths to the merged and clipped DSM and CDSM tiles, or None if the operation fails.
    """
    
    # Define paths for the output merged and clipped DSM and CDSM tiles
    merged_clipped_dsm  = os.path.join(raster_dir, f"{tile_index}", f"{tile_index}_mrg_dsm.tif")
    merged_clipped_cdsm = os.path.join(raster_dir, f"{tile_index}", f"{tile_index}_mrg_cdsm.tif")

    # Check if the merged and clipped tiles already exist
    if os.path.exists(merged_clipped_dsm) and os.path.exists(merged_clipped_cdsm):
        print(f"{merged_clipped_dsm} and {merged_clipped_cdsm} already exist.")
        print('Loading them into the project...')
        load_raster_layer(merged_clipped_dsm,  crs, project)
        load_raster_layer(merged_clipped_cdsm, crs, project)
        return merged_clipped_dsm, merged_clipped_cdsm

    # Extract latitude and longitude from the tile index 
    # (e.g., '2534_1158' -> lat=2534, lon=1158)
    lat, lon = int(tile_index[:4]), int(tile_index[5:9])
    
    # Lists to hold the paths of neighboring DSM and CDSM tiles
    dsm_tile_list  = []
    cdsm_tile_list = []

    # Loop through neighboring tiles in a 3x3 grid (including the central tile)
    for lat_offset in range(-1, 2):
        for lon_offset in range(-1, 2):
            neighbor_lat       = lat + lat_offset
            neighbor_lon       = lon + lon_offset
            neighbor_name_dsm  = f"{neighbor_lat:04d}_{neighbor_lon:04d}_dsm.tif"
            neighbor_name_cdsm = f"{neighbor_lat:04d}_{neighbor_lon:04d}_cdsm.tif"
            neighbor_dir       = os.path.join(raster_dir, f"{neighbor_lat:04d}_{neighbor_lon:04d}/")

            # Verify existence of neighboring DSM and CDSM tiles
            if os.path.exists(neighbor_dir):
                # Process DSM tiles
                dsm_tile_path = os.path.join(neighbor_dir, neighbor_name_dsm)
                if os.path.isfile(dsm_tile_path):
                    load_raster_layer(dsm_tile_path, crs, project)
                    dsm_tile_list.append(dsm_tile_path)
                    print(f"Neighbor DSM tile: {dsm_tile_path}")

                # Process CDSM tiles
                cdsm_tile_path = os.path.join(neighbor_dir, neighbor_name_cdsm)
                if os.path.isfile(cdsm_tile_path):
                    load_raster_layer(cdsm_tile_path, crs, project)
                    cdsm_tile_list.append(cdsm_tile_path)
                    print(f"Neighbor CDSM tile: {cdsm_tile_path}")
                else:
                    # Attempt to create the CDSM if it does not exist
                    new_cdsm = make_cdsm(neighbor_dir, f"{neighbor_lat:04d}_{neighbor_lon:04d}", crs, project)
                    if new_cdsm is not None:
                        load_raster_layer(new_cdsm, crs, project)
                        cdsm_tile_list.append(new_cdsm)
                        print(f"Generated new CDSM tile: {new_cdsm}")

    # Warning if the number of DSM or CDSM tiles is insufficient for a 3x3 grid
    if len(dsm_tile_list) < 9 or len(cdsm_tile_list) < 9:
        print("Warning: 3x3 composite tile is incomplete. Results may be affected.")

    # Sort the tile lists to ensure a consistent merge order
    dsm_tile_list.sort()
    cdsm_tile_list.sort()

    # Merge and clip the DSM tiles
    merged_clipped_dsm  = merge_and_clip_rasters(dsm_tile_list,  lat, lon, merged_clipped_dsm,  crs, project)

    # Merge and clip the CDSM tiles
    merged_clipped_cdsm = merge_and_clip_rasters(cdsm_tile_list, lat, lon, merged_clipped_cdsm, crs, project)
    
    return merged_clipped_dsm, merged_clipped_cdsm

# Run shadow generator
def run_shadow_generator(date, input_cdsm, input_dsm, output_dir, startyear, itertime=30, include_cdsm=True):
    """
    Run UMEP's shadow generator to calculate shadows for a given date.

    The function uses the UMEP (Urban Multi-scale Environmental Predictor) model's 
    shadow generator to simulate shadows for a specific date and time. The calculation 
    can include or exclude vegetation data (Canopy Digital Surface Model - CDSM), depending on the input. 
    This is crucial for accurate modeling in urban environments, where shadows cast by both 
    buildings and trees can significantly affect microclimates.

    Args:
        date (datetime)                    : The specific date for which shadows are calculated.
        input_cdsm (str)                   : File path to the Canopy Digital Surface Model (CDSM), representing tree heights.
        input_dsm (str)                    : File path to the Digital Surface Model (DSM), representing building heights.
        output_dir (str)                   : Directory where the generated shadow files will be stored.
        startyear (int)                    : The starting year of the simulation period, used to determine vegetation density.
        itertime (int)                     : The time interval in minutes between shadow calculations (default is 30 minutes).
        include_cdsm (bool)                : Whether to include the CDSM in the shadow calculation (default is True).

    Returns:
        str: The path to the final shadow file with the correct suffix, or None if the operation fails.
    """

    # Step 1: Convert the date to a QDate object for use with QGIS processing algorithms
    date_str  = date.strftime("%d-%m-%Y")  # Format the date as a string (e.g., "15-05-2024")
    datetorun = QDate.fromString(date_str, "d-M-yyyy")  # Convert the string to a QDate object

    # Step 2: Determine the light transmission factor (transVeg) based on the date
    # This factor adjusts the shadow effect of vegetation, accounting for seasonal changes
    if QDate(startyear, 4, 15) < datetorun < QDate(startyear, 9, 15):
        transVeg = 3  # Dense foliage in summer reduces light transmission
    elif (QDate(startyear, 3, 15) < datetorun < QDate(startyear, 4, 15) or
          QDate(startyear, 9, 15) < datetorun < QDate(startyear, 10, 15)):
        transVeg = 15  # Transitional periods with moderate foliage
    elif (QDate(startyear, 3, 1)   < datetorun < QDate(startyear, 3, 15) or
          QDate(startyear, 10, 15) < datetorun < QDate(startyear, 11, 15)):
        transVeg = 25  # Early spring or late autumn with minimal foliage
    else:
        transVeg = 50  # Winter period with little to no foliage

    # UTC offset for the simulation (default is 0 for simplicity)
    utc = 0

    # Step 3: Prepare the suffix for the output file based on the inclusion of CDSM
    suffix = "_With_CDSM" if include_cdsm else "_Without_CDSM"

    # Step 4: Define the input parameters for the UMEP shadow generator algorithm
    params_in = {
        'DATEINI'      : datetorun,                         # Date for the shadow simulation
        'DST'          : False,                             # Daylight Saving Time flag (not considered here)
        'INPUT_ASPECT' : None,                              # Aspect input (not used in this case)
        'INPUT_CDSM'   : input_cdsm if include_cdsm else None,  # Include CDSM if specified
        'INPUT_DSM'    : input_dsm,                         # DSM input representing buildings
        'INPUT_HEIGHT' : None,                              # Height input (not used here)
        'INPUT_TDSM'   : None,                              # Tree DSM input (not used here)
        'INPUT_THEIGHT': 25,                                # Default tree height (in meters)
        'ITERTIME'     : int(itertime),                     # Time interval for shadow calculations
        'ONE_SHADOW'   : False,                             # Generate multiple shadows throughout the day
        'OUTPUT_DIR'   : output_dir,                        # Directory for storing output shadow files
        'TIMEINI'      : QTime(8, 0, 0),                    # Start time for the simulation (08:00 AM)
        'TRANS_VEG'    : transVeg,                          # Vegetation light transmission factor
        'UTC'          : utc                                # UTC offset (no adjustment for time zone)
    }

    # Step 5: Run the UMEP shadow generator using the defined parameters
    print(f"[DEBUG] Running shadow generator for date: {date_str} at {params_in['TIMEINI'].toString()} with suffix: {suffix}")
    processing.run("umep:Solar Radiation: Shadow Generator", params_in)

    # Step 6: Retrieve all shadow files generated in the output directory
    all_shadow_files = sorted([f for f in os.listdir(output_dir) if f.startswith("Shadow_") and f.endswith(".tif")])
    print(f"[DEBUG] All shadow files: {all_shadow_files}")
    
    # Step 7: Rename the most recent shadow file to include the correct suffix (_With_CDSM or _Without_CDSM)
    for file in all_shadow_files:
        if not any(x in file for x in ["_With_CDSM", "_Without_CDSM"]):
            new_name = file.replace(".tif", f"{suffix}.tif")
            os.rename(os.path.join(output_dir, file), os.path.join(output_dir, new_name))
            print(f"[DEBUG] Renamed shadow file to: {new_name}")

    # Step 8: Construct the final shadow file name with the correct date, time, and suffix
    final_shadow_file = f"Shadow_{date.strftime('%Y%m%d_%H%M')}{suffix}.tif"
    final_shadow_path = os.path.join(output_dir, final_shadow_file)
    print(f"[DEBUG] Final shadow file path: {final_shadow_path}")
    
    # Return the path to the final shadow file
    return final_shadow_path



# Calculate masks
def calculate_mask(input_raster, output_raster, extent=None):
    """
    Calculate a mask for the given input raster, excluding building areas.

    The purpose of this mask is to focus on areas outside of buildings for 
    subsequent analysis, such as calculating the shadowing effect on open spaces.

    Args:
        input_raster (str)    : Path to the input raster file.
        output_raster (str)   : Path to the output mask file.
        extent (str, optional): The extent to be used for the mask calculation 
                                (in "xmin, xmax, ymin, ymax" format). If None, 
                                the entire raster extent is used.

    Explanation:
        - The function loads the input raster and verifies its validity.
        - A raster calculator expression (`mask_expr`) is used to set the 
          non-building areas (where raster values are greater than 0) to 1 
          while excluding the building areas (where raster values are 0 or negative).
        - The resulting mask is saved to the specified output path.

    Output:
        - A raster mask that highlights non-building areas with a value of 1, 
          and excludes building areas. The mask is saved as a new raster file.

    Example Usage:
        calculate_mask("input_raster.tif", "output_mask.tif", "2535000,2536000,1156000,1157000")

    """

    # Load the input raster layer to verify its validity
    rlayer = QgsRasterLayer(input_raster, "input_raster")
    if not rlayer.isValid():
        print("Failed to load input raster!")
        return

    # Expression to generate the mask: Set all positive values to 1, zero or negative to 0
    mask_expr = "(A > 0) / (A > 0)"
    
    # Define processing parameters
    processing_params = {
        'INPUT_A': input_raster,  # The input raster file
        'BAND_A' : 1,             # The band to process (usually band 1)
        'FORMULA': mask_expr,     # The mask expression
        'NO_DATA': None,          # No data value for the output
        'RTYPE'  : 5,             # Output data type (Float32)
        'OUTPUT' : output_raster  # The output mask file path
    }

    # If an extent is provided, apply the mask only within that extent
    if extent is not None:
        processing_params['PROJWIN'] = f"{extent}[EPSG:2056]"  # Define the clipping extent

    # Run the raster calculator to generate the mask
    processing.run("gdal:rastercalculator", processing_params)
    print(f"Mask created at {output_raster}")

# Apply mask
def apply_mask(input_mask, shadow_file, dir_out, extent, tile_ind, crs, project):
    """
    Apply a mask to a raster file and exclude specific areas (e.g., building surfaces).

    This function takes an input mask and applies it to a shadow raster file, effectively 
    excluding certain areas (such as building surfaces) from further analysis. The masked 
    raster is then saved with a descriptive filename, loaded into the QGIS project, and 
    returned for subsequent processing.

    Args:
        input_mask (str)                    : File path to the mask raster (e.g., building footprint mask).
        shadow_file (str)                   : File path to the shadow raster generated by the shadow generator.
        dir_out (str)                       : Directory where the masked raster will be saved.
        extent (str)                        : Extent of the area to process, defined in EPSG:2056 coordinates.
        tile_ind (str)                      : Tile index identifier (e.g., '2535_1153'), used in the output filename.
        crs (QgsCoordinateReferenceSystem)  : Coordinate reference system for the output raster.
        project (QgsProject)                : The QGIS project instance where the layers will be added.

    Returns:
        str: The path to the masked raster file, or None if the operation fails.
    """

    # Step 1: Extract the filename from the shadow raster path for reference
    shadow_filename = os.path.basename(shadow_file)
    print(f"Original shadow_file: {shadow_filename}")

    # Step 2: Define the expression to apply the mask, using the formula 'A * B'
    # 'A' represents the mask raster, and 'B' represents the shadow raster.
    mask_expr = 'A * B'
    proc = processing.run("gdal:rastercalculator", {
        'INPUT_A': input_mask,                       # Mask raster input (e.g., building footprints)
        'BAND_A': 1,                                 # Use the first band of the mask raster
        'INPUT_B': os.path.join(dir_out, shadow_filename),  # Shadow raster input
        'BAND_B': 1,                                 # Use the first band of the shadow raster
        'FORMULA': mask_expr,                        # Formula for masking (multiply A and B)
        'NO_DATA': None,                             # NoData value (optional)
        'EXTENT_OPT': 0,                             # Use input raster's extent
        'PROJWIN': f"{extent}[EPSG:2056]",           # Geographic extent for processing
        'RTYPE': 5,                                  # Output raster data type (32-bit float)
        'OUTPUT': 'TEMPORARY_OUTPUT'                 # Output is temporary until saved
    })

    # Step 3: Construct a new filename for the masked raster, ensuring it reflects the processing
    # Avoid multiple appends by checking if sub-tile index is already present
    base_name = f"{tile_ind}_masked_Shadow"
    if base_name in shadow_filename:
        output_filename = shadow_filename
    else:
        output_filename = f"{tile_ind}_masked_{shadow_filename}"
    
    # Ensure no repetitive sub-tile indices
    parts = output_filename.split('_')
    unique_parts = []
    seen_parts = set()
    for part in parts:
        if part not in seen_parts:
            unique_parts.append(part)
            seen_parts.add(part)
    output_filename = '_'.join(unique_parts)
    
    output_path = os.path.join(dir_out, output_filename)
    print(f"Generated output_filename: {output_filename}")
    print(f"Final output_path: {output_path}")

    # Step 4: Save the masked raster to the specified output directory
    gdal.Translate(output_path, proc['OUTPUT'])

    # Step 5: Load the masked raster into the QGIS project
    layer = load_raster_layer(output_path, crs, project)
    
    # Optionally rename the layer in QGIS to exclude the .tif extension for better readability
    layer.setName(output_filename.replace(".tif", ""))

    # Step 6: Return the path to the final masked raster
    return output_path




def analyze_shadow(file_path, file, date_data):
    """
    Analyze the shadow data from the given file and update the date_data dictionary.

    This function reads a shadow raster file, extracts the shadow information, calculates the 
    average shading factor, and updates a dictionary that stores the shading data. The function 
    assumes that the raster file is named in a way that includes the date and time of the shadow data.

    Args:
        file_path (str): Path to the shadow raster file.
        file (str): Filename of the shadow raster.
        date_data (dict): Dictionary to store shadow analysis results, organized by time.

    Returns:
        dict: Updated date_data dictionary with the average shading factor for each time period analyzed.

    Workflow:
        1. Extract relevant information (date, time) from the raster file name.
        2. Parse the extracted date and time into a datetime object.
        3. Load the raster file using GDAL and access the shadow data.
        4. Handle any NoData values by converting them to NaN.
        5. Calculate the average shading factor using numpy's nanmean function.
        6. Update the date_data dictionary with the calculated shading factor.
        7. Return the updated dictionary for further analysis.
    """

    # Step 1: Extract the base name of the file for easier reference
    print(f"[DEBUG] Analyzing shadow file: {file_path}")
    file_name = os.path.basename(file_path)
    
    # Split the filename into components based on underscores
    file_parts = file_name.split('_')
    print(f"[DEBUG] Filename parts: {file_parts}")

    try:
        # Step 2: Extract the date and time parts based on the filename structure
        # Assuming the format is like 'Shadow_20240515_0800_LST_With_CDSM.tif'
        date_str = file_parts[-5]  # '20240515' indicates the date
        time_str = file_parts[-4]  # '0800' indicates the time of day

        print(f"[DEBUG] Extracted date_str: {date_str}")
        print(f"[DEBUG] Extracted time_str: {time_str}")
        
        # Combine the date and time strings and parse them into a datetime object
        date_time_str = date_str + time_str
        date_time = datetime.datetime.strptime(date_time_str, '%Y%m%d%H%M')
        print(f"[DEBUG] Parsed date_time: {date_time}")
    
    except ValueError as e:
        # Handle any issues with parsing the date and time
        print(f"[ERROR] Could not parse date and time from filename: {file_name} due to {e}")
        raise ValueError(f"Could not parse date and time from filename: {file_name}")

    # Step 3: Load the raster file using GDAL for further processing
    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"[ERROR] Failed to load raster file: {file_path}")
        return date_data

    # Access the first band of the raster, which contains the shadow data
    band  = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Step 4: Handle NoData values in the raster by replacing them with NaN
    no_data_value = band.GetNoDataValue()
    if no_data_value is not None:
        array = np.where(array == no_data_value, np.nan, array)

    # Step 5: Calculate the average shading factor using np.nanmean to ignore NaN values
    shading_factor = np.nanmean(array)

    # Step 6: Update the date_data dictionary with the calculated shading factor
    # The dictionary is keyed by time (e.g., '0800'), with lists of times and shading factors as values
    time_key = date_time.strftime('%H%M')  # Format time as '0800', '1200', etc.
    if time_key not in date_data:
        date_data[time_key] = {'time': [], 'shading': []}
    date_data[time_key]['time'].append(date_time)
    date_data[time_key]['shading'].append(shading_factor)

    # Step 7: Return the updated dictionary for further use
    return date_data



def apply_mask_and_analyze_shadows(input_mask, dir_out, crs, project, tile_ind, extent=None):
    """
    Apply the mask to shadow files and analyze the resulting shadows for a specific tile.

    This function processes shadow raster files, applies a mask to them, and then analyzes 
    the resulting shadow data. The analysis is performed separately for shadow data that 
    includes a Canopy Digital Surface Model (CDSM) and for data that does not.

    Args:
        input_mask (str)   : Path to the input mask raster file.
        dir_out (str)      : Directory where the shadow files are located and where outputs will be saved.
        crs (QgsCoordinateReferenceSystem): Coordinate reference system to be used for the output.
        project (QgsProject): The QGIS project context within which the processing is done.
        tile_ind (str)     : Identifier for the tile being processed.
        extent (str, optional): The extent to apply the mask within (in "xmin, xmax, ymin, ymax" format). 
                                If None, the entire raster extent is used.

    Returns:
        tuple: Two dictionaries containing the shadow data analysis results. The first dictionary 
               contains data for the "With CDSM" case, and the second for the "Without CDSM" case.

    Workflow:
        1. Initialize dictionaries to store shadow data.
        2. Identify and separate shadow files into those with and without CDSM.
        3. Apply the mask to each shadow file and analyze the resulting shadow data.
        4. Handle any exceptions that occur during processing.
        5. Return the analyzed shadow data for further use.
    """

    # Step 1: Initialize empty dictionaries to store shadowing data, categorized by date and time
    date_data_with_cdsm = {}
    date_data_without_cdsm = {}

    # Step 2: Retrieve all files in the output directory and filter for those that are shadow files (.tif)
    shadow_files = sorted([f for f in os.listdir(dir_out) if f.endswith('.tif') and 'Shadow' in f])
    print(f"[DEBUG] Found shadow files: {shadow_files}")

    # Separate the shadow files into two groups: those that include CDSM and those that do not
    with_cdsm_files = [f for f in shadow_files if "_With_CDSM" in f]
    without_cdsm_files = [f for f in shadow_files if "_Without_CDSM" in f]

    # Step 3: Process each "With CDSM" shadow file
    for file in with_cdsm_files:
        try:
            print(f"[DEBUG] Processing shadow file: {file}")
            # Apply the mask to the shadow file and get the path to the output raster
            output_raster_path = apply_mask(input_mask, file, dir_out, extent, tile_ind, crs, project)
            
            # If the output raster was successfully created, analyze the shadow data
            if output_raster_path and output_raster_path.endswith('.tif'):
                date_data_with_cdsm = analyze_shadow(output_raster_path, file, date_data_with_cdsm)
        
        except Exception as e:
            # Handle any errors that occur during processing and log the error
            print(f"[ERROR] Failed to process shadow file {file}: {e}")

    # Step 4: Process each "Without CDSM" shadow file
    for file in without_cdsm_files:
        try:
            print(f"[DEBUG] Processing shadow file: {file}")
            # Apply the mask to the shadow file and get the path to the output raster
            output_raster_path = apply_mask(input_mask, file, dir_out, extent, tile_ind, crs, project)
            
            # If the output raster was successfully created, analyze the shadow data
            if output_raster_path and output_raster_path.endswith('.tif'):
                date_data_without_cdsm = analyze_shadow(output_raster_path, file, date_data_without_cdsm)
        
        except Exception as e:
            # Handle any errors that occur during processing and log the error
            print(f"[ERROR] Failed to process shadow file {file}: {e}")

    # Step 5: If no shadow data was successfully processed, log an error message
    if not date_data_with_cdsm and not date_data_without_cdsm:
        print(f"[ERROR] No valid shadow data was processed for tile {tile_ind}. Check input files and processing steps.")

    # Debugging output to show the final processed data for both "With CDSM" and "Without CDSM"
    print(f"[DEBUG] Final date_data_with_cdsm dictionary: {date_data_with_cdsm}")
    print(f"[DEBUG] Final date_data_without_cdsm dictionary: {date_data_without_cdsm}")

    # Return the two dictionaries containing the shadow data for further use
    return date_data_with_cdsm, date_data_without_cdsm




# Plot shading data with thermal gradient colors
# This function is only used when analysing a single tile.
# In for comparing the cases with and without CDSM, see function plot_shading_with_and_without_cdsm()
def plot_shading(date_data, tile_ind, dir_out):
    """
    Plot shading data over time with a thermal gradient color scheme.
    The plot is saved as a PDF file in the specified output directory.

    Args:
        date_data (dict): Dictionary containing shading data by time (e.g., '0800', '1200').
        tile_ind (str): Index of the tile being analyzed.
        dir_out (str): Directory to save the generated PDF file.

    Functionality:
        - The function generates a line plot of shading data for each time.
        - The x-axis represents time, and the y-axis represents the average shading factor.
        - The plot uses a colormap that transitions smoothly from cooler colors (blues) in winter
          to warmer colors (reds) in summer and back to cooler colors in autumn.
    """
    
    # Create a colormap that starts with cool colors in winter (Jan-Feb),
    # transitions to warm colors in summer (Jun-Aug), and back to cool colors in winter (Nov-Dec).
    # This uses the "twilight" colormap which naturally cycles through cool and warm tones.
    colors  = plt.get_cmap('twilight', len(date_data))

    # Initialize the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(15, 9))
    
    # Set font size for better readability
    plt.rcParams.update({'font.size': 15})

    # Plot each time's shading data with the corresponding color from the colormap
    for i, time_key in enumerate(sorted(date_data.keys())):
        # Convert times to the same day for consistent plotting
        times = [t.replace(year=2024, month=1, day=1) for t in date_data[time_key]['time']]
        shading = date_data[time_key]['shading']
        
        # Plot shading data with a thermal gradient color and markers
        ax.plot(times, shading, label=time_key, 
                color=colors(i), marker='o', linewidth=2)

    # Set the plot title and labels
    ax.set_title(f'Tile Index {tile_ind}')
    ax.set_ylabel('Average Shading Factor')
    ax.set_xlabel('Time')

    # Format the x-axis to show time in hours and minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Add a legend to distinguish different times
    ax.legend()
    
    # Enable grid lines for easier reading
    ax.grid(True)

    # Display the plot
    plt.show()

    # Save the plot as a PDF file in the specified output directory
    plt.savefig(os.path.join(dir_out, str(tile_ind) + '_shadow_analysis.pdf'), format='pdf')


def plot_shading_with_and_without_cdsm(date_data_with_cdsm, date_data_without_cdsm, 
                                       tile_ind, sub_tile_ind, dir_out):
    """
    Plot the shading data for both "With CDSM" and "Without CDSM" cases as a function of time.

    This function generates a line plot that compares the shading factors over time for two scenarios:
    one with a Canopy Digital Surface Model (CDSM) and one without it. The shading data is plotted
    for each sub-tile of a given tile.

    Args:
        date_data_with_cdsm (dict): Dictionary containing shading data for the "With CDSM" case.
        date_data_without_cdsm (dict): Dictionary containing shading data for the "Without CDSM" case.
        tile_ind (str): Identifier for the tile being analyzed.
        sub_tile_ind (str): Identifier for the sub-tile being analyzed.
        dir_out (str): Directory where the plot will be saved.

    Workflow:
        1. Extract and sort time keys from the provided data dictionaries.
        2. Extract the corresponding time and shading values for plotting.
        3. Create a line plot comparing "With CDSM" and "Without CDSM" scenarios.
        4. Save the plot to the specified directory with a unique filename.

    Example Usage:
        plot_shading_with_and_without_cdsm(date_data_with_cdsm, date_data_without_cdsm, 
                                           "2535_1153", "00", "/output/plots/")
    """

    # Step 1: Extract and sort time keys from the data dictionaries
    # This ensures that the times are plotted in chronological order
    time_keys_with_cdsm    = sorted(date_data_with_cdsm.keys())
    time_keys_without_cdsm = sorted(date_data_without_cdsm.keys())

    # Step 2: Extract the times and shading data for both scenarios
    # We assume that the first element in the 'time' and 'shading' lists corresponds to the data we need
    times_with_cdsm      = [date_data_with_cdsm[time_key]['time'][0] for time_key in time_keys_with_cdsm]
    shading_with_cdsm    = [date_data_with_cdsm[time_key]['shading'][0] for time_key in time_keys_with_cdsm]
    
    times_without_cdsm   = [date_data_without_cdsm[time_key]['time'][0] for time_key in time_keys_without_cdsm]
    shading_without_cdsm = [date_data_without_cdsm[time_key]['shading'][0] for time_key in time_keys_without_cdsm]

    # Step 3: Create a new figure for the plot
    plt.figure(figsize=(10, 6))  # Set the figure size to 10x6 inches

    # Plot the shading data for the "With CDSM" case
    # 'bo-' specifies blue color (b), circle marker (o), and solid line (-)
    plt.plot(times_with_cdsm, shading_with_cdsm, 'bo-', label='With CDSM', markersize=8, linestyle='-')

    # Plot the shading data for the "Without CDSM" case
    # 'ro-' specifies red color (r), circle marker (o), and solid line (-)
    plt.plot(times_without_cdsm, shading_without_cdsm, 'ro-', label='Without CDSM', markersize=8, linestyle='-')

    # Step 4: Add title and axis labels
    plt.title(f"Shading Analysis: Tile {tile_ind}, Sub-Tile {sub_tile_ind}")  # Add the title to the plot
    plt.xlabel("Time")  # Label for the x-axis
    plt.ylabel("Shading Factor")  # Label for the y-axis
    
    # Set y-axis limits to ensure a consistent scale across all plots
    plt.ylim(0.3, 1.0)
    
    # Format the x-axis to display time in hh:mm format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Rotate the x-axis labels to improve readability, especially if there are many time points
    plt.gcf().autofmt_xdate()

    # Add a fine grid to the plot for better visual reference
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a legend to distinguish between the "With CDSM" and "Without CDSM" data
    plt.legend()

    # Save the plot to the specified directory with a filename that includes the tile and sub-tile indices
    plot_path = os.path.join(dir_out, f"{tile_ind}_{sub_tile_ind}_shading_plot.png")
    plt.savefig(plot_path)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory


    
def run_SkyViewFactor(input_dsm, input_cdsm, dir_out, crs, project, 
                      trans_veg=None, include_cdsm=True):
    """
    Run the Sky View Factor (SVF) analysis using the Urban Multi-scale Environmental Predictor (UMEP) tool.

    The Sky View Factor (SVF) is a measure of the visible sky from a point on the ground. It's crucial for 
    understanding shading and radiative exchange in urban environments. This function calculates the SVF 
    using both a DSM (Digital Surface Model) and a CDSM (Canopy Digital Surface Model). The SVF can be 
    calculated for different vegetation transmissivity values unless a specific value is provided.

    Args:
        input_dsm (str): Path to the input DSM file, representing the bare earth's surface with buildings and other structures.
        input_cdsm (str): Path to the input CDSM file, representing the height of the canopy (trees, etc.) above the ground. 
                          Can be None if not including the CDSM in the analysis.
        dir_out (str): Directory where the output SVF files will be saved.
        crs (QgsCoordinateReferenceSystem): The coordinate reference system (CRS) to be used for the output.
        project (QgsProject): The QGIS project instance where the output layers will be added.
        trans_veg (int, optional): A specific vegetation transmissivity value to use. If None, the function 
                                   will run the SVF analysis for a set of predefined values.
        include_cdsm (bool): Flag indicating whether to include the CDSM in the SVF calculation. 
                             Defaults to True.

    Returns:
        list: A list of output file paths for the generated SVF rasters.

    Workflow:
        1. If a specific `trans_veg` value is provided, run the SVF analysis only for that value.
        2. If no `trans_veg` value is provided, run the SVF analysis for a set of predefined values.
        3. Save the resulting SVF output as a TIFF file, with filenames reflecting whether CDSM was included.
        4. Load the generated SVF raster into the QGIS project for visualization and further analysis.

    Example Usage:
        # Run SVF analysis for a specific transmissivity value with CDSM
        run_SkyViewFactor("dsm.tif", "cdsm.tif", "/output/directory", crs, project, trans_veg=15)
        
        # Run SVF analysis for default transmissivity values without CDSM
        run_SkyViewFactor("dsm.tif", None, "/output/directory", crs, project, include_cdsm=False)
    """

    # Step 1: Define the vegetation transmissivity values to be used in the SVF analysis.
    # If trans_veg is provided, use it; otherwise, use a set of predefined values.
    trans_veg_values = [trans_veg] if trans_veg else [3, 15, 25, 50]

    # Initialize a list to store the paths to the output files
    output_files = []

    # Step 2: Loop through each transmissivity value to perform the SVF calculation
    for trans_veg in trans_veg_values:
        # Determine the suffix for the output file based on whether CDSM is included
        suffix = "_With_CDSM" if include_cdsm else "_Without_CDSM"
        # Create the output file path
        output_file = os.path.join(dir_out, f'SkyViewFactor_tr{trans_veg}{suffix}.tif')

        # Step 3: Run the UMEP Sky View Factor processing tool
        proc = processing.run("umep:Urban Geometry: Sky View Factor",
                              { 'ANISO'         : True,                          # Enable anisotropic radiation calculation
                                'INPUT_CDSM'    : input_cdsm if include_cdsm else None,  # Use CDSM if included, otherwise None
                                'INPUT_DSM'     : input_dsm,                      # The DSM file is always required
                                'INPUT_TDSM'    : None,                           # Optional input TDSM (not used here)
                                'INPUT_THEIGHT' : 25,                             # Assumed tree height for calculations
                                'OUTPUT_DIR'    : dir_out,                        # Output directory
                                'OUTPUT_FILE'   : output_file,                    # Output file path
                                'TRANS_VEG'     : trans_veg })                    # Vegetation transmissivity

        # Step 4: Load the generated SVF raster into the QGIS project for visualization
        load_raster_layer(proc['OUTPUT_FILE'], crs, project)
        # Append the output file path to the list of generated files
        output_files.append(output_file)
        
        # Log the creation and loading of the SVF file
        print(f"[DEBUG] SVF file created and loaded: {output_files} ")
    
    # Return the list of output file paths for further analysis
    return output_files
        

def analyze_SkyViewFactor(output_raster_path, file, veg_data):
    """
    Analyze the Sky View Factor (SVF) from the given output raster file and update the veg_data dictionary.

    Args:
        output_raster_path (str): Path to the output raster file containing the SVF data.
        file (str): Filename of the SVF raster file.
        veg_data (dict): Dictionary to store SVF analysis results, organized by vegetation transmissivity 
                         and CDSM inclusion.

    Returns:
        dict: Updated veg_data dictionary with the average Sky View Factor for the specific 
              transmissivity and CDSM condition.
    """

    # Attempt to load the raster file using GDAL
    print(f"[DEBUG] Attempting to load raster file: {output_raster_path}")
    dataset = gdal.Open(output_raster_path)
    if dataset is None:
        print(f"[ERROR] Failed to load raster file using GDAL: {output_raster_path}")
        return None

    # Access the first band of the raster, which contains the SVF data
    print("[DEBUG] Accessing the first band of the raster.")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Handle NoData values in the raster by replacing them with NaN
    no_data_value = band.GetNoDataValue()
    print(f"[DEBUG] NoData value in raster: {no_data_value}")
    if no_data_value is not None:
        array = np.where(array == no_data_value, np.nan, array)

    # Calculate the average SVF using np.nanmean to ignore NaN values
    avg_svf = np.nanmean(array)
    print(f"[DEBUG] Calculated average SVF: {avg_svf}")

    # Extract the vegetation transmissivity value from the filename
    print(f"[DEBUG] Extracting vegetation transmissivity from filename: {file}")
    trans_veg = file.split('_')[1].replace('.tif', "").replace("tr", "")
    print(f"[DEBUG] Extracted vegetation transmissivity: {trans_veg}")

    # Determine if the file includes CDSM information
    include_cdsm = "With_CDSM" in file
    print(f"[DEBUG] CDSM inclusion status: {'With_CDSM' if include_cdsm else 'Without_CDSM'}")

    # Create a key for the veg_data dictionary that combines transmissivity and CDSM condition
    key = f"{trans_veg}_{'With_CDSM' if include_cdsm else 'Without_CDSM'}"
    print(f"[DEBUG] Generated key for veg_data: {key}")

    # Initialize the list in the veg_data dictionary if the key does not already exist
    if key not in veg_data:
        print(f"[DEBUG] Key '{key}' not found in veg_data. Initializing it.")
        veg_data[key] = []

    # Append the calculated average SVF to the list under the appropriate key
    veg_data[key].append(avg_svf)
    print(f"[DEBUG] Appended avg_svf to veg_data[{key}]. Current values: {veg_data[key]}")

    # Return the updated veg_data dictionary with the newly added SVF data
    return veg_data

def apply_mask_and_analyze_SkyViewFactor(input_mask, dir_out, crs, project, tile_ind, extent=None):
    """
    Apply the mask to Sky View Factor (SVF) files and analyze the resulting SVF data for a specific tile.

    This function processes SVF raster files, applies a mask to them, and then analyzes 
    the resulting SVF data. The analysis is performed separately for different vegetation 
    transmissivity values and for cases with and without CDSM.

    Args:
        input_mask (str)   : Path to the input mask raster file.
        dir_out (str)      : Directory where the SVF files are located and where outputs will be saved.
        crs (QgsCoordinateReferenceSystem): Coordinate reference system to be used for the output.
        project (QgsProject): The QGIS project context within which the processing is done.
        tile_ind (str)     : Identifier for the tile being processed.
        extent (str, optional): The extent to apply the mask within (in "xmin, xmax, ymin, ymax" format). 
                                If None, the entire raster extent is used.

    Returns:
        dict: A dictionary containing the SVF data analysis results, organized by vegetation transmissivity 
              and whether CDSM was included.
    
    Workflow:
        1. Initialize the veg_data dictionary to store SVF data.
        2. Identify and process SVF files based on vegetation transmissivity and CDSM inclusion.
        3. Apply the mask to each SVF file and analyze the resulting SVF data.
        4. Return the analyzed SVF data for further use.
    """

    # Initialize the veg_data dictionary to store SVF data organized by transmissivity and CDSM condition
    veg_data = {}

    # List all files in the directory that are SVF .tif files
    files = sorted([f for f in os.listdir(dir_out) if f.endswith('.tif') and 'SkyViewFactor_tr' in f])

    # Loop through each SVF file to apply the mask and analyze the SVF
    for file in files:
        try:
            print(f"[DEBUG] Processing SVF file: {file}")
            # Apply the mask to the SVF file and get the path to the output raster
            output_raster_path = apply_mask(input_mask, file, dir_out, extent, tile_ind, crs, project)
            
            # If the output raster was successfully created, analyze the SVF data
            if output_raster_path and output_raster_path.endswith('.tif'):
                veg_data = analyze_SkyViewFactor(output_raster_path, file, veg_data)
        
        except Exception as e:
            # Handle any errors that occur during processing and log the error
            print(f"[ERROR] Failed to process SVF file {file}: {e}")

    # Return the dictionary containing the SVF data for further use
    return veg_data



def plot_svf(veg_data, tile_ind, dir_out):
    # Create a plot to visualize the Sky View Factor analysis
    fig, ax = plt.subplots(figsize=(15, 9))
    plt.rcParams.update({'font.size': 15})
    
    # Plot transmissivity vs Sky View Factor
    ax.plot(veg_data['transmissivity'], veg_data['SkyViewFactor'], 
            color='g', marker='o', linewidth=2)
    
    # Set the title and labels for the plot
    ax.set_title(f'Tile Index {tile_ind}')
    ax.set_ylabel('Average Sky View Factor')
    ax.set_xlabel('Vegetation Transmissivity [%]')
    ax.legend()
    ax.grid(True)
    
    # Show the plot
    plt.show()
    
    # Save the plot as a PDF file in the output directory
    plt.savefig(os.path.join(dir_out, str(tile_ind) + '_svf_analysis.pdf'),
                format='pdf')

    
# NOTE: This function is currently not used in the main analysis pipeline.

def Get_Layer_info(rlayer):
    """
    Extracts and returns statistical information (mean, standard deviation, valid percentage)
    from a raster layer using the GDAL 'gdalinfo' tool.

    Parameters:
    rlayer (QgsRasterLayer): The raster layer to analyze.

    Returns:
    tuple: A tuple containing mean, standard deviation, and valid percentage as floats,
           or None for each value if extraction fails.
    """

    # Run the gdal:gdalinfo processing tool to get layer info
    info = processing.run("gdal:gdalinfo", 
                          {'INPUT'      : rlayer.dataProvider().dataSourceUri(), # Path to the raster layer
                           'MIN_MAX'    : True,                                  # Compute min/max values
                           'STATS'      : True,                                  # Compute statistics
                           'NOGCP'      : False,                                 # Include ground control points (GCPs)
                           'NO_METADATA': False,                                 # Include metadata
                           'EXTRA'      : '',                                    # Additional GDAL options (none)
                           'OUTPUT'     : 'TEMPORARY_OUTPUT'                     # Output is a temporary file
                           })

    # Check if the output file exists
    if not os.path.exists(info['OUTPUT']):
        print("Error: Output file does not exist.")
        return None, None, None

    # Open the temporary output file and read its content
    with open(info['OUTPUT'], 'r') as file:
        content = file.read()

    # Extract statistical values using regular expressions
    mean_match          = re.search(r'STATISTICS_MEAN=(\d+\.?\d*)', content)
    stddev_match        = re.search(r'STATISTICS_STDDEV=(\d+\.?\d*)', content)
    valid_percent_match = re.search(r'STATISTICS_VALID_PERCENT=(\d+\.?\d*)', content)

    # Convert matches to float or return None if not found
    mean          = float(mean_match.group(1)) if mean_match else None
    stddev        = float(stddev_match.group(1)) if stddev_match else None
    valid_percent = float(valid_percent_match.group(1)) if valid_percent_match else None

    # Print the extracted values (optional)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {stddev}")
    print(f"Valid Percent: {valid_percent}")

    # Return the extracted values
    return mean, stddev, valid_percent

    
def Clip_layer_by_extent(rlayer, extent, crs, project):
    """
    Clips a raster layer by the specified extent and loads the resulting clipped layer into the QGIS project.

    This function uses the GDAL `cliprasterbyextent` tool to clip the input raster layer to the 
    specified extent.
    The clipped raster is then loaded into the QGIS project.

    Args:
        rlayer (QgsRasterLayer): The input raster layer to be clipped.
        extent (str): The extent to which the raster should be clipped, specified as a string in the format
                      "xmin,xmax,ymin,ymax".
        crs (QgsCoordinateReferenceSystem): The coordinate reference system for the clipped layer.
        project (QgsProject): The QGIS project where the clipped layer will be added.

    Returns:
        QgsRasterLayer: The clipped raster layer that has been loaded into the QGIS project.

    Workflow:
        1. The input raster layer is clipped to the specified extent using the `cliprasterbyextent` GDAL tool.
        2. The output of the clipping process is a temporary raster file.
        3. The temporary clipped raster is loaded into the QGIS project as a new layer.
        4. The function returns the loaded clipped raster layer.

    Example Usage:
        clipped_layer = Clip_layer_by_extent(input_layer, "2535000,2536000,1156000,1157000", crs, project)
    """

    # Step 1: Clip the input raster layer to the specified extent using the GDAL cliprasterbyextent tool.
    clipping = processing.run("gdal:cliprasterbyextent", {
        'INPUT'     : rlayer.dataProvider().dataSourceUri(),  # The input raster layer's data source URI
        'PROJWIN'   : extent,          # The extent to clip to, in the format "xmin,xmax,ymin,ymax"
        'NODATA'    : None,            # No specific NODATA value set; could be set if needed
        'OPTIONS'   : 'COMPRESS=LZW',  # Use LZW compression to reduce file size
        'DATA_TYPE' : 5,               # Use Float32 data type to avoid potential overflow issues with Float64
        'OUTPUT'    : 'TEMPORARY_OUTPUT'  # The output is saved as a temporary file
    })

    # Step 2: Load the resulting clipped raster into the QGIS project
    clipped_layer = load_raster_layer(clipping['OUTPUT'], crs, project)
    
    # Step 3: Return the clipped raster layer
    return clipped_layer


# Building polygons are available at https://www.swisstopo.admin.ch/en/landscape-model-swissbuildings3d-2-0
# They are stored in .gdb folders (geographical database).
# we have to extract only the relevant files
def list_gdb_layers(gdb_path):
    """
    Lists all available layers in a File Geodatabase (.gdb) and returns their names.

    This function connects to a File Geodatabase (GDB) and retrieves the names of all 
    the layers (sub-layers) stored within it. These layers typically contain geographic 
    data such as building polygons, roads, and other features, which are essential for 
    spatial analysis.

    Args:
        gdb_path (str): The file path to the .gdb folder containing the Geodatabase.

    Returns:
        list: A list of strings representing the names of the layers found in the Geodatabase.
              If the Geodatabase cannot be opened or no layers are found, an empty list is returned.

    Workflow:
        1. The function constructs a URI (Uniform Resource Identifier) for the Geodatabase.
        2. It attempts to load the Geodatabase as a vector layer in QGIS.
        3. If successful, it fetches the names of all sub-layers within the Geodatabase.
        4. The function returns a list of these sub-layer names.

    Example Usage:
        layers = list_gdb_layers("/path/to/your/database.gdb")
    """

    # Step 1: Print the path of the Geodatabase being accessed
    print(f"Setting database path to: {gdb_path}")

    # Step 2: Construct a URI for the Geodatabase
    uri = QgsDataSourceUri()
    uri.setDatabase(gdb_path)
    uri_string = uri.uri()  # Get a string representation of the URI
    print(f"Constructed URI: {uri_string}")

    # Step 3: Attempt to load the Geodatabase as a vector layer in QGIS
    gdb_layer = QgsVectorLayer(uri_string, "gdb_layer", "ogr")

    # Step 4: Check if the layer is valid (i.e., if the Geodatabase was successfully opened)
    if not gdb_layer.isValid():
        print(f"Failed to open Geodatabase: {gdb_path}")
        return []  # Return an empty list if the Geodatabase couldn't be opened
    
    # Step 5: Fetch the names of all sub-layers in the Geodatabase
    print("Fetching sub-layers from the Geodatabase...")
    layers = gdb_layer.dataProvider().subLayers()

    # Step 6: Check if any sub-layers were found
    if not layers:
        print("No sub-layers found in the Geodatabase.")
        return []  # Return an empty list if no sub-layers are found
    
    # Step 7: Parse and extract the layer names from the sub-layers
    layer_names = [layer.split('!!::!!')[1] for layer in layers]
    print(f"Found {len(layers)} sub-layers in the Geodatabase: {layer_names}")

    # Step 8: Return the list of layer names
    return layer_names


def load_layer_from_gdb(gdb_path, layer_name, display_name):
    """
    Loads a specific layer from a File Geodatabase (.gdb) into the QGIS project.

    This function attempts to load a specified layer from a File Geodatabase (GDB) 
    and add it to the QGIS Layers panel. If the layer is valid and successfully 
    loaded, it is displayed with a specified name.

    Args:
        gdb_path (str): Full path to the File Geodatabase (.gdb folder).
        layer_name (str): The name of the layer to load from the Geodatabase.
        display_name (str): The name that will be used to display the layer in the QGIS Layers panel.

    Returns:
        QgsVectorLayer or bool: 
            - Returns the `QgsVectorLayer` object if the layer was loaded successfully.
            - Returns `False` if the layer could not be loaded.

    Workflow:
        1. Construct a URI (Uniform Resource Identifier) that points to the desired layer in the GDB.
        2. Attempt to load the layer using the URI and check if the layer is valid.
        3. If the layer is valid, add it to the QGIS project and display it in the Layers panel.
        4. If the layer fails to load, print an error message and return `False`.

    Example Usage:
        layer = load_layer_from_gdb("/path/to/geodatabase.gdb", "Building_solid", "Buildings")
        if layer:
            print("Layer loaded successfully.")
        else:
            print("Failed to load the layer.")
    """

    # Step 1: Construct the URI to access the specific layer within the GDB
    uri = f"{gdb_path}|layername={layer_name}"

    # Step 2: Attempt to load the specified layer using the constructed URI
    layer = QgsVectorLayer(uri, display_name, "ogr")

    # Step 3: Check if the layer was successfully loaded and is valid
    if layer.isValid():
        # Step 4: Add the valid layer to the current QGIS project
        QgsProject.instance().addMapLayer(layer)
        print(f"Layer '{display_name}' loaded successfully from {gdb_path}")
        return layer  # Return the loaded layer for further use

    else:
        # Step 5: If the layer is not valid, print an error message and return False
        print(f"Failed to load layer '{layer_name}' from {gdb_path}")
        return False


# unused functions 
'''
# Calculate masks for extracting buildings heigth and info
def Extract_building_layer(building_raster, ground_raster, output_file, crs, project, extent = None):
    
    # This formula seems to work
    # but i need another strategy to extract the average building height because this one
    # include the entire extent 
    mask_expr = "(A-B) * (((A-B) > 2) / ((A-B) > 2))"
    if extent == None:
        processing.run("gdal:rastercalculator", {
            'INPUT_A': building_raster,
            'BAND_A' : 1,
            'INPUT_B': ground_raster,
            'BAND_B' : 1,
            'FORMULA': mask_expr,
            'NO_DATA': None,
            'RTYPE'  : 5,
            'OUTPUT' : output_file
        })
    else:
        processing.run("gdal:rastercalculator", {
            'INPUT_A': building_raster,
            'BAND_A' : 1,
            'INPUT_B': ground_raster,
            'BAND_B' : 1,
            'FORMULA': mask_expr,
            'NO_DATA': None,
            'RTYPE'  : 5,
            'PROJWIN': f"{extent}[EPSG:2056]",
            'OUTPUT' : output_file
        })
    print(f"Mask created at {output_file}")
    building_layer = load_raster_layer(output_file, crs, project)
    
    return building_layer
    

def calculate_building_density(building_layers, extent):
    """
    Calculate the building density within a given extent.

    :param building_layers: List of QgsVectorLayer objects representing building polygons.
    :param extent         : QgsRectangle defining the selected extent.
    :return               : Building density (total building area / total extent area).

    Spatial Filtering:

        QgsFeatureRequest().setFilterRect(extent): This line creates a spatial filter using the extent. 
            The filter ensures that only features (building polygons) that are within the extent are processed. 
            This is done to improve performance by avoiding unnecessary processing of features outside 
            the area of interest.

    Processing Each Building Polygon:

        building_geom = feature.geometry(): Retrieves the geometry of the current feature (building polygon).
            The geometry contains information about the shape and area of the building.
        building_geom.intersects(QgsGeometry.fromRect(extent)): This checks if the building polygon 
            intersects with the extent. This is important because some buildings might only partially 
        overlap with the extent, and we still want to include those in our calculations.
        building_area = building_geom.area(): If the building intersects the extent, 
            the area of the building polygon is calculated.

    """
    # 
    # Initialize the total building area to 0
    total_building_area_within_extent     = 0
    # 
    # Calculate the area of the specified extent 
    # (the rectangular area in which we want to calculate density)
    total_extent_area      = extent.area()
    # 
    # Counters to track buildings fully and partially within the extent
    fully_within_count     = 0
    partially_within_count = 0
    # 
    # Check if the extent area is zero, which would prevent a division by zero error later on
    if total_extent_area == 0:
        return 0
    # 
    # Create a spatial filter for the extent using QgsFeatureRequest.
    # This filter will ensure that only features (buildings) 
    # within the specified extent are considered.
    request = QgsFeatureRequest().setFilterRect(extent)
    # 
    # Iterate over each building layer in the building_layers list
    for layer in building_layers:
        # Get features from the current layer that fall within the spatial filter (extent)
        for feature in layer.getFeatures(request):
            # Get the geometry of the current feature (building polygon)
            building_geom = feature.geometry()
            # 
            # Check if the building geometry intersects with the extent
            # This ensures that only buildings partially or fully inside 
            # the extent are considered
            # Check if the building is fully within the extent
            if building_geom.within(QgsGeometry.fromRect(extent)):
                fully_within_count += 1
                building_area = building_geom.area()
                total_building_area_within_extent += building_area
            #         
            # Check if the building is partially within the extent
            elif building_geom.intersects(QgsGeometry.fromRect(extent)):
                partially_within_count += 1
                
                # Calculate the intersection area (building area within the extent)
                intersection = building_geom.intersection(QgsGeometry.fromRect(extent))
                intersection_area = intersection.area()
                
                total_building_area_within_extent += intersection_area
    # 
    # Calculate the building density as the ratio of 
    # total building area to the total extent area
    building_density = total_building_area_within_extent / total_extent_area
    # 
    # Print the counts for fully and partially within buildings
    print(f"Buildings fully within the extent: {fully_within_count}")
    print(f"Buildings partially within the extent: {partially_within_count}")
    #   
    # Return the calculated building density
    return building_density 

def nearest_point_in_extent(geometry, extent):
    """
    Find the nearest point on the geometry to the center of the extent that lies within the extent.
    """
    if not geometry or geometry.isEmpty():
        return None
    # 
    extent_geom = QgsGeometry.fromRect(extent)
    centroid = extent_geom.centroid().asPoint()
    
    # Use the nearest point on geometry to the extent's centroid that is within the extent
    nearest_point = geometry.nearestPoint(QgsGeometry.fromPointXY(centroid)).asPoint()
    if extent_geom.contains(QgsGeometry.fromPointXY(nearest_point)):
        return nearest_point
    
    # If the nearest point is outside the extent, revert to the extent's centroid itself
    return centroid if extent_geom.contains(QgsGeometry.fromPointXY(centroid)) else None

    

def calculate_average_building_height(building_layers, extent):
    """
    Calculate the average building height within a specified extent across multiple layers.
    This function processes multiple QgsVectorLayers containing building data, specifically looking
    for buildings within the given geographical extent, and computes their average height based on
    a specified attribute.

    Parameters:
    - building_layers (list of QgsVectorLayer): The layers containing building data with height fields.
    - extent (QgsRectangle): The geographical extent within which to calculate the average height.

    Returns:
    - float: The average height of buildings within the extent. Returns 0 if no buildings are found.
    """
    height_field = 'DACH_MAX'  # Field name where building height is stored
    total_height = 0
    building_count = 0
    
    # Iterate over each layer in the list to process buildings
    for layer in building_layers:
        # Create a feature request to filter features by the geographical extent
        request = QgsFeatureRequest().setFilterRect(extent)
        
        # Retrieve features within the extent from the current layer
        for feature in layer.getFeatures(request):
            # Check if the feature's geometry intersects the specified extent
            if feature.geometry().intersects(extent):
                height = feature[height_field]
                if height is not None:  # Ensure the height attribute is not null
                    total_height += height
                    building_count += 1
                    # Print the feature ID and its height for tracking and verification
                    print(f"Feature ID: {feature.id()}, Height (DACH_MAX): {height}")
    
    # Calculate the average height if there are any buildings counted
    if building_count > 0:
        average_height = total_height / building_count
        print(f"Processed {building_count} buildings. Average Height: {average_height}")
        return average_height
    else:
        print("No buildings found within the specified extent.")
        return 0  # Return 0 or you might choose to return None to indicate no data

def calculate_average_relative_building_height(building_layers, ground_raster, extent):
    height_field = 'DACH_MAX'
    total_relative_height = 0
    count = 0
    provider = ground_raster.dataProvider()
    extent_geom = QgsGeometry.fromRect(extent)
    # 
    for layer in building_layers:
        request = QgsFeatureRequest().setFilterRect(extent)
        for feature in layer.getFeatures(request):
            geom = feature.geometry()
            if geom.intersects(extent_geom):
                point = geom.centroid().asPoint() if geom.within(extent_geom) else next(
                    (QgsPointXY(v) for v in geom.vertices() if extent_geom.contains(QgsGeometry.fromPointXY(QgsPointXY(v)))), None)
                
                if point:
                    value, success = provider.sample(point, 1)
                    if success:
                        building_height = feature[height_field]
                        ground_point = feature['GELAENDEPUNKT']  # If this is relevant
                        relative_height = building_height - value
                        print(f"Building ID: {feature.id()}, Building Height: {building_height}, Ground Elevation: {value}, Ground Point: {ground_point}, Relative Height: {relative_height}")
                    else:
                        print(f"Sampling failed at building ID: {feature.id()}")
                else:
                    print(f"No valid point found within the extent for building ID: {feature.id()}")
    # 
    average_relative_height = total_relative_height / count if count > 0 else None
    return average_relative_height

def sample_raster_values(raster_layer, num_samples=10):
    provider = raster_layer.dataProvider()
    extent   = raster_layer.extent()
    width    = provider.xSize()
    height   = provider.ySize()
    x_step   = extent.width() / num_samples
    y_step   = extent.height() / num_samples
    
    print(f"Raster dimensions: {width}x{height}")
    print(f"Sampling every {x_step} in X and {y_step} in Y within the extent {extent.toString()}")
    
    sample_values = []
    for i in range(num_samples):
        for j in range(num_samples):
            x = extent.xMinimum() + i * x_step
            y = extent.yMinimum() + j * y_step
            point = QgsPointXY(x, y)
            result = raster_layer.dataProvider().identify(point, QgsRaster.IdentifyFormatValue).results()
            value = result[1]  # Band 1 value
            sample_values.append(value)
            print(f"Value at ({x}, {y}): {value}")
    
    return sample_values

def calculate_average_building_height_gdal(raster_layer):
    # Extract the file path from the raster layer's data provider
    data_provider = raster_layer.dataProvider()
    raster_path = data_provider.dataSourceUri().split('|')[0]  # Gets the path before any '|' character
    print(f"Raster Path: {raster_path}")
    # 
    # Open the raster file using GDAL
    dataset = gdal.Open(raster_path)
    if not dataset:
        print("Failed to open raster file.")
        return None
    # 
    # Access the first band of the raster
    band = dataset.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    print(f"No Data Value: {no_data_value}")
    #  
    # Read the entire band into an array
    array = band.ReadAsArray()
    if array is None:
        print("Failed to read raster data as array.")
        return None
    # 
    # Apply a mask to filter out No Data values and ground level (zero values)
    mask = (array != no_data_value) & (array > 0)
    if not np.any(mask):  # Check if there's any data other than no_data_value and ground level
        print("No valid building height data found.")
        return None
    # 
    # Extract valid data using the mask
    valid_data = array[mask]
    print(f"Sample Valid Data: {valid_data[:10]}")  # Print the first 10 valid data points for debugging
    # 
    # Calculate the average height from the valid data points
    average_height = np.mean(valid_data)
    print(f"Calculated Average Building Height: {average_height}")
    return average_height

def calculate_building_area_gdal(raster_layer):
    """
    Calculate the total surface area occupied by buildings using a raster layer where buildings are represented by positive height values.

    Parameters:
        raster_layer (QgsRasterLayer): The raster layer containing building heights.

    Returns:
        float: Total surface area occupied by buildings in square meters.

    Explanation:
        - The function first retrieves the path to the raster file from the raster layer.
        - It uses GDAL's gdalinfo command to fetch metadata, particularly the NoData value.
        - It calculates the area each pixel represents by using the raster's spatial resolution.
        - It counts the number of pixels with positive values (excluding NoData pixels), assuming these represent buildings.
        - It calculates the total area occupied by these buildings by multiplying the count of building pixels by the area per pixel.
    """

    # Fetch the raster path from the layer data provider
    raster_path = raster_layer.dataProvider().dataSourceUri().split('|')[0]

    # Use GDAL to get metadata including NoData value
    command = f'gdalinfo -mm -stats {raster_path}'
    gdal_info_output = subprocess.check_output(command, shell=True).decode()
    no_data_value = float(re.search(r"NoData Value=(\S+)", gdal_info_output).group(1))

    # Calculate the area represented by each pixel
    extent = raster_layer.extent()
    width = raster_layer.width()
    height = raster_layer.height()
    pixel_area = (extent.width() / width) * (extent.height() / height)
     
    # Open the raster using GDAL and read the first band
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)  # Building heights are assumed to be in the first band
    array = band.ReadAsArray()
     
    # Count pixels where building height is greater than 0 and not equal to NoData
    building_pixels = np.count_nonzero((array > 0) & (array != no_data_value))
    
    # Compute the total building area
    total_building_area = building_pixels * pixel_area
    
    # Output for debugging or verification
    print(f"Total building area: {total_building_area} square meters")
    
    return total_building_area
'''

def calculate_building_heights_from_dsm_dtm(dsm_raster, dtm_raster, extent, crs, project):
    """
    Calculate building heights by subtracting the Digital Terrain Model (DTM) 
    from the Digital Surface Model (DSM).

    This function computes the relative height of buildings by subtracting the ground level (DTM) from the 
    surface level (DSM). The resulting raster represents the height of objects above the ground, typically 
    buildings. This operation is performed within a specified extent, and the result is loaded into 
    the QGIS project.

    Args:
        dsm_raster (QgsRasterLayer): The DSM raster layer, representing surface elevations 
        including buildings and vegetation.
        dtm_raster (QgsRasterLayer): The DTM raster layer, representing the bare ground surface 
        without any buildings or vegetation.
        extent (str): The spatial extent within which to perform the calculation. 
        The format is "xmin, xmax, ymin, ymax".
        crs (QgsCoordinateReferenceSystem): The coordinate reference system to be used for the output layer.
        project (QgsProject): The QGIS project instance where the resulting layer will be added.

    Returns:
        QgsRasterLayer: The resulting raster layer containing building heights.
    
    Workflow:
        1. Define a no-data value to be used for the output raster.
        2. Set up the parameters for the GDAL Raster Calculator tool to subtract the DTM from the DSM.
        3. Run the raster calculator to generate the building height raster.
        4. Load the resulting raster into the QGIS project and return the raster layer.

    Example Usage:
        building_heights_layer = calculate_building_heights_from_dsm_dtm(dsm_layer, dtm_layer, 
        "2535000,2540000,1150000,1155000", crs, project)
    """

    # Step 1: Define a no-data value for the output raster. 
    # This value will be assigned to pixels where data is not available.
    no_data_value = -9999

    # Step 2: Set up parameters for the GDAL Raster Calculator tool.
    # The formula 'A - B' indicates that we are subtracting the DTM (B) from the DSM (A) 
    # to get building heights.
    params = {
        'INPUT_A': dsm_raster.dataProvider().dataSourceUri(),  # URI of the DSM raster
        'BAND_A' : 1,  # Use the first band of the DSM raster
        'INPUT_B': dtm_raster.dataProvider().dataSourceUri(),  # URI of the DTM raster
        'BAND_B' : 1,  # Use the first band of the DTM raster
        'FORMULA': 'A - B',  # Subtraction formula for calculating building heights
        'NO_DATA': no_data_value,  # Assign a no-data value to the output raster
        'RTYPE'  : 5,  # Output data type: Float32 (GDAL type 5)
        'EXTENT' : extent,  # Spatial extent to apply the calculation, format: "xmin, xmax, ymin, ymax"
        'OUTPUT' : 'TEMPORARY_OUTPUT'  # Output is saved as a temporary file, to be loaded into QGIS
    }

    # Step 3: Execute the GDAL Raster Calculator with the specified parameters 
    # to produce the building height raster.
    result = processing.run("gdal:rastercalculator", params)

    # Step 4: Load the resulting raster layer into the QGIS project.
    raster_layer = load_raster_layer(result['OUTPUT'], crs, project)

    # Step 5: Return the loaded raster layer.
    return raster_layer

def calculate_feature_stats_qgis(raster_layer):
    """
    Calculate the total area, density, standard deviation, and average height of features (buildings or vegetation)
    using a QgsRasterLayer where these features are represented by positive values.

    Parameters:
        raster_layer (QgsRasterLayer): The raster layer containing feature heights.

    Returns:
        tuple: Total area, density, standard deviation, and average height of features.

    Explanation:
        - The function reads the raster data into a numpy array.
        - It uses the raster's spatial resolution to calculate the area that each pixel represents.
        - It identifies and counts the number of pixels that represent the features (non-zero and not NoData values).
        - It computes the total area by multiplying the number of feature pixels by the area each pixel represents.
        - The density is calculated as the ratio of feature area to the total raster area.
        - Standard deviation and average height of the features are also calculated from the pixel values.
    """

    # Fetch the raster path from the layer data provider and open the dataset
    raster_path = raster_layer.dataProvider().dataSourceUri().split('|')[0]
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)  # Assuming feature heights are in the first band
    array = band.ReadAsArray()

    # Retrieve NoData value and calculate the pixel area
    no_data_value = band.GetNoDataValue()
    extent = raster_layer.extent()
    width, height = raster_layer.width(), raster_layer.height()
    pixel_area = (extent.width() / width) * (extent.height() / height)

    # Filter array to ignore NoData and non-feature (zero) values
    feature_mask = (array != no_data_value) & (array > 0)
    feature_pixels = array[feature_mask]
    total_feature_area = feature_pixels.size * pixel_area

    # Calculate the total area of the raster for density calculation
    total_area = width * height * pixel_area
    density = (total_feature_area / total_area) * 100  # percentage

    # Calculate standard deviation and average height
    if feature_pixels.size > 0:
        std_deviation = np.std(feature_pixels)
        average_height = np.mean(feature_pixels)
    else:
        std_deviation = 0
        average_height = 0

    # Output for debugging or verification
    print(f"Total feature area: {total_feature_area} square meters, Density: {density}%, "
          f"Standard Deviation: {std_deviation}, Average Height: {average_height}")

    return total_feature_area, density, std_deviation, average_height


# UNTESTED FUNCTION: very computationally heavy and ran out of time for this project 
def calculate_aspect_ratio_of_urban_canyons(dsm_layer, dtm_layer, building_layer):
    """
    Calculate the aspect ratio (Height-to-Width ratio) of urban canyons in a given area.

    Parameters:
        dsm_layer (QgsRasterLayer): Digital Surface Model (DSM) raster layer containing building heights.
        dtm_layer (QgsRasterLayer): Digital Terrain Model (DTM) raster layer containing ground elevation.
        building_layer (QgsVectorLayer): Vector layer containing building footprints.

    Returns:
        float: The average aspect ratio of urban canyons in the area.
    """
    aspect_ratios = []
    raster_extent = dsm_layer.extent()

    for feature in building_layer.getFeatures():
        building_geom = feature.geometry()
        building_centroid = building_geom.centroid().asPoint()

        # Find the nearest neighboring building to calculate the width of the canyon
        nearest_building = find_nearest_building(building_geom, building_layer)

        if nearest_building is not None:
            nearest_geom = nearest_building.geometry()
            nearest_centroid = nearest_geom.centroid().asPoint()

            # Create a line representing the canyon (from centroid to centroid)
            canyon_line = QgsGeometry.fromPolylineXY([building_centroid, nearest_centroid])

            # Check if the canyon line intersects the raster extent
            if canyon_line.intersects(raster_extent):
                # Calculate the width of the canyon (distance between centroids)
                width = building_centroid.distance(nearest_centroid)

                # Calculate the height of buildings from DSM and DTM
                building_height = get_building_height(dsm_layer, dtm_layer, building_geom)
                nearest_building_height = get_building_height(dsm_layer, dtm_layer, nearest_geom)

                if building_height is not None and nearest_building_height is not None and width > 0:
                    average_height = (building_height + nearest_building_height) / 2
                    aspect_ratio = average_height / width
                    aspect_ratios.append(aspect_ratio)

    if aspect_ratios:
        return np.mean(aspect_ratios)
    else:
        print("No valid urban canyons found.")
        return None

# UNTESTED FUNCTION: very computationally heavy and ran out of time for this project 
def find_nearest_building(building_geom, building_layer):
    """
    Find the nearest neighboring building to the given building.

    Parameters:
        building_geom (QgsGeometry): Geometry of the current building.
        building_layer (QgsVectorLayer): Vector layer containing building footprints.

    Returns:
        QgsFeature: The nearest neighboring building feature, or None if not found.
    """
    min_distance = float('inf')
    nearest_building = None
    
    for feature in building_layer.getFeatures():
        if feature.geometry() != building_geom:
            distance = building_geom.distance(feature.geometry())
            if distance < min_distance:
                min_distance = distance
                nearest_building = feature
    
    return nearest_building

# UNTESTED FUNCTION: ran out of time for this project 
def get_building_height(dsm_layer, dtm_layer, building_geom):
    """
    Calculate the height of a building using DSM and DTM data.

    Parameters:
        dsm_layer (QgsRasterLayer): Digital Surface Model (DSM) raster layer.
        dtm_layer (QgsRasterLayer): Digital Terrain Model (DTM) raster layer.
        building_geom (QgsGeometry): Geometry of the building.

    Returns:
        float: The height of the building, or None if the centroid is outside the raster extent.
    """
    centroid = building_geom.centroid().asPoint()
    dsm_value = sample_raster(dsm_layer, centroid)
    dtm_value = sample_raster(dtm_layer, centroid)
    
    if dsm_value is not None and dtm_value is not None:
        return dsm_value - dtm_value
    else:
        return None
    
# UNTESTED FUNCTION: ran out of time for this project 
def sample_raster(raster_layer, point):
    """
    Sample a raster layer at a specific point.

    Parameters:
        raster_layer (QgsRasterLayer): The raster layer to sample.
        point (QgsPointXY): The point at which to sample the raster.

    Returns:
        float: The raster value at the specified point, or None if the point is outside the raster extent.
    """
    return raster_layer.dataProvider().identify(point, QgsRaster.IdentifyFormatValue).results().get(1, None)

# UNUSED FUNCTION 
def get_building_layers_in_raster_extent(building_layers, raster_layer):
    """
    Identify which building layers intersect with the extent of a given raster layer.

    Parameters:
        building_layers (list): A list of QgsVectorLayer objects representing building layers.
        raster_layer (QgsRasterLayer): The raster layer (DSM or DTM) whose extent will be used.

    Returns:
        list: A list of QgsVectorLayer objects that intersect with the raster extent.
    """
    # Get the extent of the raster layer
    raster_extent = raster_layer.extent()
    
    # Initialize an empty list to store the layers that intersect with the raster extent
    layers_in_extent = []

    # Loop through each building layer and check for intersection with the raster extent
    for layer in building_layers:
        if layer.extent().intersects(raster_extent):
            layers_in_extent.append(layer)

    return layers_in_extent


def plot_tile_summary(df, tile_ind, dir_out):
    '''
    Generate summary plots for a given tile after analyising all its sub-tiles.
    
    Args:
        df (pd.DataFrame): The data frame containing analysis results for the tile
        tile_ind (str): Identifier for the tile
        dir_out (str): directory to save the plots
    '''

    # filter the data frame for the specific tile
    tile_df = df[df['Tile Index'] == tile_ind]  

    # List of columns to plot histograms for
    columns_to_plot = {
        'Tree Density'                     : 'Tree Density (%)',
        'Tree Height Avg'                  : 'Average Tree Height (m)',
        'Building Density'                 : 'Building Density (%)',
        'Building Height Avg'              : 'Average Building Height (m)',
        'Shadow Factor 8am (With CDSM)'    : 'Shading Factor at 08:00 (With CDSM)',
        'Shadow Factor 8am (Without CDSM)' : 'Shading Factor at 08:00 (Without CDSM)',
        'Shadow Factor 12pm (With CDSM)'   : 'Shading Factor at 12:00 (With CDSM)',
        'Shadow Factor 12pm (Without CDSM)': 'Shading Factor at 12:00 (Without CDSM)',
        'Shadow Factor 4pm (With CDSM)'    : 'Shading Factor at 16:00 (With CDSM)',
        'Shadow Factor 4pm (Without CDSM)' : 'Shading Factor at 16:00 (Without CDSM)',
        'Sky View Factor (With CDSM)'      : 'Sky view factor (with CDSM)',
        'Sky View Factor (Without CDSM)'   : 'Sky view factor (without CDSM)'
    } 

    # Create histograms for each metric
    for col, label in columns_to_plot.items():
        plt.figure(figsize=(10, 6))
        plt.hist(tile_df[col].dropna(), bins=10, color='skyblue', edgecolor='black')
        plt.title(f"{label} Distribution for Tile {tile_ind}")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save the plot
        plot_path = os.path.join(dir_out, f"{tile_ind}_{col.replace(' ', '_')}_histogram.png")
        plt.savefig(plot_path)
        plt.close()


def plot_combined_summary(df, dir_out):
    """
    Generate summary plots for all tiles combined.

    Args:
        df (pd.DataFrame): The DataFrame containing analysis results for all tiles.
        dir_out (str): Directory to save the plots.
    """

    # List of columns to plot histograms for
    columns_to_plot = {
        'Tree Density'                      : 'Tree Density (%)',
        'Tree Height Avg'                   : 'Average Tree Height (m)',
        'Building Density'                  : 'Building Density (%)',
        'Building Height Avg'               : 'Average Building Height (m)',
        'Shadow Factor 8am (With CDSM)'     : 'Shading Factor at 08:00 (With CDSM)',
        'Shadow Factor 8am (Without CDSM)'  : 'Shading Factor at 08:00 (Without CDSM)',
        'Shadow Factor 12pm (With CDSM)'    : 'Shading Factor at 12:00 (With CDSM)',
        'Shadow Factor 12pm (Without CDSM)' : 'Shading Factor at 12:00 (Without CDSM)',
        'Shadow Factor 4pm (With CDSM)'     : 'Shading Factor at 16:00 (With CDSM)',
        'Shadow Factor 4pm (Without CDSM)'  : 'Shading Factor at 16:00 (Without CDSM)'
    }

    # Create histograms for each metric
    for col, label in columns_to_plot.items():
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title(f"{label} Distribution for All Tiles")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save the plot
        plot_path = os.path.join(dir_out, f"all_tiles_{col.replace(' ', '_')}_histogram.png")
        plt.savefig(plot_path)
        plt.close()

    
def Analyse_Multiple_tiles(path_to_rasters, path_to_polygons, 
                           path_to_analysis_results):
    """
    Analyze multiple tiles, dividing each into sub-tiles, and calculate various urban metrics.

    This function performs an in-depth analysis of multiple geographical tiles by dividing each tile into smaller
    sub-tiles. It calculates several urban metrics, including building density, tree density, shadow factors,
    and sky view factors (SVF), both with and without considering the Canopy Digital Surface Model (CDSM) in order
    to quantify the effect of urban vegetation.

    Workflow:
        1. Initialize a DataFrame to store results from all tiles.
        2. Set up the Coordinate Reference System (CRS) and QGIS project.
        3. Load building polygons for later use in building-related calculations.
        4. Loop through each tile in the list and perform the following for each:
            - Load or process DSM and CDSM raster layers.
            - Run shadow generator simulations (with and without CDSM).
            - Run Sky View Factor (SVF) calculations (with and without CDSM).
            - Loop through each sub-tile, apply masks, and analyze shadow factors and SVF.
            - Clip layers to the sub-tile extent and calculate feature statistics 
              (e.g., building height, tree density).
            - Store the results in the DataFrame.
            - Generate summary plots for each tile.
        5. After processing all tiles, save the DataFrame to a CSV file.

    Returns:
        DataFrame: A DataFrame containing all the calculated metrics for each sub-tile in each tile.

    example:
    df = Analyse_Multiple_tiles('/home/lmartinell/uv/data/GeoData/Lausanne/Rasters/', 
                                '/home/lmartinell/uv/data/GeoData/Lausanne/Polygons',
                                '/home/lmartinell/uv/data/GeoData/Lausanne/Parameter_Analysis/')
    """

    # Step 1: Initialize the DataFrame to store the results of the analysis.
    columns = [
        'Tile Index', 
        'Sub-Tile Index', 
        'Building Density', 
        'Building Height Avg', 
        'Building Height Stdev', 
        'Tree Density',     
        'Tree Height Avg',     
        'Tree Height Stdev',
        'Shadow Factor 8am (With CDSM)', 
        'Shadow Factor 12pm (With CDSM)', 
        'Shadow Factor 4pm (With CDSM)',
        'Sky View Factor (With CDSM)',
        'Shadow Factor 8am (Without CDSM)', 
        'Shadow Factor 12pm (Without CDSM)', 
        'Shadow Factor 4pm (Without CDSM)',
        'Sky View Factor (Without CDSM)'
    ]
    df = pd.DataFrame(columns=columns)

    # Step 2: Set the Coordinate Reference System (CRS) and QGIS project context.
    crs, project = set_crs()

    # Step 3: Specify the directory containing the raster data and get the list of tiles.
    raster_dir   = path_to_rasters
    tile_list    = os.listdir(raster_dir)
    tile_list.sort()

    # Step 4: Load building polygon files for use in building-related calculations 
    # (e.g., urban canyon analysis - to be implemented).
    building_layers = [] 
    poly_folder     = path_to_polygons
    for gdb in os.listdir(poly_folder):
        gdb_path    = os.path.join(poly_folder, gdb)
        bdl         = load_layer_from_gdb(gdb_path, 'Building_solid', 'Building Solid')
        if bdl.isValid():
            building_layers.append(bdl)
            print("Building Solid layer added to the project.")
        else:
            print("Could not add Building Solid layer.")

    # Step 5: Choose a date for which to run the analysis on all tiles (in this case, mid-May 2024).
    startyear, startmonth, startday = 2024, 5, 15
    # hardcoded: assign a value for the vegetation light transmissivity, matching that used in the 
    # run_shadow_generator function 
    trans_veg_day = 3
    date = datetime.date(startyear, startmonth, startday) 

    # Step 6: Loop over each tile in the tile list to perform the analysis.
    for tile_ind in tile_list: 
    # for tile_ind in tile_list[30:]: # to test loop on a single tile 
        
        print(f"Tile index {tile_ind}:")
        
        # Step 6.1: Look for CDSM and DSM raster layers.
        input_cdsm   = os.path.join(raster_dir, tile_ind, tile_ind + '_mrg_cdsm.tif')
        input_dsm    = os.path.join(raster_dir, tile_ind, tile_ind + '_mrg_dsm.tif')
        input_grd    = os.path.join(raster_dir, tile_ind, tile_ind + '_grd.tif')
        grd_layer    = load_raster_layer(input_grd, crs, project)
        small_dsm    = load_raster_layer(os.path.join(raster_dir, tile_ind, tile_ind + '_dsm.tif'), crs, project)
        
        # Step 6.2: Merge adjacent tiles to create DSM and CDSM raster layers that extend 100m in all directions.
        if (not os.path.isfile(input_cdsm)) or (not os.path.isfile(input_dsm)):
            cdsm_layer, dsm_layer = process_tiles(tile_ind, raster_dir, crs, project)
        else: 
            # Load DSM and CDSM layers if they already exist.
            cdsm_layer = load_raster_layer(input_cdsm, crs, project)
            
            dsm_layer  = load_raster_layer(input_dsm,  crs, project)
            # for visualisation purposes only 
            apply_custom_pseudocolor_renderer(cdsm_layer)
            # Get the extent of the raster layer
            extent = grd_layer.extent()

            # Format the extent as a string
            extent_str = f"{extent.xMinimum()},{extent.xMaximum()},{extent.yMinimum()},{extent.yMaximum()}"
            building_height_lyr     = calculate_building_heights_from_dsm_dtm(small_dsm, 
                                                                              grd_layer, 
                                                                              extent_str, 
                                                                              crs, project)
            apply_custom_pseudocolor_renderer(building_height_lyr, color_ramp="red")

        # Step 6.3: Create an output directory for the results of this tile.
        dir_init     = path_to_analysis_results
        dir_out      = os.path.join(dir_init, tile_ind) + '/'
        create_directory(dir_out)

        # Step 6.4: Run the shadow generator simulations (with and without CDSM) at different times of the day.
        shadow_files = {
            'With CDSM'   : run_shadow_generator(date, input_cdsm, input_dsm, dir_out, startyear, 
                                                 itertime=240, include_cdsm=True),
            'Without CDSM': run_shadow_generator(date, None,       input_dsm, dir_out, startyear, 
                                                 itertime=240, include_cdsm=False)
        }

        # Step 6.5: Run Sky View Factor (SVF) calculations for the entire tile (with and without CDSM).
        # Uncomment here if you want to run the SVF calculation, keeping in mind that is very time consuming 
        '''
        print("Running Sky View Factor calculation (with CDSM), can be very computationally heavy...")
        svf_files_with_cdsm    = run_SkyViewFactor(input_dsm, input_cdsm, dir_out, crs, project, 
                                                   trans_veg=trans_veg_day, include_cdsm=True)
        print("Running Sky View Factor calculation (without CDSM), can also be very computationally heavy...")
        svf_files_without_cdsm = run_SkyViewFactor(input_dsm, None,       dir_out, crs, project, 
                                                   trans_veg=trans_veg_day, include_cdsm=False)
        '''
        # Step 6.6: Generate masks if they do not already exist.
        mask_dir           = os.path.join(raster_dir, tile_ind)
        input_no_buildings = os.path.join(mask_dir, f"{tile_ind}_grd_nob.tif")
        lat_lon            = mask_dir.split('/')[-1].split('_')
        lat                = int(lat_lon[0])
        lon                = int(lat_lon[1])
        
        # Step 6.7: Loop over each sub-tile (200x200 m²) within the current tile.
        for i in range(0, 5):
            for j in range(0, 5):
        # uncomment next lines to test loop on a single sub-tile
        # for i in range(0, 2):
        #     for j in range(0, 2):

                sub_tile_ind = f"{i}{j}"
                extent_mask  = (f"{int(lat) * 1000 + (i) * 200},{int(lat) * 1000 + (i + 1) * 200},"
                                f"{int(lon) * 1000 + (j) * 200},{int(lon) * 1000 + (j + 1) * 200}")
                input_mask   = os.path.join(mask_dir, f"{tile_ind}_mask_{i}{j}.tif")
                if not os.path.exists(input_mask):
                    print("Mask does not exist. Creating mask...")
                    calculate_mask(input_no_buildings, input_mask, extent=extent_mask)
                else:
                    print("Mask already exists.")
                rlayer_mask = load_raster_layer(input_mask, crs, project)

                # Step 6.8: Calculate shadow factors and SVF for both with and without CDSM.
                shadow_factors = {}
                svf_factors    = {}

                # Apply mask and analyze shadowing.
                date_data_with_cdsm, date_data_without_cdsm = apply_mask_and_analyze_shadows(
                    input_mask, dir_out, crs, project, tile_ind, extent=extent_mask) 
                if '0800' in date_data_with_cdsm:
                    shadow_factors['With CDSM'] = {
                        '0800': np.mean(date_data_with_cdsm['0800']['shading']),
                        '1200': np.mean(date_data_with_cdsm['1200']['shading']),
                        '1600': np.mean(date_data_with_cdsm['1600']['shading'])
                    }
                    shadow_factors['Without CDSM'] = {
                        '0800': np.mean(date_data_without_cdsm['0800']['shading']),
                        '1200': np.mean(date_data_without_cdsm['1200']['shading']),
                        '1600': np.mean(date_data_without_cdsm['1600']['shading'])
                    }
                else:
                    print(f"[ERROR] '0800' key missing in date_data: {date_data_with_cdsm.keys()}")

                # Plot the shading data for the current condition.
                plot_shading_with_and_without_cdsm(date_data_with_cdsm, date_data_without_cdsm, 
                                                    tile_ind, sub_tile_ind, dir_out)
                    
                # Step 6.9: Apply mask and analyze Sky View Factor for both with and without CDSM.
                # uncomment here if you had run the SVF calculation 
                '''
                svf_data_with_cdsm = apply_mask_and_analyze_SkyViewFactor(
                    input_mask, dir_out, crs, project, tile_ind, extent=extent_mask)
                svf_data_without_cdsm = apply_mask_and_analyze_SkyViewFactor(
                    input_mask, dir_out, crs, project, tile_ind, extent=extent_mask)
                '''
                # Step 6.10: Calculate average tree density by clipping layers to the sub-tile extent.
                clipped_cdsm = Clip_layer_by_extent(cdsm_layer, extent_mask, crs, project)
                clipped_dsm  = Clip_layer_by_extent(dsm_layer,  extent_mask, crs, project)
                clipped_dtm  = Clip_layer_by_extent(grd_layer,  extent_mask, crs, project)
                # Step 6.11: Split the extent string to get the individual coordinates and create a QgsRectangle.
                xmin, xmax, ymin, ymax = map(float, extent_mask.split(','))
                extent                 = QgsRectangle(xmin, ymin, xmax, ymax) 
                # Step 6.12: Calculate the relative height of buildings from the ground level using the DSM and DTM.
                building_height_lyr    = calculate_building_heights_from_dsm_dtm(clipped_dsm, 
                                                                                 clipped_dtm, 
                                                                                 extent, 
                                                                                 crs, project)
                # Step 6.13: Calculate feature statistics (e.g., building density, tree density).
                building_area_tot, building_density, building_height_stdev, building_height_avg = \
                calculate_feature_stats_qgis(building_height_lyr)
                tree_area_tot, tree_density, tree_height_stdev, tree_height_avg = \
                calculate_feature_stats_qgis(clipped_cdsm)
                # Step 6.14: Store the calculated results for the current sub-tile in the DataFrame.
                df = df.append({
                    'Tile Index'                       : tile_ind,
                    'Sub-Tile Index'                   : sub_tile_ind,
                    'Building Density'                 : building_density,
                    'Building Height Avg'              : building_height_avg,
                    'Building Height Stdev'            : building_height_stdev,
                    'Tree Density'                     : tree_density,
                    'Tree Height Avg'                  : tree_height_avg,
                    'Tree Height Stdev'                : tree_height_stdev,
                    'Shadow Factor 8am (With CDSM)'    : shadow_factors['With CDSM']['0800'],
                    'Shadow Factor 12pm (With CDSM)'   : shadow_factors['With CDSM']['1200'],
                    'Shadow Factor 4pm (With CDSM)'    : shadow_factors['With CDSM']['1600'],
                    'Sky View Factor (With CDSM)'      : None, # np.mean(svf_data_with_cdsm['3_With_CDSM']), # HARDCODED: replace 3 with the used trans_veg for the svf calculation 
                    'Shadow Factor 8am (Without CDSM)' : shadow_factors['Without CDSM']['0800'],
                    'Shadow Factor 12pm (Without CDSM)': shadow_factors['Without CDSM']['1200'],
                    'Shadow Factor 4pm (Without CDSM)' : shadow_factors['Without CDSM']['1600'],
                    'Sky View Factor (Without CDSM)'   : None, # np.mean(svf_data_without_cdsm['3_Without_CDSM']) # HARDCODED: replace 3 with the used trans_veg for the svf calculation
                }, ignore_index=True)

                time.sleep(5)

                # Step 6.15: Print the last few entries of the DataFrame for inspection.
                print("Last added features to the DataFrame:")
                print(df.tail())

        # Step 7: After processing all sub-tiles for the current tile, generate summary plots.
        plot_tile_summary(df, tile_ind, dir_out)

        # Save the DataFrame to a CSV file at the end of each tile analysis
        # so in case QGIS crashes, we have some data at least 
        output_csv_path = os.path.join(dir_init, 'analysis_results.csv')
        df.to_csv(output_csv_path, index=False)

    # After the loop over all tiles is done
    plot_combined_summary(df, dir_init)
                
    return df

# Main processing workflow for analysing a single tile for several days of the year
# CAVEAT:
# some function declaration might need to be revisited after modification of:
#  
# run_shadow_generator
# apply_mask_and_analyze_shadows
# plot_shading
# run_SkyViewFactor 
# apply_mask_and_analyze_SkyViewFactor
# 
# Check function definition before calling main() 
def main():
    # Set CRS and project (defined by SwissTopo data)
    crs, project = set_crs()

    # Directory containing raster data
    raster_dir   = '/home/lmartinell/uv/data/GeoData/Lausanne/Rasters/'
    tile_list    = os.listdir(raster_dir)
    # Choose the tile index you want to analyse 
    tile_ind     = '2536_1157'
    if not tile_ind in tile_list:
        print("Not existing tile index")
        return None

    # Look for CDSM and DSM raster layers
    input_cdsm   = os.path.join(raster_dir, tile_ind, tile_ind + '_mrg_cdsm.tif')
    input_dsm    = os.path.join(raster_dir, tile_ind, tile_ind + '_mrg_dsm.tif')

    if (not os.path.isfile(input_cdsm)) or (not os.path.isfile(input_dsm)):
        # merge the adjacent tiles to create the dsm and cdsm raster that extend 100m
        # in all directions with respect to the original tile size
        process_tiles(tile_ind, raster_dir, crs, project)
    else: 
        # load dsm and cdsm 
        load_raster_layer(input_cdsm, crs, project)
        load_raster_layer(input_dsm,  crs, project)
    
    # Directory setup for results
    startyear, startmonth, startday = 2024, 1, 1
    dir_init     = '/home/lmartinell/uv/data/GeoData/Lausanne/UMEP_results/'
    dir_out      = os.path.join(dir_init, tile_ind) + '/'
    create_directory(dir_out)

    # Generate mask if it not already existing
    mask_dir   = os.path.join(raster_dir, tile_ind)
    input_mask = os.path.join(mask_dir, f"{tile_ind}_mask.tif")
    if not os.path.exists(input_mask):
        print("Mask does not exist. Creating mask...")
        input_no_buildings = os.path.join(mask_dir, f"{tile_ind}_grd_nob.tif")
        # Mask's size is that of the original tile
        calculate_mask(input_no_buildings, input_mask)
    else:
        print("Mask already exists.")

    # Load the mask layer
    rlayer_mask  = load_raster_layer(input_mask, crs, project)

    # Run the shadow generator for the following dates
    for i in range(0,12):
        # as example, I am running the shadow generator every 30 days during the year 2024. 
        # Alternatively, one can build a list of selected days over which running the algorithm. 
        date = datetime.date(startyear, startmonth, startday) + datetime.timedelta(days=30*i)
        run_shadow_generator(date, input_cdsm, input_dsm, dir_out, startyear)

    # Apply mask excluding the shadow over building surfaces 
    # and calculate average shading over the original tile
    date_data = apply_mask_and_analyze_shadows(input_mask, dir_out, crs, project, tile_ind)

    # save output file
    np.save(os.path.join(dir_out, tile_ind + '_DailyShadowing'), date_data) 
    # Plot average shading over time with thermal gradient colors
    plot_shading(date_data, tile_ind, dir_out)

    # Calculate and analyse sky view factor
    run_SkyViewFactor(input_dsm, input_cdsm, dir_out, crs, project)
    svf_data = apply_mask_and_analyze_SkyViewFactor(input_mask, dir_out, crs, project, tile_ind)
    # save output file
    np.save(os.path.join(dir_out, tile_ind + '_SkyViewFactor'), svf_data) 
    plot_svf(svf_data, tile_ind, dir_out) 


if __name__ == "__main__":
    main()