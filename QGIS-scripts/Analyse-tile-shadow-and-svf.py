import datetime
import os
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import processing
from   matplotlib import pyplot as plt
from   qgis.core import QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem
from   qgis.PyQt.QtCore import QDate, QTime

# Define the coordinate reference system (CRS) and create a QGIS project
def set_crs(epsg_code="EPSG:2056"):
    """
    Set the coordinate reference system (CRS) for the QGIS project.
    https://epsg.io/2056 Swiss Coordinate Reference System used for the SwissTopo data

    Args:
        epsg_code (str): EPSG code for the CRS.
    Returns:
        crs (QgsCoordinateReferenceSystem): The CRS object.
        project (QgsProject): The QGIS project instance.
    """
    crs     = QgsCoordinateReferenceSystem(epsg_code)
    project = QgsProject.instance()
    project.setCrs(crs)
    return crs, project

# Load a raster layer
def load_raster_layer(file_path, crs, project):
    """
    Load a raster layer into the QGIS project.

    Args:
        file_path (str): Path to the raster file.
        crs (QgsCoordinateReferenceSystem): The CRS object.
        project (QgsProject): The QGIS project instance.

    Returns:
        rlayer (QgsRasterLayer): The loaded raster layer.
    """
    layer_name = os.path.basename(file_path).replace(".tif", "")
    rlayer     = QgsRasterLayer(file_path, layer_name)
    if not rlayer.isValid():
        print(f"Failed to load layer: {layer_name}")
    else:
        project.addMapLayer(rlayer)
        rlayer.setCrs(crs)
        print(f"Layer loaded: {rlayer.name()}")
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

def make_cdsm(raster_dir, tile_index, crs, project):
    '''
    Build a Canopy Digital Surface Model (CDSM) for the selected tile.

    The CDSM is created by subtracting the DEM (Digital Elevation Model)
    from the DSM (Digital Surface Model). The resulting CDSM describes the
    relative height of vegetation from the ground, as required by UMEP.

    Args:
        raster_dir (str): Directory containing raster files.
        tile_index (str): Index of the tile to process.
        crs (QgsCoordinateReferenceSystem): Coordinate reference system.
        project (QgsProject): QGIS project instance.

    Returns:
        str: Path to the generated CDSM raster, or None if the operation fails.
    '''
    vegetation_file = os.path.join(raster_dir, tile_index, tile_index+'_grd_trs.tif')
    if os.path.isfile(vegetation_file):
        load_raster_layer(vegetation_file, crs, project)
    else:
        print('No DSM file with vegetation for tile %s' %tile_index)
        return None
    
    ground_file = os.path.join(raster_dir, tile_index, tile_index+'_grd.tif')
    if os.path.isfile(ground_file):
        load_raster_layer(ground_file, crs, project)
    else:
        print('No DEM file for tile %s' %tile_index)
        return None
    
    output_cdsm = os.path.join(raster_dir, tile_index, f"{tile_index}_cdsm.tif")
    proc        = processing.run("gdal:rastercalculator", 
                            {'INPUT_A'   : vegetation_file,
                            'BAND_A'     : 1,
                            'INPUT_B'    : ground_file,
                            'BAND_B'     : 1,
                            'FORMULA'    : 'A-B',
                            'NO_DATA'    : None,
                            'EXTENT_OPT' : 0,
                            'PROJWIN'    : None,
                            'RTYPE'      : 5,
                            'OPTIONS'    : '',
                            'EXTRA'      : '',
                            'OUTPUT'     : output_cdsm})
     
    load_raster_layer(proc['OUTPUT'], crs, project)
    return proc['OUTPUT']

def merge_and_clip_rasters(tile_list, lat, lon, output_path, crs, project):
    """
    Merges and clips adjacent raster tiles to form a 3x3 grid centered on the given tile.
    The resulting raster extends 100m beyond the edges of the central tile to account for 
    shadow effects from nearby buildings and trees.
    
    Args:
        tile_list (list)                  : List of tile paths to merge.
        lat (int)                         : Latitude of the central tile.
        lon (int)                         : Longitude of the central tile.
        output_path (str)                 : Path to save the merged and clipped raster.
        crs (QgsCoordinateReferenceSystem): Coordinate reference system.
        project (QgsProject)              : QGIS project instance.

    Returns:
        str: The path to the merged and clipped raster, or None if the operation fails.
    """
    if not tile_list:
        print(f"No tiles to merge for {output_path}.")
        return None
    
    # Merge the tiles using GDAL merge
    print('merging: ' )
    print(*tile_list, sep = "\n")
    merge_res = processing.run("gdal:merge",
                               {'INPUT'        : tile_list,
                                'PCT'          : False,
                                'SEPARATE'     : False,
                                'NODATA_INPUT' : None,
                                'NODATA_OUTPUT': None,
                                'OPTIONS'      : '',
                                'EXTRA'        : '', 
                                'DATA_TYPE'    : 1,
                                'OUTPUT'       : 'TEMPORARY_OUTPUT'})
    
    if 'OUTPUT' not in merge_res:
        print(f"Merge failed for {output_path}.")
        return None
    
    # Load the merged tiles to check if the merging process worked 
    load_raster_layer(merge_res['OUTPUT'], crs, project)

    # Clip the merged raster around the central tile
    print('clipping the merging output: %s' % merge_res['OUTPUT'])
    clip_extent = (f"{lat * 1000 - 100},{lat * 1000 + 1100},"
                   f"{lon * 1000 - 100},{lon * 1000 + 1100}"
                    "[EPSG:2056]")
    clip_res = processing.run("gdal:cliprasterbyextent", 
                              {'INPUT'    : merge_res['OUTPUT'],
                               'PROJWIN'  : clip_extent,
                               'OVERCRS'  : False,
                               'NODATA'   : None,
                               'OPTIONS'  : '',
                               'DATA_TYPE': 0,
                               'EXTRA'    : '',
                               'OUTPUT'   : output_path})
    
    if 'OUTPUT' not in clip_res:
        print(f"Clip failed for {output_path}.")
        return None
    
    # Load the merged and clipped tile into the QGIS project
    load_raster_layer(clip_res['OUTPUT'] , crs, project)
    
    return output_path

def process_tiles(tile_index, raster_dir, crs, project):
    """
    Process and merge adjacent tiles based on the given tile index.
    If the merged and clipped tile does not exist, it creates it.

    Args:
        tile_index (str): The tile index in the format 'lat_lon'.
        raster_dir (str): The directory containing the raster tiles.
        crs (QgsCoordinateReferenceSystem): Coordinate reference system.
        project (QgsProject): QGIS project instance.

    Returns:
        tuple: Paths to the merged and clipped DSM and CDSM tiles.
    """

    # Define paths for merged and clipped DSM and CDSM tiles
    merged_clipped_dsm  = os.path.join(raster_dir, f"{tile_index}", f"{tile_index}_mrg_dsm.tif")
    merged_clipped_cdsm = os.path.join(raster_dir, f"{tile_index}", f"{tile_index}_mrg_cdsm.tif")

    # Check if the merged and clipped tiles already exist
    if os.path.exists(merged_clipped_dsm) and os.path.exists(merged_clipped_cdsm):
        print(f"{merged_clipped_dsm} and {merged_clipped_cdsm} already exists.")
        print('Loading them to the project...')
        load_raster_layer(dsm_tile_path,  crs, project)
        load_raster_layer(cdsm_tile_path, crs, project)
        return merged_clipped_dsm, merged_clipped_cdsm

    # Extract latitude and longitude from the tile index
    lat, lon       = int(tile_index[:4]), int(tile_index[5:9])
    dsm_tile_list  = []
    cdsm_tile_list = [] 

    # Find and collect adjacent tiles
    for lat_offset in range(-1, 2):
        for lon_offset in range(-1, 2):
            neighbor_lat       = lat + lat_offset
            neighbor_lon       = lon + lon_offset
            neighbor_name_dsm  = f"{neighbor_lat:04d}_{neighbor_lon:04d}{'_dsm.tif'}"
            neighbor_name_cdsm = f"{neighbor_lat:04d}_{neighbor_lon:04d}{'_cdsm.tif'}"
            neighbor_dir       = os.path.join(raster_dir,f"{neighbor_lat:04d}_{neighbor_lon:04d}{'/'}")

            # Verify existence of neighboring layers
            if os.path.exists(neighbor_dir):
                dsm_tile_path = os.path.join(neighbor_dir, neighbor_name_dsm)
                if os.path.isfile(dsm_tile_path):
                    load_raster_layer(dsm_tile_path, crs, project)
                    dsm_tile_list.append(dsm_tile_path)
                    print("Neighbour DSM tile:  %s" % dsm_tile_path)

                # repeat for cdsm 
                cdsm_tile_path = os.path.join(neighbor_dir, neighbor_name_cdsm)
                if os.path.isfile(cdsm_tile_path):
                    load_raster_layer(cdsm_tile_path, crs, project)
                    cdsm_tile_list.append(cdsm_tile_path)
                    print("Neighbour CDSM tile:  %s" % cdsm_tile_path)
                else:
                    # try build a cdsm file
                    new_cdsm = make_cdsm(neighbor_dir, str(neighbor_lat + '_' + neighbor_lon), crs, project)
                    if new_cdsm is not None:
                        load_raster_layer(new_cdsm, crs, project)
                        cdsm_tile_list.append(new_cdsm)
                        print("Neighbour cdsm tile:  %s" % new_cdsm)

    if len(dsm_tile_path) < 9 or len(cdsm_tile_path) < 9:
        Warning("3x3 composite tile incomplete")

    # Process DSM tiles
    dsm_tile_list.sort()
    merged_clipped_dsm  = merge_and_clip_rasters(dsm_tile_list, lat, lon, merged_clipped_dsm, crs, project)

    # Process CDSM tiles
    cdsm_tile_list.sort()
    merged_clipped_cdsm = merge_and_clip_rasters(cdsm_tile_list, lat, lon, merged_clipped_cdsm, crs, project)
    
    return merged_clipped_dsm, merged_clipped_cdsm

# Run shadow generator
def run_shadow_generator(date, input_cdsm, input_dsm, output_dir, startyear):
    """
    Run UMEP's shadow generator to calculate shadows for a given date.
    The vegetation light transmission varies during the year with a step function
    to simulate the thicker canopy in late spring/summer and the absence of leafs in 
    late autumn and winter.
    Requires UMEP and UMEP for processing plugins to be installed.

    Save the calculated shadows in .tif files in the chosen output directory.
    Shadow files take the name "Shadow_YYYYmmdd_HHMM_LST.tif"

    Args:
        date (datetime.date): Date for shadow calculation.
        input_cdsm (str):     Path to the canopy DSM.
        input_dsm (str):      Path to the DSM.
        output_dir (str):     Directory to save the output shadow files.
        startyear (int):      The start year for calculations.
    """
    date_str  = date.strftime("%d-%m-%Y")
    datetorun = QDate.fromString(date_str, "d-M-yyyy")

    # values are arbitrary: they can depend on the latitude, kind of vegetation...
    if    QDate(startyear, 4, 15) < datetorun < QDate(startyear, 9, 15):
        # late spring and summer vegetation canopy 
        transVeg = 3
    elif (QDate(startyear, 3, 15) < datetorun < QDate(startyear, 4, 15) or
          QDate(startyear, 9, 15) < datetorun < QDate(startyear, 10, 15)):
        # early spring and early autumn vegetation canopy 
        transVeg = 15
    elif (QDate(startyear, 3, 1) < datetorun < QDate(startyear, 3, 15) or
          QDate(startyear, 10, 15) < datetorun < QDate(startyear, 11, 15)):
        # late winter and mid autumn vegetation canopy 
        transVeg = 25
    else:
        # late autumn and winter vegetation canopy 
        transVeg = 50

    # time zone used in the UV climatology 
    utc = 0

    params_in = {
        'DATEINI'      : datetorun,
        'DST'          : False,
        'INPUT_ASPECT' : None,
        'INPUT_CDSM'   : input_cdsm,
        'INPUT_DSM'    : input_dsm,
        'INPUT_HEIGHT' : None,
        'INPUT_TDSM'   : None,
        'INPUT_THEIGHT': 25,
        'ITERTIME'     : 30,
        'ONE_SHADOW'   : False,
        'OUTPUT_DIR'   : output_dir,
        'TIMEINI'      : QTime(8, 0, 0),
        'TRANS_VEG'    : transVeg,
        'UTC'          : utc
    }

    print(f"Running shadow generator for date: {date_str}")
    processing.run("umep:Solar Radiation: Shadow Generator", params_in)

# Calculate masks
def calculate_mask(input_raster, output_raster):

    """
    Calculate a mask for the given input raster, excluding the building area 

    Args:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to the output mask file.
    """
    rlayer = QgsRasterLayer(input_raster, "input_raster")
    if not rlayer.isValid():
        print("Failed to load input raster!")
        return

    # replace the height of the raster with 1s outside the building area 
    mask_expr = "(A > 0) / (A > 0)"
    processing.run("gdal:rastercalculator", {
        'INPUT_A': input_raster,
        'BAND_A' : 1,
        'FORMULA': mask_expr,
        'NO_DATA': None,
        'RTYPE'  : 5,
        'OUTPUT' : output_raster
    })
    print(f"Mask created at {output_raster}")

# Apply mask
def apply_mask(input_mask, shadow_file, dir_out, extent, tile_ind, crs, project):
    '''
    A mask is necessary to exclude the building surface from the calculation
    of average shadowing or average sky view factor of the analysed tile.
    Since the mask built with the calculate_mask function has only ones 
    where the surface is not a building and None otherwise, we can simply 
    make a multiplication between the mask and raster file we want to analyse.
    '''
    mask_expr = 'A * B'
    
    proc = processing.run("gdal:rastercalculator", {
        'INPUT_A'   : input_mask,
        'BAND_A'    : 1,
        'INPUT_B'   : os.path.join(dir_out, shadow_file),
        'BAND_B'    : 1,
        'FORMULA'   : mask_expr,
        'NO_DATA'   : None,
        'EXTENT_OPT': 0,
        'PROJWIN'   : f"{extent}[EPSG:2056]",
        'RTYPE'     : 5,
        'OUTPUT'    : 'TEMPORARY_OUTPUT'
    })

    layer = load_raster_layer(proc['OUTPUT'], crs, project)
    layer.setName(tile_ind + '_' + shadow_file.replace("_LST.tif", ""))

    return proc['OUTPUT']  

def analyze_shadow(output_raster_path, file, date_data):
    '''
    Calculate average shadowing factor (how much light reaches the ground)
    using the masked shadow files
    '''
    # use xarray to analyze the .tif file
    ds = xr.open_dataset(output_raster_path)
    # Exclude nan pixels from the average calculation 
    avg_shading = np.nanmean(ds.Band1)
    
    # extract date and time info from the shadow file name 
    date_str, time_str = file.split('_')[1], file[-12:-8]
    date_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
    
    if date_obj.date() not in date_data:
        date_data[date_obj.date()] = {'time': [], 'shading': []}
    
    # arrange data to know the average shadowing effect
    # as function of time of the day and time of the year
    date_data[date_obj.date()]['time'].append(date_obj)
    date_data[date_obj.date()]['shading'].append(avg_shading)
    
    return date_data

def apply_mask_and_analyze_shadows(input_mask, dir_out, crs, project, tile_ind):
    '''
    Apply the previous two functions for all shadow files generated for the selected tile
    '''
    date_data = {}
    files = sorted([f for f in os.listdir(dir_out) if (f.endswith('.tif') and 'Shadow' in f)])
    
    for file in files:
        lat, lon = dir_out[-10:-6], dir_out[-5:-1]
        extent = (f"{int(lat) * 1000},{int(lat) * 1000 + 1000},"
                  f"{int(lon) * 1000},{int(lon) * 1000 + 1000}")

        output_raster_path = apply_mask(input_mask, file, dir_out, extent, tile_ind, crs, project)
        date_data          = analyze_shadow(output_raster_path, file, date_data)
    
    return date_data

# Plot shading data with thermal gradient colors
def plot_shading(date_data, tile_ind, dir_out):
    """
    Plot shading data over time.
    Save figure in pdf format.

    Args:
        date_data (dict): Dictionary containing shading data by date.
        tile_ind (str): Tile index.
        dir_out (str) : where to save the pdf
    """
    colors  = plt.get_cmap('coolwarm', len(date_data))
    fig, ax = plt.subplots(figsize=(15,9))
    plt.rcParams.update({'font.size': 15})

    for i, date in enumerate(sorted(date_data.keys())):
        times   = [t.replace(year=2024, month=1, day=1) for t in date_data[date]['time']]
        shading = date_data[date]['shading']
        ax.plot(times, shading, label=date.strftime('%Y-%m-%d'), 
                color=colors(i), marker='o', linewidth=2)

    ax.set_title(f'Tile Index {tile_ind}')
    ax.set_ylabel('Average Shading Factor')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend()
    ax.grid(True)
    plt.show()
    plt.savefig(os.path.join(dir_out, str(tile_ind) + '_shadow_analysis.pdf'),
                format = 'pdf')
    
def run_SkyViewFactor(input_dsm, input_cdsm, dir_out, crs, project):
    # replicate step values from the run_shadow_generator function
    # option: pass trans_veg_values as input argument for run_SkyViewFactor and run_shadow_generator
    trans_veg_values = [3, 15, 25, 50] 
    for trans_veg in trans_veg_values:
        proc = processing.run("umep:Urban Geometry: Sky View Factor",
                        { 'ANISO'         : True,
                          'INPUT_CDSM'    : input_cdsm,
                          'INPUT_DSM'     : input_dsm,
                          'INPUT_TDSM'    : None, 
                          'INPUT_THEIGHT' : 25,
                          'OUTPUT_DIR'    : dir_out,
                          'OUTPUT_FILE'   : os.path.join(dir_out, 'SkyViewFactor_tr' + str(trans_veg)+ '.tif'),
                          'TRANS_VEG'     : trans_veg })
        # load output in the project
        load_raster_layer(proc['OUTPUT_FILE'], crs, project)

def analyze_SkyViewFactor(output_raster_path, file, veg_data):
    # use xarray to analyze the .tif file
    ds      = xr.open_dataset(output_raster_path)
    # calculate average sky view factor 
    avg_svf = np.nanmean(ds.Band1)
    
    # extract date and time info from the shadow file name 
    trans_veg = file.split('_')[1].replace('.tif', "")
    trans_veg = trans_veg.replace("tr", "")

    veg_data['transmissivity'].append(int(trans_veg))
    veg_data['SkyViewFactor'].append(avg_svf)
    
    return veg_data


def apply_mask_and_analyze_SkyViewFactor(input_mask, dir_out, crs, project, tile_ind):
    veg_data = {}
    veg_data['transmissivity'] = [] 
    veg_data['SkyViewFactor']  = [] 
    files    = sorted([f for f in os.listdir(dir_out) if (f.endswith('.tif') and 'SkyViewFactor_tr' in f)])
    
    for file in files:
        lat, lon = dir_out[-10:-6], dir_out[-5:-1]
        extent   = (f"{int(lat) * 1000},{int(lat) * 1000 + 1000},"
                    f"{int(lon) * 1000},{int(lon) * 1000 + 1000}")

        output_raster_path = apply_mask(input_mask, file, dir_out, extent, tile_ind, crs, project)
        veg_data           = analyze_SkyViewFactor(output_raster_path, file, veg_data)
    
    return veg_data

def plot_svf(veg_data, tile_ind, dir_out):

    fig, ax = plt.subplots(figsize=(15,9))
    plt.rcParams.update({'font.size': 15})

    ax.plot(veg_data['transmissivity'] , veg_data['SkyViewFactor'], 
            color='g', marker='o', linewidth=2)

    ax.set_title(f'Tile Index {tile_ind}')
    ax.set_ylabel('Average Sky View Factor')
    ax.set_xlabel('Vegetation Transmissivity [%]')
    ax.legend()
    ax.grid(True)
    plt.show()
    plt.savefig(os.path.join(dir_out, str(tile_ind) + '_svf_analysis.pdf'),
                format = 'pdf')

# Main processing workflow
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

    # Plot average shading over time with thermal gradient colors
    plot_shading(date_data, tile_ind, dir_out)

    # Calculate and analyse sky view factor
    run_SkyViewFactor(input_dsm, input_cdsm, dir_out, crs, project)
    veg_data = apply_mask_and_analyze_SkyViewFactor(input_mask, dir_out, crs, project, tile_ind)
    plot_svf(veg_data, tile_ind, dir_out) 

if __name__ == "__main__":
    main()