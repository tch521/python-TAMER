#!/usr/bin/env python3
import os

def CloudCompare_RasterizeLidarData(
     cc_dir   = "/home/lmartinell/projets/CloudComPy_Conda310_Linux64_20240420/CloudComPy310/lib/cloudcompare/",
     data_dir = "/home/lmartinell/uv/data/GeoData/Lausanne/"):

    '''
    Create raster files from LiDAR point cloud files.
    LiDAR files for Switzerland can be downloaded from 
    https://www.swisstopo.admin.ch/en/height-model-swisssurface3d
    and should be stored in the data_dir folder.

    Outputs: digital surface models for ground+trees, ground+building and ground only.
    Additionally, a ground+trees surface is produced leaving holes instead of interpolating
    the building surface in order to have indication on the "walkable surface", 
    for averagin purposes in later steps analysis.
    The .tif (geotiff) output files are the input of UMEP's shadow generator calculation.
    '''

    os.environ["_CCTRACE_"]="ON" # only if you want C++ debug traces

    # move to folder containing CloudComPy libraries and import modules
    os.chdir(cc_dir)
    import cloudComPy as cc

    # folder containing LiDAR data of Lausanne
    geodata_dir = data_dir
    # get file names in the folder 
    names       = os.listdir(geodata_dir)
    # sort to alphabetic order
    names.sort()

    # iterate over .las files in the geodata folder
    for name in names:
        if name.endswith(".las"):
            cloud       = cc.loadPointCloud(geodata_dir+name)
            cloud_name  = cloud.getName()
            print("cloud name: %s" % cloud_name)
            # extract name of the lidar file - remove .las extension from string
            cloud_coord = cloud_name[0:-4]
            
            # create folder where to save the output files 
            if not os.path.exists(geodata_dir + "Rasters/"):
                # if not already existing, create a Rasters folder 
                os.mkdir(geodata_dir + "Rasters/")
            # create a folder in Rasters for the given point cloud tile 
            new_folder = geodata_dir + "Rasters/" + cloud_coord + "/"
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            
            # check if rasters were already created
            # sometimes the computation is too much and the routine crashes mid way
            if (os.path.isfile(new_folder + cloud_coord + "_Ground_Buildings_RASTER_Z.tif") and 
                os.path.isfile(new_folder + cloud_coord + "_Ground_Trees_RASTER_Z.tif") and
                os.path.isfile(new_folder + cloud_coord + "_Ground_RASTER_Z.tif") and
                os.path.isfile(new_folder + cloud_coord + "_Ground_Trees_WalkableSurface_RASTER_Z.tif")):
                print("%s file already fully rasterised, continuing..." %cloud_name)
                continue
                
            # get info about the .las file
            '''
            npts        = cloud.size()
            print("number of points: %i" % npts)
            res         = cloud.hasScalarFields()
            print("hasScalarField: %s" % res)
            
            sf = cloud.getScalarField(0)
            '''

            #### ---- Access scalar fields ---- ####
            
            # number of scalar fields
            nsf = cloud.getNumberOfScalarFields()
            if cloud.hasScalarFields():
                # create scalar field dictionary
                dic = cloud.getScalarFieldDic()
            
            # Clean the point cloud using the Statistical Outliers Removal (SOR) filter 
            print("Applying SOR filter to %s point cloud" %cloud_name)
            refCloud        = cc.CloudSamplingTools.sorFilter(cloud)
            (sorCloud, res) = cloud.partialClone(refCloud)
            sorCloud.setName("sorCloud")
            if res != 0:
                raise RuntimeError
            else:
                print("SOR filter completed")
                
            # Access to point cloud classification from the cleaned point cloud
            # Info on the classification can be found at https://www.swisstopo.admin.ch/en/height-model-swisssurface3d-raster 

            # Filter by classification     
            sorCloud.setCurrentOutScalarField(dic["Classification"])
            # ground class is #2
            print("selecting ground points...")
            fcloud_ground    = cc.filterBySFValue(1.5, 2.5, sorCloud)
            # trees class  is #3
            print("selecting vegetation points...")
            fcloud_trees     = cc.filterBySFValue(2.5, 3.5, sorCloud)
            # buildings class is #6
            print("selecting buildings points...")
            fcloud_buildings = cc.filterBySFValue(5.5, 6.5, sorCloud)
            
            # merge the filtered clouds into ground+trees, ground+buildings and ground only
            # in preparation to UMEP requirements
            cloud_ground_trees     = cc.MergeEntities([fcloud_ground, fcloud_trees])
            cloud_ground_buildings = cc.MergeEntities([fcloud_ground, fcloud_buildings])
            
            # save point clouds (necessary? --- deprecated
            # perhaps for comparison between automatic and manual version)
            ''' 
            ret = cc.SavePointCloud(fcloud_ground,          geodata_dir +
                                    cloud_coord + "_ground.las")
            ret = cc.SavePointCloud(cloud_ground_trees,     geodata_dir + 
                                    cloud_coord + "_ground_trees.las")
            ret = cc.SavePointCloud(cloud_ground_buildings, geodata_dir + 
                                    cloud_coord + "_ground_buildings.las")
            '''
            # rasterize filling the empty cells using a "Kriging" interpolation method 
            # (computationally heavier but allows to interpolate cells all over 
            # the raster grid and not only inside the non-empty cells convex hull)
            print("Rasterising ground + trees point cloud - interpolation with Kriging method...")
            cloud_ground_trees.setName(cloud_coord + "_Ground_Trees")     # output name   
            cc.RasterizeGeoTiffOnly(cloud_ground_trees,
                                    gridStep                = 0.5,                            # SwissTopo resolution in [m]
                                    outputRasterZ           = True,                           # Rasterise points height 
                                    pathToImages            = new_folder,                     # output folder
                                    emptyCellFillStrategy   = cc.EmptyCellFillOption.KRIGING, # interpolation method 
                                    export_perCellAvgHeight = True)
            
            # Repeat for grounds and buildings and for ground alone 
            print("Rasterising ground + buildings point cloud - interpolation with Kriging method...")
            cloud_ground_buildings.setName(cloud_coord + "_Ground_Buildings")
            cc.RasterizeGeoTiffOnly(cloud_ground_buildings,
                                    gridStep                = 0.5, # SwissTopo resolution [m]
                                    outputRasterZ           = True,
                                    pathToImages            = new_folder, # output folder
                                    emptyCellFillStrategy   = cc.EmptyCellFillOption.KRIGING,
                                    export_perCellAvgHeight = True)
            print("Rasterising ground point cloud - interpolation with Kriging method...")
            fcloud_ground.setName(cloud_coord + "_Ground")
            cc.RasterizeGeoTiffOnly(fcloud_ground,
                                    gridStep                = 0.5, # SwissTopo resolution [m]
                                    outputRasterZ           = True,
                                    pathToImages            = new_folder, # output folder
                                    emptyCellFillStrategy   = cc.EmptyCellFillOption.KRIGING,
                                    export_perCellAvgHeight = True)
            # Last raster file will be useful during the analysis
            # as the UV exposure should be averaged over the ground and trees data, where
            # people can actually stay (i.e. people don't usually stand on roofs)
            print("Rasterising ground point cloud - No interpolation, filling building surfaces with 0")
            fcloud_ground.setName(cloud_coord + "_Ground_Trees_WalkableSurface")
            cc.RasterizeGeoTiffOnly(fcloud_ground,
                                    gridStep                = 0.5, # SwissTopo resolution [m]
                                    outputRasterZ           = True,
                                    pathToImages            = new_folder, # output folder
                                    emptyCellFillStrategy   = cc.EmptyCellFillOption.FILL_CUSTOM_HEIGHT,
                                    customHeight            = 0.,
                                    export_perCellAvgHeight = True)
            





# CloudComPy module folder
cc_folder   = "/home/lmartinell/projets/CloudComPy_Conda310_Linux64_20240420/CloudComPy310/lib/cloudcompare/"
# Folder containing LiDAR data 
data_folder = "/home/lmartinell/uv/data/GeoData/Lausanne/"

CloudCompare_RasterizeLidarData(
     cc_dir   = cc_folder,
     data_dir = data_folder  
)