import pandas as pd
import os
import numpy as np
import seaborn as sns
from   matplotlib import pyplot as plt

path  = '/home/lmartinell/uv/data/GeoData/Lausanne/Parameter_Analysis/'
file1 = 'analysis_results.csv'
file2 = 'analysis_results_2.csv'
file3 = 'analysis_results_3.csv'
file4 = 'analysis_results_4.csv'

df1 = pd.read_csv(os.path.join(path, file1))
df2 = pd.read_csv(os.path.join(path, file2))
df3 = pd.read_csv(os.path.join(path, file3))


# df  = pd.concat([df1, df2], ignore_index=True, axis=0)
df  = pd.concat([df1, df2, df3], ignore_index=True, axis=0)
# delete all rows containing a Nan after having removed the sky view factor
# columns because these are only Nan 
df  = df.drop(['Sky View Factor (With CDSM)', 'Sky View Factor (Without CDSM)'], 
              axis = 1)
df  = df.dropna() 
# filter rows for which the calculation of building density and tree density did not work 
df  = df[(df['Building Density'] > 0) &  (df['Tree Density'] >0)]
df.head()

when = '4pm'
# Define the bin width (0.05) and range of bins
bin_width = 0.05
bins = np.arange(0, 1 + bin_width, bin_width)

#####################################################
#             SHADOW FACTOR HISTOGRAM               # 
#####################################################
plt.figure(figsize=(10, 6))
sns.histplot(df[f'Shadow Factor {when} (With CDSM)'], 
             color = 'green', 
             label = 'With CDSM', 
             kde   = False, 
             bins  = bins, 
             alpha=0.6)
sns.histplot(df[f'Shadow Factor {when} (Without CDSM)'], 
             color = 'orange', 
             label =' Without CDSM', 
             kde   = False, 
             bins  = bins, 
             alpha=0.6)

# Add labels and title
plt.xlabel(f'Shadow Factor at {when}', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.title(f'Distribution of Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.legend()
plt.savefig(os.path.join(path, f'ShadowDistribution_{when}.png'))
# Show the plot
plt.show()


#####################################################
#             TREE AND BUILDING DENSITY             #
#####################################################
bins = np.arange(0, 100 + 5, 5)
plt.figure(figsize=(10, 6))
sns.histplot(df['Tree Density'], 
             color = 'green', 
             label = 'Vegetation surface %', 
             kde   = False, 
             bins  = bins, 
             alpha=0.6)
sns.histplot(df['Building Density'], 
             color = 'orange', 
             label = 'Building surface %', 
             kde   = False, 
             bins  = bins, 
             alpha = 0.6)

# Add labels and title
plt.xlabel('Surface %', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.title('Distribution of Vegetation and Building density', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.legend()
plt.savefig(os.path.join(path, f'FeatureSurfaceDistribution.png'))
# Show the plot
plt.show()


#####################################################
#             TREE AND BUILDING HEIGHT              #
#####################################################
plt.figure(figsize=(10, 6))
sns.histplot(df['Tree Height Avg'], 
             color = 'green', 
             label = 'Vegetation Height', 
             kde   = False, 
             alpha =0.6)
sns.histplot(df['Building Height Avg'], 
             color = 'orange', 
             label = 'Building Height', 
             kde   = False,
             alpha = 0.6)

# Add labels and title
plt.xlabel('Height [m]', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.title('Distribution of Vegetation and Building height', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.legend()
plt.savefig(os.path.join(path, f'FeatureHeightDistribution.png'))
# Show the plot
plt.show()


#####################################################
#             TREE AND BUILDING STDEV               #
#####################################################
plt.figure(figsize=(10, 6))
sns.histplot(df['Tree Height Stdev'], 
             color = 'green', 
             label = 'Vegetation Height stdev', 
             kde   = False, 
             alpha =0.6)
sns.histplot(df['Building Height Stdev'], 
             color = 'orange', 
             label = 'Building Height stdev' , 
             kde   = False,
             alpha = 0.6)

# Add labels and title
plt.xlabel('Height Stdev [m]', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.title('Distribution of Vegetation and Building height stdev', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.legend()
plt.savefig(os.path.join(path, f'FeatureHeightStdevDistribution.png'))
# Show the plot
plt.show()


#####################################################
#       SHADOW FACTOR VS TREE HEIGHT SCATTER        #
#####################################################
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the scatter plot for 'With CDSM'
sns.scatterplot(x=df['Tree Height Avg'], 
                y=df[f'Shadow Factor {when} (With CDSM)'], 
                color='green', 
                label='With CDSM')
# Plot the scatter plot for 'Without CDSM'
sns.scatterplot(x=df['Tree Height Avg'], 
                y=df[f'Shadow Factor {when} (Without CDSM)'], 
                color='orange', 
                label='Without CDSM')
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add labels and title
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Fine grid with transparency
plt.xlabel('Average Tree Heigth [m]', fontsize = 14)
plt.ylabel(f'Shadow Factor at {when}', fontsize = 14)
plt.title(f'Tree Height vs Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)

# Add a legend
plt.legend()
plt.savefig(os.path.join(path, f'Scatter_Height_VS_Shadow_{when}.png'))
# Show the plot
plt.show()


#####################################################
#       SHADOW FACTOR VS TREE DENSITY SCATTER       #
#####################################################
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the scatter plot for 'With CDSM'
sns.scatterplot(x=df['Tree Density'], 
                y=df[f'Shadow Factor {when} (With CDSM)'], 
                color='green', 
                label='With CDSM')
# Plot the scatter plot for 'Without CDSM'
sns.scatterplot(x=df['Tree Density'], 
                y=df[f'Shadow Factor {when} (Without CDSM)'], 
                color='orange', 
                label='Without CDSM')

# Add labels and title
plt.xlabel('Vegetation surface [%] ', fontsize = 14)
plt.ylabel(f'Shadow Factor at {when}', fontsize = 14)
plt.title(f'Tree Density vs Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Fine grid with transparency
plt.legend()
plt.savefig(os.path.join(path, f'Scatter_Density_VS_Shadow_{when}.png'))
# Show the plot
plt.show()


#####################################################
#  SHADOW FACTOR VS TREE/BUILDING DENSITY SCATTER   #
#####################################################
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the scatter plot for 'With CDSM'
sns.scatterplot(x=df['Tree Density'], 
                y=df[f'Shadow Factor {when} (With CDSM)'], 
                color='green', 
                label='With CDSM')
# Plot the scatter plot for 'Without CDSM'
sns.scatterplot(x=df['Building Density'], 
                y=df[f'Shadow Factor {when} (Without CDSM)'], 
                color='orange', 
                label='Without CDSM')

# Add labels and title
plt.xlabel('Vegetation/Building surface [%]', fontsize = 14)
plt.ylabel(f'Shadow Factor at {when}', fontsize = 14)
plt.title(f'Tree/Building Density vs Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Fine grid with transparency
plt.legend()
plt.savefig(os.path.join(path, f'Scatter_TreeBuildings_Density_VS_Shadow_{when}.png'))
# Show the plot
plt.show()


#####################################################
#     SHADOW FACTOR VS BUILDING HEIGHT SCATTER      #
#####################################################
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the scatter plot for 'With CDSM'
sns.scatterplot(x=df['Building Height Avg'], 
                y=df[f'Shadow Factor {when} (With CDSM)'], 
                color='green', 
                label='With CDSM')
# Plot the scatter plot for 'Without CDSM'
sns.scatterplot(x=df['Building Height Avg'], 
                y=df[f'Shadow Factor {when} (Without CDSM)'], 
                color='orange', 
                label='Without CDSM')

# Add labels and title
plt.xlabel('Average Building Heigth [m]', fontsize = 14)
plt.ylabel(f'Shadow Factor at {when}', fontsize = 14)
plt.title(f'Building Height vs Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks


# Add a legend
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Fine grid with transparency
plt.legend()
plt.savefig(os.path.join(path, f'Scatter_BuildingHeight_VS_Shadow_{when}.png'))
# Show the plot
plt.show()


#####################################################
#    SHADOW FACTOR VS BUILDING DENSITY SCATTER      #
#####################################################
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the scatter plot for 'With CDSM'
sns.scatterplot(x=df['Building Density'], 
                y=df[f'Shadow Factor {when} (With CDSM)'], 
                color='green', 
                label='With CDSM')
# Plot the scatter plot for 'Without CDSM'
sns.scatterplot(x=df['Building Density'], 
                y=df[f'Shadow Factor {when} (Without CDSM)'], 
                color='orange', 
                label='Without CDSM')

# Add labels and title
plt.xlabel('Building surface [%]', fontsize = 14)
plt.ylabel(f'Shadow Factor at {when}', fontsize = 14)
plt.title(f'Building Density vs Shadow Factor at {when} (With and Without CDSM)', fontsize = 15)
plt.xticks(fontsize=12)  # Adjust the font size for x-axis tick marks
plt.yticks(fontsize=12)  # Adjust the font size for y-axis tick marks

# Add a legend
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Fine grid with transparency
plt.savefig(os.path.join(path, f'Scatter_BuildingDensity_VS_Shadow_{when}.png'))
# Show the plot
plt.show()