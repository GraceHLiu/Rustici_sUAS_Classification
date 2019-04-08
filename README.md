# Rustici_sUAS_Classification
In this project, we implemented random forest classifications on eight post-fire and post-juniper cut rangelands in Northern California. 

# Code discriptions
This code is designed for predicting classification maps based on training data in shapefiles and its corresponding multispectral input images (in .tif) using random forest methods.

1. Packages

Pandas, numpy, sklearn, osr, gdal, dbfread, glob, os, pydot*, matplotlib*, and pickle*.
*Pydot, matplotlib, and pickle are for pulling a single decision tree, plotting variable importance, and saving the trained random forest respectively; These packages are optional is the corresponding steps are not necessary.)

2. Implementation

To run the code for your own data, simply redefine the directory variables (Input_Raster_dir and Output_dir) and some necessary formating variables (e.g. in glob.glob())

# Contact
Reach out to me at my website at https://gracehliu.weebly.com/contact.html
