
# coding: utf-8

# In[2]:


import glob
import pandas as pd
from dbfread import DBF
# read in ROI dbf
Input_Raster_dir = 'D:/Box Sync/research/Leslie/Classification_sUAS/'
Files = glob.glob(Input_Raster_dir + '/**/Training*.dbf', recursive=True)
# Set the output dir
Output_dir = 'D:/Box Sync/research/Leslie/Classification_sUAS/Results'


# In[3]:


# Read in training data table and display first 5 rows
features = pd.DataFrame()#creates a new dataframe that's empty
for file in Files:
    table = DBF(file)
    features_temp = pd.DataFrame(iter(table))
    features_temp.columns = ['Id','Class','ClassID','Blue','Green','Red','NIR','Rededge','VegHeight','NDVI_April','NDVI_May','NDVI_June','sUAS_NDVI']
    features = pd.concat([features, features_temp])


# In[4]:


print('The shape of our features is:', features.shape)


# In[5]:


# Descriptive statistics for each column
features.describe()


# In[6]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['ClassID'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Class', axis = 1)
#features= features.drop('FID', axis = 1)
features= features.drop('ClassID', axis = 1)
features= features.drop('Id', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
print(features)
features = np.array(features)


# In[7]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 40)


# In[8]:


print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[9]:


from sklearn import model_selection
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier(oob_score=True)
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)


# In[10]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
y = sklearn.preprocessing.label_binarize(labels, classes=[1, 2, 3, 4, 5])
X = features
#rfc_cv_score = cross_val_score(rfc,X, y, cv=2, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
# print("=== All AUC Scores ===")
# print(rfc_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rfc.oob_score_ * 100))


# In[11]:


from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
print(rfc_random.best_params_)


# In[14]:


rfc = RandomForestClassifier(n_estimators = 1400,#rfc_random.best_params_['n_estimators'],
                             max_features = 'auto',#rfc_random.best_params_['max_features'],
                             max_depth = 260,#rfc_random.best_params_['max_depth'],
                             oob_score=True)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_predict_train = rfc.predict(X_train)
print("=== Confusion Matrix (test) ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Confusion Matrix (train) ===")
print(confusion_matrix(y_train, rfc_predict_train))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rfc.oob_score_ * 100))


# In[15]:


# Fit the model
rfc_random.fit(features, labels)
# print results
print(rfc_random.best_params_)


# In[16]:


rfc_full = RandomForestClassifier(n_estimators = 2000,#rfc_random.best_params_['n_estimators'],
                                  max_features ='auto',# rfc_random.best_params_['max_features'],
                                  max_depth = 300,#rfc_random.best_params_['max_depth'],
                                  oob_score=True)
rfc_full.fit(features,labels)
print('Our OOB prediction of accuracy for the full model is: {oob}%'.format(oob=rfc_full.oob_score_ * 100))


# In[15]:


import os
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rfc.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rfc.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'Tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('Tree.dot')
# Write graph to a png file
graph.write_png(os.path.join(Output_dir,'Tree.tif'))


# In[19]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rfc_full.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rfc_full.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'Tree_full.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('Tree_full.dot')
# Write graph to a png file
graph.write_png(os.path.join(Output_dir,'Tree_full.png'))


# In[17]:


# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[18]:


# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[26]:


# # saving full model
# from sklearn.externals import joblib
# # Save to file in the current working directory
# joblib_file = os.path.join(Output_dir,'RFmodel_10b_full.pickle')
# joblib.dump(rfc_full, joblib_file)

# # Load from file
# joblib_model = joblib.load(joblib_file)

# # Calculate the accuracy and predictions
# score = joblib_model.score(X_test, y_test)  
# print("Test score: {0:.2f} %".format(100 * score))  


# In[20]:


# saving full model using pickle
import os
import pickle
# Save to file in the current working directory
pkl_filename = os.path.join(Output_dir,'RFmodel_10b_full_V2.pickle')
with open(pkl_filename, 'wb') as file:  
    pickle.dump(rfc_full, file)
# # Load from file
# with open(pkl_filename, 'rb') as file:  
#     pickle_model = pickle.load(file)

# # Calculate the accuracy score and predict target values
# score = pickle_model.score(X_test, y_test)  
# print("Test score: {0:.2f} %".format(100 * score))   


# In[21]:


# Predicting the rest of the image
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import os
import osr
def raster2array(rasterfn,i):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(i)
    return band.ReadAsArray()
def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    


# In[22]:


# Read in training data table and display first 5 rows
#Sites = ['Rush_Fire']#,'Ashley_Fire','Blue_Door_Fire','Blue_Fire','Horse_Fire_North','Horse_Fire_South','Horse_Lake_Fire','Nelson_Fire','Scorpion_Fire']        
Files = glob.glob(Input_Raster_dir + '/**/Input_*.tif', recursive=True)
for file in Files:
    print('working on ' + file )
    site = os.path.basename(file)[6:-4]
    img_ds = gdal.Open(file, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for i in np.arange(img_ds.RasterCount):
        img[:,:,i] = raster2array(file,int(i)+1)
    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape,
                                            n=img_as_array.shape))
    # Now predict for each pixel
    class_prediction = rfc_full.predict(img_as_array)
    # Reshape our classification map
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    # Set example dir for meta data
    Example_dir = glob.glob(os.path.join(Input_Raster_dir,site,'Raster','I_Cliped','*Multispectral_NDVI.tif'))[0]
    # Set the output dir
    Classification_dir = os.path.join(Input_Raster_dir,site,'Raster','III_Classification')
    if not os.path.exists(Classification_dir):
        os.makedirs(Classification_dir)
    array2raster(Example_dir,
                os.path.join(Classification_dir,site+'_RFClass_V2.tif'),
                class_prediction)


# In[23]:


## visiualization
# plt.imshow(class_prediction, interpolation='none')
# plt.show()

