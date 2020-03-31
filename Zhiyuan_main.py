# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:30:02 2020

@author: Zhiyuan Zhang
"""


import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from statistics import mean
import xgboost as xgb
import time
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

"""
Path
"""
DATA_PATH = "../data/train_set"
IMAGE_FOLDER = os.path.join(DATA_PATH, "images")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = DATA_PATH

def read_all_images(height = 750, width = 1000, crop_gray = False, points = None):
    """
    Read 2500 training images from the IMAGE_FOLDER, resized to 'height x width'
    
    :param height: resized height of images
    :param width: resized width of images
    :return: a 4d numpy array in form of (index, height, width, channels), channels is RGB 
    """
    files = [file for file in os.listdir(IMAGE_FOLDER) if file.endswith('.jpg')]
    files.sort()
    
    
    face_images_arr = np.zeros((len(files), height, width, 1 if crop_gray else 3 ))
    
    for index, filename in enumerate(files):
        face_img = Image.open(os.path.join(IMAGE_FOLDER, filename))
        # if crop_gray, then the image will be cropped to fit the facial part
        # and then it will return a grayscale version
        if crop_gray:
            face_img = face_img.convert('L')

            face_points = points[index]
            # calculate crop position
            left = np.min(face_points[:,0]) 
            right = np.max(face_points[:,0]) 
            top = np.min(face_points[:,1])
            bot = np.max(face_points[:,1])

            face_img = face_img.crop((left,top,right,bot))
        
        face_img = face_img.resize((width, height))
        # fit the dimension
        face_images_arr[index] = np.array(face_img).reshape((height,width, -1))
        
    return face_images_arr

def read_labels():
    """
    Read the image labels from the label.csv file
    :return: a pandas.DataFrame with 3 columns: 'emotion_idx','emotion_cat','type'
    """
    labels_df = pd.read_csv(os.path.join(LABELS_FOLDER, 'label.csv'))
    labels_df = labels_df.loc[:,['emotion_idx','emotion_cat','type']]
    return labels_df
    

def read_all_points():
    """
    Read all face coordinates points
    :return: a tuple of shape (2500, 78, 2). Because for each of 2500 images there are 78 points associated with it
    """
    files = [file for file in os.listdir(POINTS_FOLDER) if file.endswith('.mat')]
    files.sort()
    
    face_points = np.zeros((len(files), 78, 2))
    for index, filename in enumerate(files):
        face_points_dict = loadmat(os.path.join(POINTS_FOLDER, filename))
    
        face_points[index] = face_points_dict.get('faceCoordinatesUnwarped',  face_points_dict.get('faceCoordinates2'))
    return face_points

def load_data(loadImage = False, height = 750, width = 1000, crop_gray = False):
    """
    Load training data from local files
    
    :param loadImage: if it's False, this function will not load original images
    :return: a tuple (images, points, labels)
        if loadImage is False, the 'images' will None. Otherwise its a numpy array with shape (2500,750,1000,3)
        points is a numpy array with shape (2500, 78, 2)
        labels is a pandas.DataFrame
    """
    
    face_images_points = read_all_points()
    
    face_images_ndarr =  read_all_images(height, width, crop_gray, face_images_points) if loadImage else None
    labels = read_labels()
    
    return face_images_ndarr, face_images_points, labels



def show_image(index, all_images = None):
    """
    Display the (index)th image.
    all_images is passed, the this image numpy array can be easily retrieved from it. 
    Otherwise the original images needs to be read from disk
    
    :param index: the index to specify which image to disply
    :param all_images: the return value of 'read_all_images' function
    """
    
    if all_images is not None and index < len(all_images):
        face_img_arr = all_images[index].astype('uint8')
        if face_img_arr.shape[2] == 1:
            face_img_arr = face_img_arr.reshape((face_img_arr.shape[0],face_img_arr.shape[1]))
    else:
        face_img_arr = plt.imread(os.path.join(IMAGE_FOLDER, f"{index:04}.jpg"))
    plt.imshow(face_img_arr, cmap='gray')
    plt.show() 
height = 200
width = 200
images, points, labels = load_data(loadImage= False,crop_gray=True, height = height, width = width)

if images:
    print(images.shape)
print(points.shape)
##define distance
distances = np.zeros((2500, 3003))
for i in range(2500):
    current = points[i]
    distances[i,] = pdist(current)

X_points = points.reshape((points.shape[0], -1))
X = distances
Y = labels['emotion_idx']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=666)
#%%
### baseline knn tuning
neighbors = np.arange(5, 50, 10) 
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
      
    # Compute traning and test data accuracy 
    train_accuracy[i] = knn.score(X_train, y_train) 
    test_accuracy[i] = knn.score(X_test, y_test) 

plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show()
#%%
### baseline knn
knn_start = time.time()
baseline = KNeighborsClassifier(n_neighbors=25)
baseline.fit(X_train, y_train)
knn_end = time.time() #end time
print("Training Time:","%s seconds"%(knn_end - knn_start))
print("BaseLine training dataset Accuracy")
knn_train_accuracy = baseline.score(X_train, y_train) 
print(knn_train_accuracy)
print("BaseLine testing dataset Accuracy")
test_accuracy = baseline.score(X_test, y_test) 
print(test_accuracy)

#%%
### GBM 
GBM_start = time.time()
gbm = GradientBoostingClassifier(learning_rate=0.2,min_samples_split = 4,n_estimators=30,min_samples_leaf =1)
gbm.fit(X_train, y_train)
GBM_end = time.time() 
print("GBM Training Time:","%s seconds"%(GBM_end - GBM_start))
print("GBM Training dataset Accuracy")
GBM_train_accuracy = gbm.score(X_train, y_train)
print(GBM_train_accuracy)
gbm_preds = gbm.predict(X_test)
print("GBM Testing dataset Accuracy")
GBM_test_accuracy = gbm.score(X_test, y_test)
print(GBM_test_accuracy)

#%%
### Random Forest
RF_start = time.time()
param_grid = {
    'max_depth': [10, 20],
    'max_features': [2, 3],
    'min_samples_leaf': [10, 15],
    'min_samples_split': [10, 12],
    'n_estimators': [200, 400]
}
rf = RandomForestClassifier()
rf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
best_rf = rf.fit(X_train, y_train)
RF_end = time.time()
print("Random Forest Training Time:","%s seconds"%(RF_end - RF_start))
print("Random Forest Training dataset Accuracy")
RF_train_accuracy = best_rf.score(X_train, y_train)
print(RF_train_accuracy)
RF_preds = best_rf.predict(X_test)
print("Random Forest Testing dataset Accuracy")
RF_test_accuracy = best_rf.score(X_test, y_test)
print(RF_test_accuracy)

#%%
### Logistic Regression
lr_start = time.time()
param_grid = {'C': [0.001, 0.01, 1, 5, 10, 25] }
lr = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=5000)
lr = GridSearchCV(estimator = lr, param_grid = param_grid, cv = 3, n_jobs = -1)
best_lr = lr.fit(X_train, y_train)
lr_end = time.time()
print("Logistic Regression Training Time:","%s seconds"%(lr_end - lr_start))
print("Logistic Regression Training dataset Accuracy")
lr_train_accuracy = best_lr.score(X_train, y_train)
print(lr_train_accuracy)
lr_preds = best_lr.predict(X_test)
print("Logistic Regression Testing dataset Accuracy")
lr_test_accuracy = best_lr.score(X_test, y_test)
print(lr_test_accuracy)

#%%
### xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'booster':'gbtree',
'objective': 'multi:softmax', 
'num_class':23,
'n_estimators':500,
'max_depth':20, 
'alpha': 3,
'gamma': 1,
'silent': 1,
'subsample': 0.8,
'eta': 0.1,
'learning_rates': 0.03}
num_round= 5

xgboost_start = time.time() 
bst = xgb.train(param, dtrain, num_round)
xgboost_end = time.time() 
print("xgboost training Time:","%s seconds"%(xgboost_end - xgboost_start))
print("xgboost training dataset Accuracy")
xgboost_train_preds = bst.predict(dtrain)
xgboost_train_accuracy = accuracy_score(y_train, xgboost_train_preds) 
print(xgboost_train_accuracy)
print("xgboost_train_testing dataset Accuracy")
xgboost_preds = bst.predict(dtest)
xgboost_test_accuracy = accuracy_score(y_test, xgboost_preds) 
print(xgboost_test_accuracy)

#%%
### svc
svc_start =  time.time()
param_grid = {'C':[0.03,0.1, 0.5, 1]}
svc_start = time.time()
svc = LinearSVC(max_iter = 5000)
svc = GridSearchCV(estimator = svc, param_grid = param_grid, cv = 3, n_jobs = -1)
best_svc = svc.fit(X_train, y_train)
svc_end = time.time()

print("SVM Training Time:","%s seconds"%(svc_end - svc_start))
print("SVM Training dataset Accuracy")
svc_train_accuracy = best_svc.score(X_train, y_train)
print(svc_train_accuracy)
svc_preds = best_svc.predict(X_test)
print("SVM Testing dataset Accuracy")
svc_test_accuracy = best_svc.score(X_test, y_test)
print(svc_test_accuracy)

#%%
### svm with kernal rbf
svc_rbf_start = time.time()
rbfsvc = SVC(kernel='rbf')
svc_rbf = rbfsvc.fit(X_train, y_train)
svc_rbf_end = time.time()

print("SVM RBF Training Time:","%s seconds"%(svc_rbf_end - svc_rbf_start))
print("SVM RBF Training dataset Accuracy")
svc_rbf_train_accuracy = svc_rbf.score(X_train, y_train)
print(svc_rbf_train_accuracy)
svc_rbf_preds = svc_rbf.predict(X_test)
print("SVM RBF Testing dataset Accuracy")
svc_rbf_test_accuracy = svc_rbf.score(X_test, y_test)
print(svc_rbf_test_accuracy)

#%%
### svm with kernal poly
svc_poly_start = time.time()
polysvc = SVC(kernel='poly')
svc_poly = polysvc.fit(X_train, y_train)
svc_poly_end = time.time()

print("SVM POLY Training Time:","%s seconds"%(svc_poly_end - svc_poly_start))
print("SVM POLY Training dataset Accuracy")
svc_poly_train_accuracy = svc_poly.score(X_train, y_train)
print(svc_poly_train_accuracy)
svc_poly_preds = svc_poly.predict(X_test)
print("SVM POLY Testing dataset Accuracy")
svc_poly_test_accuracy = svc_poly.score(X_test, y_test)
print(svc_poly_test_accuracy)










