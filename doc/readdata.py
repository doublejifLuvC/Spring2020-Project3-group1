# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:36:45 2020

@author: zhang
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
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = DATA_PATH


def read_labels():
    labels_df = pd.read_csv(os.path.join(LABELS_FOLDER, 'label.csv'))
    labels_df = labels_df.loc[:,['emotion_idx','emotion_cat','type']]
    return labels_df

def read_points():
    files = [file for file in os.listdir(POINTS_FOLDER) if file.endswith('.mat')]
    files.sort()
    
    face_points = np.zeros((len(files), 78, 2))
    for index, filename in enumerate(files):
        face_points_dict = loadmat(os.path.join(POINTS_FOLDER, filename))
    
        face_points[index] = face_points_dict.get('faceCoordinatesUnwarped',  face_points_dict.get('faceCoordinates2'))
    return face_points

points = read_points()
labels = read_labels()
### train test split
X_points_train, X_points_test, y_train, y_test = train_test_split(points,labels,test_size=0.2, random_state=666)

### Feature Extraction time on training set:
feature_training_start = time.time()
X_train = np.zeros((X_points_train.shape[0], 3003))
for i in range(X_points_train.shape[0]):
    current = X_points_train[i]
    X_train[i,] = pdist(current)
feature_training_end = time.time()
print("Feature Extraction time on training set:","%s seconds"%(feature_training_end - feature_training_start))
    
### Feature Extraction time on test set:
feature_test_start = time.time()
X_test = np.zeros((X_points_test.shape[0], 3003))
for i in range(X_points_test.shape[0]):
    current = X_points_test[i]
    X_test[i,] = pdist(current)
feature_test_end = time.time()
print("Feature Extraction time on test set:","%s seconds"%(feature_test_end - feature_test_start))



### predict label
predict_DATA_PATH = "../data/test_set"
POINTS_FOLDER = os.path.join(predict_DATA_PATH, "points")

predict_points = read_points()

predict_start = time.time()
X_predict = np.zeros((predict_points.shape[0], 3003))
for i in range(predict_points.shape[0]):
    current = predict_points[i]
    X_predict[i,] = pdist(current)
predict_end = time.time()
print("Feature Extraction time on test set:","%s seconds"%(predict_end - predict_start))



