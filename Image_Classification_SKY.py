#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:03:30 2019

@author: l-r-h
"""

#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

image_train = pd.read_csv('/Users/l-r-h/Desktop/IE/Term_2/AI:ML/Classification/segmentation.csv', sep=',')
img_test = pd.read_csv('/Users/l-r-h/Desktop/IE/Term_2/AI:ML/Classification/segtest.csv', sep=',' )

final = pd.read_csv('/Users/l-r-h/Desktop/IE/Term_2/AI:ML/Classification/segtest.csv', sep=',' )
#%%

sns.heatmap(image_train.corr())
sns.heatmap(image_train.corr(), annot=True) 
sns.pairplot(image_train)

corr = image_train.select_dtypes(include=[np.number]).corr()
#%%

image_train = image_train.reset_index()
image_train = image_train.drop(['REGION-PIXEL-COUNT'], axis=1)
image_train['Output'] = np.where((image_train['index']== 'SKY'), 1,0)
#cups_co['high']= np.where((cups_co['Hour']>=8) , 1,0)
Y_train = image_train['Output']

img_test = img_test.reset_index()
img_test = img_test.drop(['REGION-PIXEL-COUNT'], axis=1)
img_test['Output'] = np.where((img_test['index']== 'SKY'), 1,0)
Y_test = img_test['Output']
#%%
img_train = image_train
#%%

X_train = img_train[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2',
                    'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                    'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN','EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN','HUE-MEAN']]

#%%

X_test = img_test[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2',
                    'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                    'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN','EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN','HUE-MEAN']]

#%%
final = final.reset_index()
final = final.drop(['REGION-PIXEL-COUNT'], axis=1)
final['Output'] = np.where((final['index']== 'SKY'), 1,0)
#%%

X = final[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2',
                    'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                    'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN','EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN','HUE-MEAN']]


#%%
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB()}

    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()

#%%

dict_models = {}
train = []
train1= []
def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8, verbose = True):
    
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        train1.append(classifier.fit(X_train, Y_train))
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        train.append(train_score)
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models
    
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))
#%%

dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8)
display_dict_models(dict_models)   

#%%

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# establish baseline

#%%

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
Y = model.predict(X)
final['Output'].astype(int)
final['Prediction'] = Y

#%%

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(final['Output'], final['Prediction'])
print(cm)
#%%

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()