#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:59:28 2017

@author: zhenrui
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import pandas as pd
import pickle

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA



def visData(df, xlim=12, ylim=12, theta=None, fname='rawdata.png', n_feature=10):
    plt.figure(figsize=(50,90))
    ctr = 0
    for i in range(n_feature):
        for j in range(i):
            plt.subplot(9,5,ctr+1)
            ctr+=1
            plt.scatter(df[df["label"] == 1][i], df[df["label"] == 1][j], color='b', s=200)
            plt.scatter(df[df["label"] == 2][i], df[df["label"] == 2][j], color='r', s=200)
            
            if theta is not None:
                xs = np.arange(-xlim,xlim,0.1)
                ys = list(map(lambda x: (-theta[0]-theta[i]*x)/theta[j], xs))
                plt.plot(xs,ys, linewidth=5, color='black')
            plt.xlim([-xlim,xlim])
            plt.ylim([-ylim,ylim])
            plt.xlabel('Dimension {}'.format(i))
            plt.ylabel('Dimension {}'.format(j))
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def train():   
    #load orignial data set 
    X = pickle.load( open( "Data_x.pkl", "rb" ) )
 
    y = pickle.load( open( "Data_y.pkl", "rb" ) )    
    n =X.shape[0]
    d= X.shape[1]

    #modify the data set
    Xt=np.copy(X)  
    Xt2=np.copy(X)
    Xt3=np.copy(X)
    for i in range(50000, n):
        a= Xt[i]
        Xt[i]=a + 3

    for i in range(50000, n):
        a= Xt2[i]
        Xt2[i]=a + 0.5

    for i in range(50000, n):
        a= Xt3[i][0]
        Xt3[i][0]=a + 0.5

  
    print (Xt)
    print (Xt2)
    print (Xt3)
  
    
    dft2 = pd.DataFrame(np.vstack([Xt2 ]))
    dft2["label"]= np.hstack([y])
    
    dft = pd.DataFrame(np.vstack([Xt ]))
    dft["label"]= np.hstack([y])
    
    dft3 = pd.DataFrame(np.vstack([Xt3]))
    dft3["label"]= np.hstack([y])
    
 
    LLS=KNeighborsClassifier(n_neighbors=3)
    LLS= LinearSVC(C=1e5)
    LLS=KNeighborsClassifier(n_neighbors=3)
    LLS = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    
    pca=PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    Xt_reduced = pca.fit_transform(Xt)
    Xt2_reduced = pca.fit_transform(Xt2)
    Xt3_reduced = pca.fit_transform(Xt3)
    
    LLS=KNeighborsClassifier(n_neighbors=5)
    #LLS = AdaBoostClassifier(n_estimators=100)
    #cross-validation
    
    
    X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.3, random_state=0)
    
    
    #cross-validation  
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
            Xt_reduced, y, test_size=0.3, random_state=0)
    
    
    Xt2_train, Xt2_test, yt2_train, yt2_test = train_test_split(
            Xt2_reduced, y, test_size=0.3, random_state=0)
    
    
    Xt3_train, Xt3_test, yt3_train, yt3_test = train_test_split(
            Xt3_reduced, y, test_size=0.3, random_state=0)
    
    
    #fit model
    LLS.fit(X_train, y_train)
   
 
    #scores for fitting different data set
    train_accuracy = LLS.score(X_train, y_train)
    test_accuracy = LLS.score(X_test, y_test)
    traint_accuracy = LLS.score(Xt_train, yt_train)
    testt_accuracy = LLS.score(Xt_test, yt_test)
    train_accuracy2 = LLS.score(Xt2_train, yt2_train)
    test_accuracy2 = LLS.score(Xt2_test, yt2_test)
    train_accuracy3 = LLS.score(Xt3_reduced, y)
    test_accuracy3 = LLS.score(Xt3_test, yt3_test)
    
    
    print ("\n============")
    print ("orignal data")
    print("Training accuracy: {}".format(train_accuracy))
    print("Testing accuracy: {}".format(test_accuracy))
    
    print ("\n============")
    print ("modified fitting data") 
    print("Training accuracy: {}".format(traint_accuracy))
    print("Testing accuracy: {}".format(testt_accuracy))
    
    print ("\n============")
    print ("modified data2") 
    print("Training accuracy: {}".format(train_accuracy2))
    print("Testing accuracy: {}".format(test_accuracy2))
    
    print ("\n============")
    print ("modified data3") 
    print("Training accuracy: {}".format(train_accuracy3))
    print("Testing accuracy: {}".format(test_accuracy3))
    
   
    

  

    
    

    
    
    
    

if __name__ == "__main__":  
   train()
