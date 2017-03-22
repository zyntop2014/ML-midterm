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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def genData():
    X0 = 2*np.random.randn(100,2)+3.
    X1 = 3*np.random.randn(100,2)
    
    df = pd.DataFrame(np.vstack([X0,X1]))
    df[2] = np.random.randn(200)-1
    df[3] = np.random.randn(200)-1
    df["label"] = np.hstack([np.zeros(100), np.ones(100)])
    
    return df

def visData(df, xlim=10, ylim=10, theta=None, fname='example.png'):
    plt.figure(figsize=(30,20))
    ctr = 0
    for i in xrange(4):
        for j in xrange(i):
            plt.subplot(2,3,ctr+1)
            ctr+=1
            plt.scatter(df[df["label"] == 0][i], df[df["label"] == 0][j], color='r', s=200)
            plt.scatter(df[df["label"] == 1][i], df[df["label"] == 1][j], color='b', s=200)
            if theta is not None:
                xs = np.arange(-xlim,xlim,0.1)
                ys = map(lambda x: (-theta[0]-theta[i]*x)/theta[j], xs)
                plt.plot(xs,ys, linewidth=5, color='black')
            plt.xlim([-xlim,xlim])
            plt.ylim([-ylim,ylim])
            plt.xlabel('Dimension {}'.format(i))
            plt.ylabel('Dimension {}'.format(j))
    
    plt.tight_layout()
    plt.savefig(fname)
    
def trainLogistic(X, y, test_size=0.3):
    
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.model_selection import train_test_split

    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0)

    LR = LogisticRegression(penalty='l1', tol=0.0001, C=1, 
                            fit_intercept=True, intercept_scaling=1, 
                            class_weight=None, random_state=None, 
                            solver='liblinear', max_iter=100)
    
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    train_accuracy = LR.score(X_train, y_train)
    test_accuracy = LR.score(X_test, y_test)
    print("Training accuracy: {}".format(train_accuracy))
    print("Testing accuracy: {}".format(test_accuracy))
    
    return LR
    

if __name__ == "__main__":

    df = genData()
    visData(df, fname='example.png')
        
    LR = trainLogistic(df[[0,1,2,3]], df["label"])
    theta = np.squeeze(LR.coef_)
    visData(df, theta=theta, fname='example_boundary.png')

    X_std = (df[[0,1,2,3]]-df[[0,1,2,3]].mean()) / df[[0,1,2,3]].std()
    LR = trainLogistic(X_std, df["label"])
    theta = np.squeeze(LR.coef_)
    visData(df, theta=theta, xlim=3, ylim=3, fname='example_std.png')
