#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:26:04 2019


"""
import numpy as np
import pandas as pd 
import sklearn as sk  
from sklearn.linear_model import LogisticRegression  
from sklearn import svm  
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
import preprocessing as pp


#This function runs the supervised models of Logistic Regression, SVM, Random Forest, and MLP
def initModels(X_trainTweets, y_trainTweets,testData,numFeatures,labelIndex):
    train_x = X_trainTweets
    train_y = y_trainTweets
    test_x = testData.iloc[:,:numFeatures]
    test_y = testData.iloc[:,labelIndex]#.values.reshape(-1, 1)
    
    warnings.filterwarnings("ignore")
    
    print('Results for {} features:'.format(numFeatures))
    #Logistic Regression

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_x,train_y)  
    scoresLR = cross_val_score(LR,train_x,train_y, cv=10)
    predsLR = LR.predict(test_x)  
    #print('Scores LR:')
    #print(scoresLR)
    print('F1 Score for Logistic Regression: {}.'.format(round(f1_score(test_y, predsLR, average='macro'), 4))) 
    print('Accuracy for Logistic Regression: {}.'.format(round(LR.score(test_x,test_y), 4))) 

    
    #Naive Base
    gnb = GaussianNB()
    scoresGNB = cross_val_score(gnb,train_x,train_y, cv=10)
    gnb.fit(train_x,train_y).predict(test_x)
    predsGNB = gnb.predict(test_x)  
    #print('Scores GNB:')
    #print(scoresGNB)
    print('F1 Score for Naive Bayes: {}.'.format(round(f1_score(test_y, predsGNB, average='macro'), 4))) 
    print('Accuracy for Naive Bayes: {}.'.format(round(gnb.score(test_x, test_y), 4))) 
    
    #Decision tree:
    clf = tree.DecisionTreeClassifier()
    scoresCLF = cross_val_score(clf,train_x,train_y, cv=10)
    clf = clf.fit(train_x, train_y)
    predsCLF = clf.predict(test_x) 
    print('F1 Score for Decision tree: {}.'.format(round(f1_score(test_y, predsCLF, average='macro'), 4))) 
    print('Accuracy for Decision tree: {}.'.format(round(clf.score(test_x, test_y), 4))) 
    #print('Scores CLF:')
    #print(scoresCLF)


    # Random forest:

    RF=RandomForestClassifier(random_state=10)
    scoresRF = cross_val_score(RF,train_x,train_y, cv=10)
    RF.fit(train_x, train_y)
    predsRF = RF.predict(test_x) 
    print('F1 Score for Random Forest: {}.'.format(round(f1_score(test_y, predsRF, average='macro'), 4))) 
    print('Accuracy for Random Forest: {}.'.format(round(RF.score(test_x, test_y), 4)))
    #print('Scores RF:')
    #print(scoresRF)

    
    #Linear SVM
    SVM = svm.LinearSVC()  
    scoresSVM = cross_val_score(SVM,train_x,train_y, cv=10)
    SVM.fit(train_x, train_y).predict(test_x)  
    predsSVM = SVM.predict(test_x) 
    print('F1 Score for Linear SVM: {}.'.format(round(f1_score(test_y, predsSVM, average='macro'), 4))) 
    print('Accuracy for Linear SVM: {}.'.format(round(SVM.score(test_x,test_y), 4)))  
    
    #Gaussian SVM
    GSVM = svm.SVC()
    scoresGSVM = cross_val_score(GSVM,train_x,train_y, cv=10)
    GSVM.fit(train_x, train_y)
    predsGSVM = GSVM.predict(test_x) 
    print('F1 Score for Gaussian SVM: {}.'.format(round(f1_score(test_y, predsGSVM, average='macro'), 4))) 
    
    print('Accuracy for Gaussian SVM: {}.'.format(round(GSVM.score(test_x,test_y), 4))) 
    #print('Scores GSVM:')
    #print(scoresGSVM)
    
    #Poly SVM
    POLYSVM = svm.SVC(kernel='poly',degree=2)
    scoresPOLYSVM = cross_val_score(POLYSVM,train_x,train_y, cv=10)
    POLYSVM.fit(train_x, train_y)
    predsPOLYSVM = POLYSVM.predict(test_x) 
    print('F1 Score for Polynomial SVM: {}.'.format(round(f1_score(test_y, predsPOLYSVM, average='macro'), 4))) 
    
    print('Accuracy for Polynomial SVM: {}.'.format(round(POLYSVM.score(test_x,test_y), 4)))
    #print('Scores polySVM:')
    #print(scoresPOLYSVM)

    #MLP
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)  
    scoresNN = cross_val_score(NN,train_x,train_y, cv=10)
    NN.fit(train_x, train_y)
    predsNN = NN.predict(test_x) 
    print('F1 Score for MLP: {}.'.format(round(f1_score(test_y, predsNN, average='macro'), 4))) 
    print('Accuracy for MLP: {}.'.format(round(NN.score(test_x,test_y), 4)))
    #print('Scores NN:')
    #print(scoresNN)
    
    #AdaBoost
    adb = AdaBoostClassifier(n_estimators=100, random_state=0)
    scoresADB = cross_val_score(adb,train_x,train_y, cv=10)
    adb.fit(train_x, train_y) 
    predsADB = adb.predict(test_x) 
    print('F1 Score for Adaboost: {}.'.format(round(f1_score(test_y, predsADB, average='macro'), 4))) 
    print('Accuracy for Adaboost: {}.'.format(round(adb.score(test_x,test_y), 4)))
    #print('Scores ADB:')
    #print(scoresADB)

    
    
    
    
    
