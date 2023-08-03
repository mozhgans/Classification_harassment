#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:23:47 2019


"""
import numpy as np
import pandas as pd
import preprocessing as pp
import models as md
import printdatasets as pr
from sklearn.preprocessing import MinMaxScaler


def main():
    #Step 1: read the data
    trainArch = 'insert your data here'
    validationArch = 'insert your data here'
    testArch = 'insert your data here'
    #trainArchSize = 6374
    control = -1
    print('---------------------------------------------------')
    print('- Welcome to the SIMAH Competition 2019!')
    print('---------------------------------------------------\n ') 
    while control != 0:
        print('Type one of the options below:\n 1 - Read The data set. \n 2 - Pre-process the data sets and run the models. \n 3 - Print the data set.\n 4 - Plot word clouds. \n press any other key to quit the system.')
        control = int(input('Which option would you like to perform?:'))
        if control == 1:
            trainingData = pp.readData(trainArch)
            validationData = pp.readData(validationArch)
            print('The training and test sets were read!')
        elif control == 2:
            trainTweets = trainingData
            testTweets = validationData
            X_trainTweets, y_trainTweets = pp.trainTaskA(trainTweets,numFeatures,labelIndex)
            md.initModels(X_trainTweets,y_trainTweets,testTweets,numFeatures,labelIndex)
        elif control == 3:
            numFeatures = 45
            pr.initializePCA(testTweets,numFeatures)
        elif control == 4:
            dfWC = df.loc[df['harassment']==1]
            pr.wordCloud(dfWC)
            ##################################
            dfiWC = df.loc[df['IndirectH']==1]
            pr.wordCloud(dfiWC)
            ##################################
            dfpWC = df.loc[df['PhysicalH']==1]
            pr.wordCloud(dfpWC)
            ##################################
            dfsWC = df.loc[df['SexualH']==1]
            pr.wordCloud(dfsWC)
            print('Total Instances for train: {}.'.format(dfWC.iloc[:,4].values.tolist().count(1)))
            print('Total Instances for test: {}.'.format(dfWC.iloc[:,4].values.tolist().count(1)))
            #testeWC = pp.readEmbeddeddata(competitionArch)
            #pr.wordCloud(trainWC)
            #pr.wordCloud(testeWC)
        else:
            control = 0
    
if __name__=='__main__':
    main()
    
    
