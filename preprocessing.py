#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:11:16 2019


"""
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

def trainTaskA(df,numFeatures,labelIndex):
    #This function balances the labels of each class by use od smote
    X_train = df.iloc[:,:numFeatures]
    y_train = df.iloc[:,labelIndex]
    
    ###########################################################################
    # Apply SMOTE
    
    smt = SMOTE()
    X_train, y_train = smt.fit_sample(X_train, y_train)
    
    return X_train, y_train

def readData(file):
    df = pd.read_csv(file,sep=',', header=0, index_col=0)
    return df

def readEmbeddeddata(file):
    df = pd.read_csv(file,sep=',', header=0)
    return df

def preProcessTraining(dataFrame,numFeatures):
    additionalStopWords = ['don','http','amp','cc','rt','x89', 'x8f', 'x95', 'x9d', 'na']
    answer = []
    englishStemmer=SnowballStemmer("english")
    vectorizer = TfidfVectorizer(stop_words='english',max_features=numFeatures)
    corpus = []
    for index, row in dataFrame.iterrows():
        sentence_words = nltk.word_tokenize(row['tweet_content'])
        word_list = []
        for word in sentence_words:
            word = englishStemmer.stem(word)
            if word not in additionalStopWords:
                word_list.append(word)
                word_list.append(' ')
            else: pass
        corpus.append(''.join(word_list)) 
    vectors = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    print('Lenght of features list: {}.'.format(len(features)))
    print(features)
    vectors = vectors.toarray()
    for index, row in dataFrame.iterrows():
        vec = np.append(vectors[index],[row['harassment'],row['IndirectH'],row['PhysicalH'],row['SexualH']])
        answer.append(vec)
    newDataFrame = pd.DataFrame(answer)
    #print(newDataFrame)
    return newDataFrame, features

def preProcessTest(df,featuresList,status):
    #Status == 0 -> validation
    #Status == 1 -> test
    additionalStopWords = ['don','http','amp','cc','rt','x89', 'x8f', 'x95', 'x9d','na']
    answer = []
    i = range(0,len(featuresList))
    featuresNames = dict(zip(featuresList, i))
    englishStemmer=SnowballStemmer("english")
    vectorizer = TfidfVectorizer(stop_words='english',max_features=len(featuresList),vocabulary=featuresNames)
    corpus = []
    for index, row in df.iterrows():
        sentence_words = nltk.word_tokenize(row['tweet_content'])
        word_list = []
        for word in sentence_words:
            word = englishStemmer.stem(word)
            if word not in additionalStopWords:
                word_list.append(word)
                word_list.append(' ')
            else: pass
        corpus.append(''.join(word_list)) 
    vectors = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    print('Lenght of features list: {}.'.format(len(features)))
    #print(features)
    vectors = vectors.toarray()
    #print(df)
    for index, row in df.iterrows():
        if status == 0:
            vec = np.append(vectors[index-6374],[row['harassment'],row['IndirectH'],row['PhysicalH'],row['SexualH']])
        elif status == 1:
            vec = np.append(vectors[index],[row['harassment'],row['IndirectH'],row['PhysicalH'],row['SexualH']])
        answer.append(vec)
    newDataFrame = pd.DataFrame(answer)
    return newDataFrame

def toLowercase(dataFrame):
    for index, row in dataFrame.iterrows():
        dataFrame.loc[index,['tweet_content']] = row['tweet_content'].lower()
    return dataFrame


