#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:43:00 2019


"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
    
def initializePCA(df,nFeatures):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:,:nFeatures].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    #df['pca-three'] = pca_result[:,2]
    #print(df.columns)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="pca-one", y="pca-two",
    hue=nFeatures,
    data=df
    )
    plt.show()


def wordCloud(df):
    #results = df.loc[df['IndirectH']==1]
    wordList = df['tweet_content'].values.tolist()
    text = ','.join(wordList).lower()
    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(['don','ava','http','amp','cc','rt','x89', 'x8f', 'x95', 'x9d', 'na','co','dont','dm'])
    #This function plots a wordcloud of the lemmas in Semcor data set
    wordcloud = WordCloud(width=1600, height=800,stopwords=stopwords,max_font_size=300, max_words=100,background_color="white",collocations=False).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    #'harassment','IndirectH','PhysicalH','SexualH'
