# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:03:10 2020

@author: thecr
"""


import nltk
import pandas
import sklearn
from sklearn.cluster import KMeans
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from wordcloud import WordCloud, STOPWORDS

lastStatementFileName = "offenders_label_ROR.csv"

lastStatementCV = CountVectorizer(input='content', analyzer='word', stop_words='english', lowercase=True, max_features=100)
lastStatementTV = TfidfVectorizer(input='content', stop_words='english',max_features=100)

allStatements = []
allSentiments = []

#file = open(filename, encoding="utf8")

with open(lastStatementFileName, 'r', encoding='utf8', errors='ignore', newline='\n') as file:
    #file.encode('utf-8').strip()
    print("Test")
    #file.strip()
    file.readline()
    try:
        for row in file:
            #print(row)
            sentimentLabel, statement = row.split(',',1)
            allStatements.append(statement)
            allSentiments.append(sentimentLabel)
    except ValueError:
        print(row)
        
#print(len(allStatements))
        
lastStatementCVFitTransform = lastStatementCV.fit_transform(allStatements)
lastStatementTVFitTransform = lastStatementTV.fit_transform(allStatements)

colNames = lastStatementTV.get_feature_names()
#print(colNames)

sentimentDataFrame = pandas.DataFrame(lastStatementCVFitTransform.toarray(), columns=colNames)
#print(sentimentDataFrame)

for nextCol in sentimentDataFrame.columns:
    if(any(char.isdigit() for char in nextCol)):
        sentimentDataFrame.drop([nextCol], axis=1)
    elif(len(str(nextCol)) <= 3):
        sentimentDataFrame.drop([nextCol], axis=1)
    
#sentimentDataFrame.insert(loc=0, column='LABEL', value=allSentiments)
#print(sentimentDataFrame)

sentimentMatrix = sentimentDataFrame.values

# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(sentimentMatrix)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(sentimentMatrix)

#print(sentimentDataFrame)

# plot the 3 clusters
plt.scatter(
    sentimentMatrix[y_km == 0, 0], sentimentMatrix[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    sentimentMatrix[y_km == 1, 0], sentimentMatrix[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    sentimentMatrix[y_km == 2, 0], sentimentMatrix[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(sentimentMatrix)
y_kmeans = kmeans.predict(sentimentMatrix)

plt.scatter(sentimentMatrix[:, 0], sentimentMatrix[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
