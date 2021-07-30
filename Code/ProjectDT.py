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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn import tree
import seaborn

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
    
sentimentMatrix = sentimentDataFrame.values

sentimentTrainDF, sentimentTestDF = train_test_split(sentimentDataFrame, test_size=.3)

X_train, X_test, y_train, y_test = train_test_split(allStatements, allSentiments, test_size=0.3, random_state=1337)
X_train = lastStatementCV.fit_transform(X_train)
X_test = lastStatementCV.fit_transform(X_test)

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print(lastStatementTV.get_feature_names())

plt.figure(figsize=(25,10))
a = plot_tree(dt, feature_names=lastStatementTV.get_feature_names(),
              class_names=lastStatementTV.get_feature_names(),
              filled=True,
              rounded=True,
              fontsize=14)