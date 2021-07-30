# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:46:07 2020

@author: thecr
"""

import nltk
import pandas
import sklearn
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy
import seaborn

lastStatementFileName = "offenders_label_Remorse.csv"

sentimentDF = pandas.DataFrame()

allSentiments = []
allStatements = []

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
    
print(len(allStatements))

statementCV = CountVectorizer(input='content', analyzer='word', stop_words='english',lowercase=True, max_features=100)
statementTV = TfidfVectorizer(input='content',stop_words='english',max_features=100)

statementCVFT = statementCV.fit_transform(allStatements)
statementTVFT = statementTV.fit_transform(allStatements)

colNames = statementTV.get_feature_names()

sentimentDataFrame = pandas.DataFrame(statementCVFT.toarray(), columns=colNames)

for nextCol in sentimentDataFrame.columns:
    if(any(char.isdigit() for char in nextCol)):
        sentimentDataFrame.drop([nextCol], axis=1)
    elif(len(str(nextCol)) <= 3):
        sentimentDataFrame.drop([nextCol], axis=1)
            
sentimentDataFrame.insert(loc = 0, column = 'LABEL', value = allSentiments)

#print(sentimentDataFrame)

sentimentTrainDF, sentimentTestDF = train_test_split(sentimentDataFrame, test_size=.3)
testSentimentLabels = sentimentTestDF["LABEL"]
#print(testSentimentLabels)
sentimentTestDF = sentimentTestDF.drop(["LABEL"], axis=1)

trainSentimentLabels = sentimentTrainDF["LABEL"]
sentimentTrainDF = sentimentTrainDF.drop(["LABEL"], axis=1)

#print(sentimentTrainDF)

sentimentModelNB = MultinomialNB()

sentimentModelNB.fit(sentimentTrainDF, trainSentimentLabels)

#print(sentimentModelNB.get_params(deep=True))

sentimentPrediction = sentimentModelNB.predict(sentimentTestDF)
#print(sentimentPrediction)

sentimentConfusionMatrix = confusion_matrix(testSentimentLabels, sentimentPrediction)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in sentimentConfusionMatrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in (sentimentConfusionMatrix/numpy.sum(sentimentConfusionMatrix)).flatten()]
labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(group_names, group_counts, group_percentages)]
labels = numpy.asarray(labels).reshape(2,2)
seaborn.heatmap(sentimentConfusionMatrix/numpy.sum(sentimentConfusionMatrix), annot=labels, fmt='', cmap='Blues')
