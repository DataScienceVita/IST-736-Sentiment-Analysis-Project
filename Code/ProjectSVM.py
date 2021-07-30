# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 19:24:44 2020

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
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

lastStatementFileName = "offenders_label_Remorse.csv"

sentimentDF = pandas.DataFrame()

allSentiments = []
allStatements = []

with open(lastStatementFileName, 'r', encoding='utf8', errors='ignore', newline='\n') as file:
    #file.encode('utf-8').strip()
    #print("Test")
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


#SVM
#LINEAR
linearSvmModel = LinearSVC(C=10, max_iter=100000)
linearSvmModel.fit(sentimentTrainDF, trainSentimentLabels)



linearSvmPredict = linearSvmModel.predict(sentimentTestDF)
print(linearSvmPredict)
linearSvmMatrix = confusion_matrix(testSentimentLabels, linearSvmPredict)
print(linearSvmMatrix)

#l_group_names = ['True Neg','False Pos','False Neg','True Pos']
#l_group_counts = ['{0:0.0f}'.format(value) for value in linearSvmMatrix.flatten()]
#l_group_percentages = ['{0:.2%}'.format(value) for value in (linearSvmMatrix/numpy.sum(linearSvmMatrix)).flatten()]
#l_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(l_group_names, l_group_counts, l_group_percentages)]
#l_labels = numpy.asarray(l_labels).reshape(2,2)

#seaborn.heatmap(linearSvmMatrix/numpy.sum(linearSvmMatrix), annot=l_labels, fmt='', cmap='Blues')

#POLY
polySvmModel = sklearn.svm.SVC(C=50, kernel='poly',degree=3, gamma='auto',verbose=True)
polySvmModel.fit(sentimentTrainDF, trainSentimentLabels)
polySvmPredict = polySvmModel.predict(sentimentTestDF)
polySvmMatrix = confusion_matrix(testSentimentLabels, polySvmPredict)
print(polySvmMatrix)

#p_group_names = ['True Neg','False Pos','False Neg','True Pos']
#p_group_counts = ['{0:0.0f}'.format(value) for value in polySvmMatrix.flatten()]
#p_group_percentages = ['{0:.2%}'.format(value) for value in (polySvmMatrix/numpy.sum(polySvmMatrix)).flatten()]
#p_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(p_group_names, p_group_counts, p_group_percentages)]
#p_labels = numpy.asarray(p_labels).reshape(2,2)

#seaborn.heatmap(polySvmMatrix/numpy.sum(polySvmMatrix), annot=p_labels, fmt='', cmap='Blues')

#RBF
rbfSvmModel = sklearn.svm.SVC(C=10000, kernel='rbf', verbose=True, gamma='auto')
rbfSvmModel.fit(sentimentTrainDF, trainSentimentLabels)
rbfSvmPredict = rbfSvmModel.predict(sentimentTestDF)
rbfSvmMatrix = confusion_matrix(testSentimentLabels, rbfSvmPredict)
print(rbfSvmMatrix)

r_group_names = ['True Neg','False Pos','False Neg','True Pos']
r_group_counts = ['{0:0.0f}'.format(value) for value in rbfSvmMatrix.flatten()]
r_group_percentages = ['{0:.2%}'.format(value) for value in (rbfSvmMatrix/numpy.sum(rbfSvmMatrix)).flatten()]
r_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(r_group_names, r_group_counts, r_group_percentages)]
r_labels = numpy.asarray(r_labels).reshape(2,2)

seaborn.heatmap(rbfSvmMatrix/numpy.sum(rbfSvmMatrix), annot=r_labels, fmt='', cmap='Blues')
