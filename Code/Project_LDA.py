#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:18:45 2020

@author: josephamico
"""

import nltk
import pandas as pd
import sklearn
import re  
import numpy as np
import os
import random as rd
from datetime import datetime 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, TreebankWordTokenizer

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot

# =============================================================================
# import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
# =============================================================================

import pyLDAvis.sklearn as LDAvis
import pyLDAvis
#import pyLDAvis.gensim 

## Set working directory 
os.chdir('/Users/josephamico/OneDrive - Syracuse University/Semester 8_Summer 2020/IST 736 - Text Mining/Project')

###############
##Import Data##
###############

# =============================================================================
# ## Set filename for easy data
# file_name = 'Tarleton_Last_Statements.csv'
# 
# ## Import into Dataframe, Remove nan values from last_statement, export csv
# tmp = pd.read_csv(file_name)
# tmp.dropna(subset = ['Last_Statement'], inplace = True)
# tmp['Last_Statement'] = tmp['Last_Statement'].str.lower()
# tmp['Last_Statement'] = tmp['Last_Statement'].str.replace(r'[^\w\s]', '')
# tmp = tmp[~tmp.Last_Statement.str.contains('decline')]
# tmp = tmp[~tmp.Last_Statement.str.contains('none')]
# tmp = tmp[~tmp.Last_Statement.str.contains('no statement')]
# tmp = tmp[~tmp.Last_Statement.str.contains('no record')]
# tmp = tmp[~tmp.Last_Statement.str.contains('made no final statement')]
# tmp = tmp[~tmp.Last_Statement.str.contains('written')]
# tmp.drop(['#','Name','Crime_Date', 'Execution_Date', 'Source'], axis = 1, inplace = True)
# tmp['State']=tmp['State'].str.strip()
# 
# ## Import region data from census excel file
# statedata = pd.read_excel('state-geocodes-v2019.xlsx')
# statedata = statedata.iloc[4:69]
# statedata.columns = statedata.iloc[0]
# statedata.drop(4, axis = 0, inplace = True)
# statedata['State (FIPS)'] = statedata['State (FIPS)'].apply(pd.to_numeric) 
# statedata = statedata.loc[statedata['State (FIPS)'] != 0]
# statedata = statedata.reset_index(drop = True)
# statedata.drop(['State (FIPS)'], axis = 1, inplace = True)
# statedata['Region'] = statedata['Region'].apply(pd.to_numeric)
# statedata['Division'] = statedata['Division'].apply(pd.to_numeric)
# 
# ## Merge dataframes
# new_tmp = tmp.merge(statedata, left_on = 'State', right_on = 'Name')
# new_tmp.drop('Name', axis = 1, inplace=True)
# new_tmp = new_tmp[['Gender', 'Plea', 'Statement Type', 'Execution Type'
#                    , 'State', 'Region','Division', 'Last_Statement']]
# ## Export to CSV
# new_tmp.to_csv('Tarleton_Last_Statements_Reduced.csv')
# =============================================================================



## Set filename to new csv
file_name1 = 'Tarleton_Last_Statements_Reduced.csv'

## Create empty lists
StatementRowCount = []
RegionList = []
DivisionList = []
GenderList = []
PleaList = []
StatementTypeList = []
StateList = []
LastStatementList = []
with open(file_name1,'r') as FILE:
    FILE.readline() # skip the first line to remove the header 
    count = 0 #set varible to 0 to count number of rows
    for row in FILE:   #starts on row 2
        #print(type(row))
        #print(row)
        row = row.lower()
        index, gender, plea, statement_type, execution_type, state, region, division, last_statement = row.split(',', 8)
        #print(name)
        #print(gender)
        #print(last_statement)
        count += 1 #add one to the count variable
        StatementRowCount.append(count)
        gender = gender.strip()
        GenderList.append(gender)
        plea = plea.strip()
        PleaList.append(plea)
        statement_type = statement_type.strip()
        StatementTypeList.append(statement_type)
        state = state.strip()
        StateList.append(state)
        region = region.strip()
        RegionList.append(region)
        division = division.strip()
        DivisionList.append(division)
        LastStatementList.append(last_statement)
    print('Total Rows:', StatementRowCount[-1])

np.unique(GenderList, return_counts = True)
region_number, region_count = np.unique(RegionList, return_counts=True)
np.unique(DivisionList, return_counts=True)
np.unique(StateList, return_counts=True)

region_DF = pd.DataFrame(zip(region_number, region_count), columns = ['Region', 'Count'])
region_DF.plot(x='Region', y='Count', kind='bar',  title = 'Number of Executions per Region')

## Create region data
RegionRowCount = []
Region2_Statements = []
Region3_Statements = []
Region4_Statements = []
with open(file_name1,'r') as FILE:
    FILE.readline() # skip the first line to remove the header 
    count = 0 #set varible to 0 to count number of rows
    for row in FILE:   #starts on row 2
        row = row.lower()
        index, gender, plea, statement_type, execution_type, state, region, division, last_statement = row.split(',', 8)
        if region == '2':
            Region2_Statements.append(last_statement)
        elif region == '3':
            Region3_Statements.append(last_statement)
        elif region == '4':
            Region4_Statements.append(last_statement)
        count += 1
        RegionRowCount.append(count)
    print('Total Rows:', RegionRowCount[-1])
    print('Region 2 Total Statements:', len(Region2_Statements))
    print('Region 3 Total Statements:', len(Region3_Statements))    
    print('Region 4 Total Statements:', len(Region4_Statements))

## Create division data
DivisionRowCount = []
Division3_Statements = []
Division4_Statements = []
Division5_Statements = []
Division6_Statements = []
Division7_Statements = []
Division8_Statements = []
Division9_Statements = []
with open(file_name1,'r') as FILE:
    FILE.readline() # skip the first line to remove the header 
    count = 0 #set varible to 0 to count number of rows
    for row in FILE:   #starts on row 2
        row = row.lower()
        index, gender, plea, statement_type, execution_type, state, region, division, last_statement = row.split(',', 8)
        if division == '3':
            Division3_Statements.append(last_statement)
        elif division == '4':
            Division4_Statements.append(last_statement)
        elif division == '5':
            Division5_Statements.append(last_statement)
        elif division == '6':
            Division6_Statements.append(last_statement)            
        elif division == '7':
            Division7_Statements.append(last_statement)
        elif division == '8':
            Division8_Statements.append(last_statement)
        elif division == '9':
            Division9_Statements.append(last_statement)
        count += 1
        DivisionRowCount.append(count)
    print('Total Rows:', DivisionRowCount[-1])
    print('Division 3 Total Statements:', len(Division3_Statements))
    print('Division 4 Total Statements:', len(Division4_Statements))
    print('Division 5 Total Statements:', len(Division5_Statements))
    print('Division 6 Total Statements:', len(Division6_Statements))
    print('Division 7 Total Statements:', len(Division7_Statements))
    print('Division 8 Total Statements:', len(Division8_Statements))
    print('Division 9 Total Statements:', len(Division9_Statements))    

########################
##Tokenize & Vectorize##
########################

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

STEMMER=PorterStemmer()
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words
    
MyCV1 = CountVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        #, tokenizer = MY_STEMMER
                        )

MyCV2 = CountVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        , max_df = 0.90
                        , min_df = 4)

MyCV1_TF = TfidfVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text)

MyCV2_TF = TfidfVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        , max_df = 0.95
                        , min_df = 2)

## CV1 ##
Last_DTM = MyCV1.fit_transform(LastStatementList)
Last_ColNames = MyCV1.get_feature_names()
print(Last_ColNames[:100])

## Import into Dataframe
Last_DF = pd.DataFrame(Last_DTM.toarray(), columns = Last_ColNames)
print(Last_DF.head())

## CV2 ##
Last_DTM_2 = MyCV2.fit_transform(LastStatementList)
Last_ColNames_2 = MyCV2.get_feature_names()
print(Last_ColNames_2[:100])

## Import into Dataframe
Last_DF_2 = pd.DataFrame(Last_DTM_2.toarray(), columns = Last_ColNames_2)
print(Last_DF_2.head())

## CV1_TF ##
Last_DTM_3 = MyCV1_TF.fit_transform(LastStatementList)
Last_ColNames_3 = MyCV1_TF.get_feature_names()
print(Last_ColNames_3[:100])

## Import into Dataframe
Last_DF_3 = pd.DataFrame(Last_DTM_3.toarray(), columns = Last_ColNames_3)
print(Last_DF_3.head())

## CV2_TF ##
Last_DTM_4 = MyCV2_TF.fit_transform(LastStatementList)
Last_ColNames_4 = MyCV2_TF.get_feature_names()
print(Last_ColNames_4[:100])

## Import into Dataframe
Last_DF_4 = pd.DataFrame(Last_DTM_4.toarray(), columns = Last_ColNames_4)
print(Last_DF_4.head())

# =============================================================================
# ## Remove numeric columns from dataframe
# for col in Last_DF:
#     if(re.search(r'[^A-Za-z]+', col)):
#         print(col)
#         Last_DF.drop([col], axis=1, inplace = True)
#         
# print(Last_DF.head())
# =============================================================================

################
##LDA Modeling##
################
## Model start time
t3 = datetime.now().time()
## Number of topics
num_topics = 4
## MyCV1 Model ##
lda_model_1 = LatentDirichletAllocation(n_components=num_topics
                                        , max_iter=1000
                                        , learning_method='online'
                                        , verbose = 1
                                        #, random_state = 10
                                        )
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
t0 = datetime.now().time()
LDA_Model_1 = lda_model_1.fit_transform(Last_DF)
t1 = datetime.now().time()
print('Time Started:', t0)
print('Time Finished:', t1)

# =============================================================================
# ## Interactive visualization
# panel = LDAvis.prepare(lda_model_1, Last_DTM, MyCV1, mds='tsne')
# pyLDAvis.show(panel)
# =============================================================================

# =============================================================================
# ## MyCV2 model ##
# lda_model_2 = LatentDirichletAllocation(n_components=num_topics
#                                         , max_iter=1000
#                                         , learning_method='online'
#                                         , verbose = 1
#                                         , random_state = 11)
# 
# t0 = datetime.now().time()
# LDA_Model_2 = lda_model_2.fit_transform(Last_DF_2)
# t1 = datetime.now().time()
# print('Time Started:', t0)
# print('Time Finished:', t1)
# =============================================================================

# =============================================================================
# panel = LDAvis.prepare(lda_model_2, Last_DTM_2, MyCV2, mds='tsne')
# pyLDAvis.show(panel)
# =============================================================================


# =============================================================================
# ## MyCV1_TF Model ##
# lda_model_3 = LatentDirichletAllocation(n_components=num_topics
#                                         , max_iter=1000
#                                         , learning_method='online'
#                                         , verbose = 1
#                                         , random_state = 12)
# t0 = datetime.now().time()
# LDA_Model_3 = lda_model_3.fit_transform(Last_DF_3)
# t1 = datetime.now().time()
# print('Time Started:', t0)
# print('Time Finished:', t1)
# 
# # =============================================================================
# # panel = LDAvis.prepare(lda_model_3, Last_DTM_3, MyCV1_TF, mds='tsne')
# # pyLDAvis.show(panel)
# # =============================================================================
# 
# ## MyCV2_TF Model ##
# lda_model_4 = LatentDirichletAllocation(n_components=num_topics
#                                         , max_iter=1000
#                                         , learning_method='online'
#                                         , verbose = 1
#                                         , random_state = 13)
# 
# t0 = datetime.now().time()
# LDA_Model_4 = lda_model_4.fit_transform(Last_DF_4)
# t1 = datetime.now().time()
# print('Time Started:', t0)
# print('Time Finished:', t1)
# 
# # =============================================================================
# # panel = LDAvis.prepare(lda_model_4, Last_DTM_4, MyCV2_TF, mds='tsne')
# # pyLDAvis.show(panel)
# # =============================================================================
# 
# =============================================================================
t4 = datetime.now().time()
print('Final Time Started:', t3)
print('Final Time Finished:', t4)

vocab = MyCV1.get_feature_names()
vocab_array = np.asarray(vocab)
word_topic = np.array(lda_model_1.components_)
word_topic = word_topic.transpose()

## Create defintion to print top 10 words for each topic
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)
      
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
                        ## gets top n elements in decreasing order
    

## Use print_topics to get top 10 words in each topic
print_topics(lda_model_1, MyCV1)


num_top_words = 10
fontsize_base = 11.5

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t+1))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()


#####################################################################################################
#################################### Regional Data Analysis #########################################
#####################################################################################################

########################
##Tokenize & Vectorize##
########################

Region2_CV = CountVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        #, tokenizer = MY_STEMMER
                        )

Region3_CV = CountVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        #, tokenizer = MY_STEMMER
                        )

Region4_CV = CountVectorizer(input='content' 
                        , strip_accents = 'unicode'
                        #, lowercase = True
                        , stop_words = 'english'
                        , preprocessor = preprocess_text
                        #, tokenizer = MY_STEMMER
                        )

## Region2_CV ##
Region2_DTM = Region2_CV.fit_transform(Region2_Statements)
Region2_ColNames = Region2_CV.get_feature_names()
print(Region2_ColNames[:100])

## Import into Dataframe
Region2_DF = pd.DataFrame(Region2_DTM.toarray(), columns = Region2_ColNames)
print(Region2_DF.head())

## Region3_CV ##
Region3_DTM = Region3_CV.fit_transform(Region3_Statements)
Region3_ColNames = Region3_CV.get_feature_names()
print(Region3_ColNames[:100])

## Import into Dataframe
Region3_DF = pd.DataFrame(Region3_DTM.toarray(), columns = Region3_ColNames)
print(Region3_DF.head())

## Region4_CV ##
Region4_DTM = Region4_CV.fit_transform(Region4_Statements)
Region4_ColNames = Region4_CV.get_feature_names()
print(Region4_ColNames[:100])

## Import into Dataframe
Region4_DF = pd.DataFrame(Region4_DTM.toarray(), columns = Region4_ColNames)
print(Region4_DF.head())

##################
## LDA Modeling ##
##################
t5 = datetime.now().time()

#####################
## Model parameter changes
num_topics = 2
max_iterations = 1000


## Region2_CV Model ##
lda_model_region2 = LatentDirichletAllocation(n_components=num_topics
                                        , max_iter=max_iterations
                                        , learning_method='online'
                                        , verbose = 1
                                        , random_state = 10)

LDA_Model_Region2 = lda_model_region2.fit_transform(Region2_DF)


# =============================================================================
# ## Interactive visualization
# panel = LDAvis.prepare(lda_model_region2, Region2_DTM, Region2_CV, mds='tsne')
# pyLDAvis.show(panel)
# =============================================================================

vocab = Region2_CV.get_feature_names()
vocab_array = np.asarray(vocab)
word_topic = np.array(lda_model_region2.components_)
word_topic = word_topic.transpose()
num_top_words = 10
fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t+1))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

## Model parameter changes
num_topics = 4
max_iterations = 1000

#################################
## Region3_CV Model ##
lda_model_region3 = LatentDirichletAllocation(n_components=num_topics
                                        , max_iter=max_iterations
                                        , learning_method='online'
                                        , verbose = 1
                                        , random_state = 10)

LDA_Model_Region3 = lda_model_region3.fit_transform(Region3_DF)


# =============================================================================
# ## Interactive visualization
# panel = LDAvis.prepare(lda_model_region3, Region3_DTM, Region3_CV, mds='tsne')
# pyLDAvis.show(panel)
# =============================================================================

vocab = Region3_CV.get_feature_names()
vocab_array = np.asarray(vocab)
word_topic = np.array(lda_model_region3.components_)
word_topic = word_topic.transpose()
num_top_words = 10
fontsize_base = 12

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t+1))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

###################################
num_topics = 2
max_iterations = 1000

## Region4_CV Model ##
lda_model_region4 = LatentDirichletAllocation(n_components=num_topics
                                        , max_iter=max_iterations
                                        , learning_method='online'
                                        , verbose = 1
                                        , random_state = 10)

LDA_Model_Region4 = lda_model_region4.fit_transform(Region4_DF)


# =============================================================================
# ## Interactive visualization
# panel = LDAvis.prepare(lda_model_region4, Region4_DTM, Region4_CV, mds='tsne')
# pyLDAvis.show(panel)
# =============================================================================

vocab = Region4_CV.get_feature_names()
vocab_array = np.asarray(vocab)
word_topic = np.array(lda_model_region4.components_)
word_topic = word_topic.transpose()
num_top_words = 10
fontsize_base = 15

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t+1))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

t6 = datetime.now().time()
print('Region Time Started:', t5)
print('Region Time Finished:', t6)
