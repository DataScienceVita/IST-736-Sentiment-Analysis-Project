#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:05:19 2020

@author: josephamico
"""

from selenium import webdriver
import time
import pandas as pd

##--Web scrape the data from Tarleton website--##

## Call webdriver, establish options, provide path to Chrome driver, establish function
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless') # Prevents launching of a browser window
driver = webdriver.Chrome('/usr/local/bin/chromedriver', options = options) # Pass chromedriver path and options to webdriver
time.sleep(1)

## Access website and pull html info
url = 'https://www.tarleton.edu/scripts/deathrow/'
driver.get(url)
time.sleep(7) # Allow enough time for website to load 
button = driver.find_element_by_xpath('/html/body/div[1]/div[4]/div[1]/div/fieldset[1]/ul/li[1]/a') # xpath to View All Button
button.click() #Select, or 'click', View All button on website
time.sleep(5) # Allow enough time for website to load
page_source = driver.page_source #Download full HTML from current webpage 

## Parse html with pandas and close chrome session 
html_df_list = pd.read_html(page_source) # Parse HTML using pandas
driver.quit() #Close chrome session

################################
## Create and Clean Dataframe ##
################################

## Create dataframe
deathDF = html_df_list[0] # Pull needed dataframe out of dataframe list
deathDF.head()
deathDF.columns

##--Clean Information Column--##

## Find missing values
deathDF.Information.str.contains(r'(Gender:\sPlea:)').sum()
deathDF.Information.str.contains(r'(Plea:\sStatement Type:)').sum()
deathDF.Information.str.contains(r'(Statement Type:\sExecution Type:)').sum()
deathDF.Information.str.contains(r'(Execution Type:\sState:)').sum()
## Replace missing values with 'None'
deathDF['Information'] = deathDF.Information.str.replace(r'(Gender:\sPlea:)','Gender: None  Plea:')
deathDF['Information'] = deathDF.Information.str.replace(r'(Plea:\sStatement Type:)','Plea: None  Statement Type:')
deathDF['Information'] = deathDF.Information.str.replace(r'(Statement Type:\sExecution Type:)'
                                                         ,'Statement Type: None  Execution Type:')
deathDF['Information'] = deathDF.Information.str.replace(r'(Execution Type:\sState:)','Execution Type: None  State:')
## Remove column names from strings
deathDF['Information'] = deathDF.Information.str.replace(r'(Gender:)', '')
deathDF['Information'] = deathDF.Information.str.replace(r'(Plea:)', '')
deathDF['Information'] = deathDF.Information.str.replace(r'(Statement Type:)','')
deathDF['Information'] = deathDF.Information.str.replace(r'(Execution Type:)', '')
deathDF['Information'] = deathDF.Information.str.replace(r'(State:)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Lethal Injection)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Electrocution)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Firing Squad)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Gas Chamber)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Hanging)', '')
# deathDF['Information'] = deathDF.Information.str.replace(r'(Lethal Gas)', '')
# deathDF.Information.str.split('  ', expand = True)

## Split the column into seperate columns 
deathDF[['Name', 'Gender', 'Plea', 'Statement Type'
         , 'Execution Type', 'State']] = deathDF.Information.str.split('  ', expand = True)
## Check for null values
deathDF.isna().sum()
## Drop Information Column
deathDF = deathDF.drop('Information', axis = 1)
deathDF.columns

##--Clean Dates Column--##

## Find missing values 
deathDF.Dates.str.contains(r'(Crime:\sExecution:)').sum()
## Replace missing dates with None
deathDF['Dates'] = deathDF.Dates.str.replace(r'(Crime:\sExecution:)', 'Crime: None  Execution:')
deathDF['Dates'] = deathDF.Dates.str.replace(r'(Crime:)', '')
deathDF['Dates'] = deathDF.Dates.str.replace(r'(Execution:)', '')
## Split column into seperate columns 
deathDF[['Crime_Date', 'Execution_Date']] = deathDF.Dates.str.split('  ', expand = True)
## Check for null values 
deathDF.isna().sum()
## Drop Dates column 
deathDF = deathDF.drop('Dates', axis = 1)
## Check columns 
deathDF.columns

##--Clean Last Statement Column--##
## Split column on source
deathDF[['Last_Statement', 'Source']] = deathDF['Last Statement'].str.split('Source:', expand = True)
## Drop Last Statement Column
deathDF.drop('Last Statement', axis = 1, inplace = True)
## Check columns 
deathDF.columns

## Export to CSV
outfile_path = '/Users/josephamico/OneDrive - Syracuse University/Semester 8_Summer 2020/IST 736 - Text Mining/Project'
deathDF.to_csv(outfile_path + '/Tarleton_Last_Statements.csv', index = False)








