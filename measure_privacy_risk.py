# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:34:27 2023

@author: SGoethals
"""


import matplotlib.ticker as mtick
import operator
import math
import random
import copy
import time
from statistics import mean
import seaborn as sns
import pandas as pd
from privacy_functions import *

from nice import NICE
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
#from nice.explainers import NICE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from mondrian_k_anonymization import mondrian_all, calculate_metrics_mondrian
# %% settings
k = 10
alfa = 20

max_iterations = 3


#%% measure privacy risks

def measure_privacy_risk(dataset,data,qid):
    counts=data[qid].value_counts()
    # Filter out the value combinations with counts greater than or equal to 10
    ten_counts = counts[counts < 10]
    one_counts= counts[counts <=1]
    # Sum up the counts of the remaining value combinations
    privacy_risk = ten_counts.sum()/len(data)*100
    print('The privacy risk for dataset {} measured with k=10 is {} %'.format(dataset, privacy_risk))
    print('The percent of people uniquely identifiable in  {} is {} %'.format(dataset, one_counts.sum()/len(data)*100))
    #only on the training set
    data_train, data_test = train_test_split(data,  test_size=0.4,random_state=0)
    counts=data_train[qid].value_counts()
    # Filter out the value combinations with counts greater than or equal to 10
    ten_counts = counts[counts < 10]
    one_counts= counts[counts <=1]
    # Sum up the counts of the remaining value combinations
    privacy_risk = ten_counts.sum()/len(data_train)*100
    print('The privacy risk for training set {} measured with k=10 is {} %'.format(dataset, privacy_risk))
    print('The percent of people uniquely identifiable in training set {} is {} %'.format(dataset, one_counts.sum()/len(data_train)*100))
    print(' ')
    

dataset = 'heart_c'
heart_c = fetch_data(dataset)
qid = ['age', 'sex']
measure_privacy_risk(dataset, heart_c, qid)

headers = ['WifeAge', 'WifeEducation',
           'HusbandEducation', 'ChildrenBorn',
           'WifeReligion', 'WifeWorking',
           'HusbandOccupation', 'SOLIndex',
           'MediaExposure', 'ContraceptiveMethodUsed']

cmc = pd.read_csv("DATA/cmc.data",
                  header=None,
                  names=headers,
                  sep=',',
                  engine='python')

qid = ['WifeAge', 'ChildrenBorn']
measure_privacy_risk('cmc', cmc, qid)

german = fetch_data('german')
qid = ['Age', 'Personal-status', 'Foreign', 'Residence-time','Employment','Property','Housing','Job']
measure_privacy_risk('german',german, qid)

dataset = 'adult'
adult = fetch_data(dataset)
qid = ['age', 'marital-status', 'relationship', 'race', 'sex']
measure_privacy_risk('adult',adult, qid)

url = 'https://github.com/kaylode/k-anonymity/blob/main/data/informs/informs.csv'
url = 'https://raw.githubusercontent.com/kaylode/k-anonymity/main/data/informs/informs.csv'
informs = pd.read_csv(url, error_bad_lines=False, sep=';')
informs.dropna(inplace=True)
informs = informs.sample(n=5000, ignore_index=True)
qid = ['DOBMM', 'DOBYY', 'SEX', 'EDUCYEAR', 'marry','RACEX']
measure_privacy_risk('informs',informs, qid)




dataset='hospital'
hospital = pd.read_csv('DATA/hospital_discharge.csv', sep=';', header=[0], on_bad_lines='skip')
hospital=hospital.dropna()
hospital.reset_index(drop=True, inplace=True)
hospital['Length of Stay']=pd.to_numeric(hospital['Length of Stay'], errors ='coerce').fillna(120).astype('int')
hospital['Total Costs'] = hospital['Total Costs'].replace('[\$,]', '', regex=True).astype(float)
hospital['Total Charges'] = hospital['Total Charges'].replace('[\$,]', '', regex=True).astype(float)

hospital = hospital.sample(n=5000, ignore_index=True, random_state=0)
#X = hospital.drop(columns=['Health Service Area','Discharge Year', 'Hospital County','Operating Certificate Number', 'Facility Id', 'Facility Name', 'Attending Provider License Number','Operating Provider License Number', 'Other Provider License Number','CCS Diagnosis Code', 'CCS Procedure Code',  'APR DRG Code','APR MDC Code','APR Severity of Illness Code','Total Charges', 'Total Costs'])
X = hospital.drop(columns=['Health Service Area','Discharge Year', 'Hospital County','Operating Certificate Number', 'Facility Id', 'Facility Name', 'CCS Diagnosis Code', 'CCS Procedure Code',  'APR DRG Code','APR MDC Code','APR Severity of Illness Code','Total Charges', 'Total Costs'])
#X['Zip Code - 3 digits']=X['Zip Code - 3 digits'].astype(str)
y = hospital['Total Costs'] > mean(hospital['Total Costs'])
qid = ['Age Group', 'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity']
measure_privacy_risk('hospital',hospital, qid)
