# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:01:41 2022

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
import pickle, joblib

from mondrian_k_anonymization import mondrian_all, calculate_metrics_mondrian
# %% settings
k = 10
alfa = 20

max_iterations = 3


#%%
dataset = 'heart_c'
heart_c = fetch_data(dataset)
X = heart_c.drop(columns=['target'])
y = heart_c.loc[:, 'target']
feature_names = list(X.columns)
cat_feat = [1, 2, 5, 6, 8, 10, 11, 12]
num_feat = [0, 3, 4, 7, 9]
qid = ['age', 'sex']
target_outcome = 1
discr_attr = 'sex'

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)


#Save results modeling
filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')

#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 



#%%
dataset='cmc'
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
cmc.head()
X = cmc.drop(columns=['ContraceptiveMethodUsed'])
y = cmc['ContraceptiveMethodUsed'] == 1
num_feat = [0, 3]
cat_feat = [1, 2, 4, 5, 6, 7, 8]
target_outcome = True  # geen anticonceptie (genant)
feature_names = list(X.columns)
qid = ['WifeAge', 'ChildrenBorn']
discr_attr = 'WifeReligion'

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)


#Save results modeling

filename=filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')

#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 




#%%german
dataset='german'
german = fetch_data('german')
X = german.drop(columns=['target'])
y = german.loc[:, 'target']
feature_names = list(X.columns)

print(feature_names)
cat_feat = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
num_feat = [1, 4, 7, 10, 12, 15, 17]

qid = ['Age', 'Personal-status', 'Foreign', 'Residence-time','Employment','Property','Housing','Job']

target_outcome = 1

discr_attr = 'Personal-status'  # of Personal-status

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)

#Save results modeling

filename=filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')

#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 


# %% adult
dataset = 'adult'
adult = fetch_data(dataset)
X = adult.drop(columns=['education-num', 'fnlwgt', 'target'])
y = adult.loc[:, 'target']
feature_names = list(X.columns)
cat_feat = [1, 2, 3, 4, 5, 6, 7, 11]
num_feat = [0, 8, 9, 10]
qid = ['age', 'marital-status', 'relationship', 'race', 'sex']
target_outcome = 0
discr_attr = 'sex'

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]

X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)

#Save results modeling

filename=filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')

#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 



    #%%
    
dataset='informs'
url = 'https://github.com/kaylode/k-anonymity/blob/main/data/informs/informs.csv'
url = 'https://raw.githubusercontent.com/kaylode/k-anonymity/main/data/informs/informs.csv'
informs = pd.read_csv(url, error_bad_lines=False, sep=';')
informs.dropna(inplace=True)
random.seed(0)
informs = informs.sample(n=5000, ignore_index=True, random_state=0)
informs.head()

X = informs.drop(columns=['income', 'DUID', 'PID', 'ID','RACEAX','RACEBX','RACEWX','RACETHNX'])
y = informs['income'] > mean(informs['income'])

feature_names = list(X.columns)
print(feature_names)
cat_feat = [2, 3, 4, 5,8,9]
num_feat = [0, 1, 6,7]
qid = ['DOBMM', 'DOBYY', 'SEX', 'EDUCYEAR', 'marry','RACEX']
target_outcome = True
discr_attr = 'RACEX'

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)

trainingdata=X.iloc[X_train]
trainingset=trainingdata[qid]


#Save results modeling
filename=filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')


#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close()  

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 


# %%
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

feature_names = list(X.columns)
print(feature_names)
cat_feat = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
num_feat = [5,18]
qid = ['Age Group', 'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity']
target_outcome = True
discr_attr = 'Gender'

#modelling
num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
X_train, y_train, X_test,y_test,clf,NICE_fit=model(X,y,feature_names, cat_feat, num_feat, target_outcome)




#Save results modeling

filename=filename = 'Revision/modelling_results/'+dataset
joblib.dump(clf, filename  +  '_pipeline.pkl')


#GRASP
ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list=grasp_results(discr_attr,X,y,X_train,X_test,y_test,y_train,target_outcome,  k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit)

#Save results GRASP
filename = 'Revision/grasp_results_time/'+dataset
outfile = open(filename,'wb')  
grasp_dict={'NCP_list': NCP_list, 'pureness_list': pureness_list, 'ex_times':ex_times, 'gen_list': gen_list, 'discr_list': discr_list, 'instance_list': instance_list}      
pickle.dump(grasp_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 

#Mondrian
cat_indices=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
dfn=mondrian_all(X.iloc[X_train],qid, cat_indices, k, part = 's', aggr = 'r')
NCP_list_dataset,pureness_list_dataset,ex_times_dataset, sel_frame_dataset= calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn)

#SAVE RESULTS MONDRIAN
filename = 'Revision/mondrian_results/'+dataset
outfile = open(filename,'wb')        
mondr_dict={'NCP_list':NCP_list_dataset,'pureness_list':pureness_list_dataset,'ex_times':ex_times_dataset,'sel_frame':sel_frame_dataset,'dfn':dfn }
pickle.dump(mondr_dict,outfile)
#pickle.dump(b,outfile)
outfile.close() 