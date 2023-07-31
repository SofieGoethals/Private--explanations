# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:50:19 2022

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
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

#%% loading data


results_mondrian, results_grasp={},{}

def load_data(dataset):
    filename_mondrian= 'Revision/mondrian_results/' + dataset
    filename_grasp= 'Revision/grasp_results_time/' + dataset
    infile = open(filename_mondrian,'rb')
    results_mondrian[dataset]= pickle.load(infile)
    infile.close()
    infile = open(filename_grasp,'rb')
    results_grasp[dataset]= pickle.load(infile)
    infile.close()


load_data('heart_c')
load_data('cmc')
load_data('german')
load_data('adult')
load_data('informs')
load_data('hospital')

#%% load the extra metrics
# Load the excel file
df = pd.read_excel('Revision/extra_metrics.xlsx',sheet_name='Mondrian')
# Convert each column to a dictionary of dictionaries
extra_metrics_mondrian={}
for column in df.columns[1:]:
    column_data = {}
    for i, row in df.iterrows():
        #if i == 0:
            #continue
        column_data[row[0]] = row[column]
    extra_metrics_mondrian[column] = column_data
    
# Load the excel file
df = pd.read_excel('Revision/extra_metrics.xlsx',sheet_name='Grasp')
# Convert each column to a dictionary of dictionaries
extra_metrics_grasp={}
for column in df.columns[1:]:
    column_data = {}
    for i, row in df.iterrows():
        #if i == 0:
            #continue
        column_data[row[0]] = row[column]
    extra_metrics_grasp[column] = column_data




#%% calculate metrics
from statistics import median,mean
print('Calculating metrics')
for key in results_grasp.keys():
    print(key)
    print(mean(results_grasp[key]['NCP_list']))
    print(mean(results_grasp[key]['pureness_list']))
    #a=[median(results_grasp[key]['ex_times']) if x>200 else x for x in results_grasp[key]['ex_times']] #outliers for when computer went to sleep
    #print(mean(a))
    print(median(results_grasp[key]['ex_times'])) ## more robust
    print(extra_metrics_grasp[key]['dm_metric'])
    print('DM metric / amount of anonymized explanations: {}'.format(extra_metrics_grasp[key]['dm_metric']/len(results_grasp[key]['NCP_list'])))
    print(extra_metrics_grasp[key]['cm_metric'])
    print('MONDRIAN')
    print(mean(results_mondrian[key]['NCP_list']))
    print(mean(results_mondrian[key]['pureness_list']))
    print(median(results_mondrian[key]['ex_times'])) ## to be fixed
    print(extra_metrics_mondrian[key]['dm_metric'])
    print('DM metric / amount of anonymized explanations: {}'.format(extra_metrics_mondrian[key]['dm_metric']/len(results_mondrian[key]['NCP_list'])))
    print(extra_metrics_mondrian[key]['cm_metric'])
    print(' ')

#%% comparison with mondrian
print('Comparison with Mondrian')
#unsorted NCP
for key in results_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(results_mondrian[key]['NCP_list'],'b-', label='dataset',linewidth=3)
    plt.plot(results_grasp[key]['NCP_list'],'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('NCP', fontsize=50)
    plt.xlabel('Instance of test set',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  

    plt.show()
    
#sorted NCP
for key in results_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(sorted(results_mondrian[key]['NCP_list']),'b-', label='dataset',linewidth=3)
    plt.plot(sorted(results_grasp[key]['NCP_list']),'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('NCP', fontsize=50)
    plt.xlabel('Instances (sorted)',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  
    filename='Figures/Mondrian/'+key +'_NCP'
    #plt.savefig(filename, bbox_inches='tight')  
    plt.show()
    
#sorted pureness    
for key in results_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(sorted(results_mondrian[key]['pureness_list']),'b-', label='dataset',linewidth=3)
    plt.plot(sorted(results_grasp[key]['pureness_list']),'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('Pureness', fontsize=50)
    plt.xlabel('Instances (sorted)',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  
    filename='Figures/Mondrian/'+key +'_pureness'
    #plt.savefig(filename, bbox_inches='tight')  
    plt.show()

#%% fairness results
# watch out that majority/minority group is correct
plt.rcParams['font.size'] = '18'

discr_attr, discr_attr_values,discr_list={},{},{}

discr_list['german']=pd.cut(x=results_grasp['german']['discr_list'],bins=[0,0.99,3], labels=['F','M'], include_lowest=True).tolist()
discr_list['informs']=pd.cut(x=results_grasp['informs']['discr_list'],bins=[0,1,6], labels=['White','Non-White']).tolist()
discr_list['heart_c']=pd.cut(x=results_grasp['heart_c']['discr_list'],bins=[0,0.99,1], labels=['F','M'], include_lowest=True).tolist()
discr_list['adult']=pd.cut(x=results_grasp['adult']['discr_list'],bins=[0,0.99,1], labels=['F','M'], include_lowest=True).tolist()
discr_list['cmc']=pd.cut(x=results_grasp['cmc']['discr_list'],bins=[0,0.99,1], labels=['Non-Islam','Islam'], include_lowest=True).tolist()
discr_list['hospital']=results_grasp['hospital']['discr_list']

discr_attr['heart_c']='Sex'
discr_attr['cmc']='WifeReligion'
discr_attr['german']= 'Sex'
discr_attr['adult']= 'Sex'
discr_attr['informs']= 'Race'
discr_attr['hospital']= 'Sex'

discr_attr_values['heart_c']=['Male', 'Female'] #Male is majority group (1)
discr_attr_values['cmc']=['Islam', 'Non-Islam'] #Islam is majority group(1)
discr_attr_values['german']= ['Male', 'Female'] #Male is majority group (1,2,3)
discr_attr_values['adult']= ['Male', 'Female'] #Male is majority group (1)
discr_attr_values['informs']= ['White', 'Non-white'] #1 is white (majority group), 2-5 are non-white (minority group)
discr_attr_values['hospital']= ['Female', 'Male'] #Female is majority group here F-M

from operator import itemgetter
def calc_unfairness(key):
    a = []
    b = []
    c=[]
    vals,counts=np.unique(discr_list[key], return_counts=True)
    count_ind=np.argsort(-counts)
    for val in vals[count_ind]: #majority group first
        print(val)
        a.append(val)
        indices = [i for i, x in enumerate(discr_list[key]) if x == val]
        res_list = list(itemgetter(*indices)(results_grasp[key]['NCP_list']))
        print(mean(res_list))
        b.append(mean(res_list))
        res_list2=list(itemgetter(*indices)(results_grasp[key]['pureness_list']))
        print(mean(res_list2))
        c.append(mean(res_list2))
    return a,b

maj_patch = mpatches.Patch(color='royalblue', label='Majority group')
min_patch = mpatches.Patch(color='lightsteelblue', label='Minority group')
mid_patch= mpatches.Patch(color='cornflowerblue', label='Middle group')

for key in results_grasp.keys():
    print(key)
    a,b=calc_unfairness(key) 
    plt.bar(a, b, color=('royalblue','lightsteelblue'))
    plt.xlabel(discr_attr[key])
    plt.ylabel('NCP')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xticks([0, 1], discr_attr_values[key], fontsize=15, rotation=0)
    plt.legend(handles=[maj_patch,min_patch], loc = 'lower left',fontsize=15)
    filename='Revision/Figures/Fairness/'+key
    plt.title(key)
    plt.savefig(filename, bbox_inches="tight") 
    plt.show()


#%% analyze the CF

