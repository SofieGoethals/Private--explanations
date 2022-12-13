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

NCP_mondrian={}
pureness_mondrian={}
ex_times_mondrian={}
sel_frame_mondrian={}
NCP_grasp={}
pureness_grasp={}
ex_times_grasp={}
gen_list_grasp={}
discr_list_grasp={}

def load_data(dataset):
    filename_mondrian= 'Results/mondrian_results/' + dataset
    filename_grasp= 'Results/grasp_results_time/' + dataset
    infile = open(filename_mondrian,'rb')
    results= pickle.load(infile)
    infile.close()
    NCP_mondrian[dataset],pureness_mondrian[dataset],ex_times_mondrian[dataset], sel_frame_mondrian[dataset]=results
    infile = open(filename_grasp,'rb')
    results= pickle.load(infile)
    infile.close()
    NCP_grasp[dataset],pureness_grasp[dataset],ex_times_grasp[dataset],gen_list_grasp[dataset], discr_list_grasp[dataset]=results


load_data('heart_c')
load_data('cmc')
load_data('german')
load_data('adult')
load_data('informs')
load_data('hospital')





#%% calculate metrics
from statistics import median,mean
print('Calculating metrics')
for key in NCP_grasp.keys():
    print(key)
    print(mean(NCP_grasp[key]))
    print(mean(pureness_grasp[key]))
    print(mean(ex_times_grasp[key]))
    print(mean(ex_times_mondrian[key]))
    print('median')
    print(median(NCP_grasp[key]))
    print(median(pureness_grasp[key]))
    print(median(ex_times_grasp[key]))
    print(median(ex_times_mondrian[key]))
    print(' ')

#%% comparison with mondrian
print('Comparison with Mondrian')
#unsorted NCP
for key in NCP_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(NCP_mondrian[key],'b-', label='dataset',linewidth=3)
    plt.plot(NCP_grasp[key],'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('NCP', fontsize=50)
    plt.xlabel('Instance of test set',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  

    plt.show()
    
#sorted NCP
for key in NCP_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(sorted(NCP_mondrian[key]),'b-', label='dataset',linewidth=3)
    plt.plot(sorted(NCP_grasp[key]),'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('NCP', fontsize=50)
    plt.xlabel('Instances (sorted)',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  
    filename='Figures/Mondrian/'+key +'_NCP'
    plt.savefig(filename, bbox_inches='tight')  
    plt.show()
    
#sorted pureness    
for key in pureness_grasp.keys():
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size'] = '40'
    plt.plot(sorted(pureness_mondrian[key]),'b-', label='dataset',linewidth=3)
    plt.plot(sorted(pureness_grasp[key]),'--', color='lightsteelblue', linewidth=4, label='explanations')
    plt.ylabel('Pureness', fontsize=50)
    plt.xlabel('Instances (sorted)',fontsize=50)
    plt.legend(fontsize=45)
    plt.title(key,fontsize=60)  
    filename='Figures/Mondrian/'+key +'_pureness'
    plt.savefig(filename, bbox_inches='tight')  
    plt.show()

#%% fairness results
# watch out that majority/minority group is correct
plt.rcParams['font.size'] = '18'

discr_attr, discr_attr_values,discr_list={},{},{}

discr_list['german']=pd.cut(x=discr_list_grasp['german'],bins=[0,0.99,3], labels=['F','M'], include_lowest=True).tolist()
discr_list['informs']=pd.cut(x=discr_list_grasp['informs'],bins=[0,1,6], labels=['White','Non-White']).tolist()
discr_list['heart_c']=pd.cut(x=discr_list_grasp['heart_c'],bins=[0,0.99,1], labels=['F','M'], include_lowest=True).tolist()
discr_list['adult']=pd.cut(x=discr_list_grasp['adult'],bins=[0,0.99,1], labels=['F','M'], include_lowest=True).tolist()
discr_list['cmc']=pd.cut(x=discr_list_grasp['cmc'],bins=[0,0.99,1], labels=['Non-Islam','Islam'], include_lowest=True).tolist()
discr_list['hospital']=discr_list_grasp['hospital']

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
    for val in vals[count_ind]: #majority grup first
        print(val)
        a.append(val)
        indices = [i for i, x in enumerate(discr_list[key]) if x == val]
        res_list = list(itemgetter(*indices)(NCP_grasp[key]))
        print(mean(res_list))
        b.append(mean(res_list))
        res_list2=list(itemgetter(*indices)(pureness_grasp[key]))
        print(mean(res_list2))
        c.append(mean(res_list))
    return a,b

maj_patch = mpatches.Patch(color='royalblue', label='Majority group')
min_patch = mpatches.Patch(color='lightsteelblue', label='Minority group')
mid_patch= mpatches.Patch(color='cornflowerblue', label='Middle group')

for key in NCP_grasp.keys():
    print(key)
    a,b=calc_unfairness(key) 
    plt.bar(a, b, color=('royalblue','lightsteelblue'))
    plt.xlabel(discr_attr[key])
    plt.ylabel('NCP')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xticks([0, 1], discr_attr_values[key], fontsize=15, rotation=0)
    plt.legend(handles=[maj_patch,min_patch], loc = 'lower left',fontsize=15)
    filename='Figures/Fairness/'+key
    plt.savefig(filename, bbox_inches="tight") 
    plt.show()


#%% analyze the CF

