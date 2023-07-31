# import libraries
import numpy as np
import math
import pandas as pd
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nice import NICE
from tqdm import tqdm
import random
from collections import Counter

#%%

# expect: script.py | dataset | k_level k | strict/relaxed partitioning s|r | range/mean aggregation r|m
# arguments = sys.argv
# if len(arguments) != 5:
#     print('Invalid number of arguments. Expected 5 arguments.')
#     sys.exit(1)

# print('Reading dataset...')
# # load the dataset and apply column names
# try:
#     df = pd.read_csv('{}'.format(arguments[1]), sep=",", header=None, index_col=False, engine='python')
# except:
#     print('Loading failed. Exiting...')
# 	

# # remove NaNs
# df.dropna(inplace=True)
# df.reset_index(inplace=True)
# df = df.iloc[:,1:]

# # infer data types
# types = list(df.dtypes)
# cat_indices = [i for i in range(len(types)) if types[i]=="object"]

# # convert df to numpy array
# df = np.array(df)

# function to compute the span of a given column while restricted to a subset of rows (a data partition)
def colSpans(df,q, cat_indices, partition):
    spans = dict()
    #types = list(df.dtypes) 
    for column in range(q):
    		dfp = df[partition,column] # restrict df to the current column
    		if column in cat_indices:
     			span = len(np.unique(dfp)) # span of categorical variables is its number of unique classes
    		else:
     			span = np.max(dfp)-np.min(dfp) # span of numerical variables is its range
    		spans[column] = span
    return spans
 	

# function to split rows of a partition based on median value (categorical vs. numerical attributes)
def splitVal(df, dim, part, cat_indices, mode):
    dfp = df[part,dim] # restrict whole dataset to a single attribute and rows in this partition
    unique = list(np.unique(dfp))
    length = len(unique)
    if dim in cat_indices: # for categorical variables
        if mode=='strict': # i do not mind about |lhs| and |rhs| being equal
            lhv = unique[:length//2]
            rhv = unique[length//2:]
            lhs_v = list(list(np.where(np.isin(dfp,lhv)))[0]) # left partition
            rhs_v = list(list(np.where(np.isin(dfp,rhv)))[0]) # right partition
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
        elif mode=='relaxed': # i want |lhs| = |rhs| +-1
            lhv = unique[:length//2]
            rhv = unique[length//2:]
            lhs_v = list(list(np.where(np.isin(dfp,lhv)))[0]) # left partition
            rhs_v = list(list(np.where(np.isin(dfp,rhv)))[0]) # right partition
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
            diff = len(lhs)-len(rhs)
            if diff==0:
                pass
            elif diff<0:
                lhs1 = rhs[:(np.abs(diff)//2)] # move first |diff|/2 indices from rhs to lhs
                rhs = rhs[(np.abs(diff)//2):] 
                lhs = np.concatenate((lhs,lhs1))
            else:
                rhs1 = lhs[-(diff//2):]
                lhs = lhs[:-(diff//2)]
                rhs = np.concatenate((rhs,rhs1))
        else:
            lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
    else: # for numerical variables, split based on median value (strict or relaxed)
        median = np.median(dfp)
        if mode=='strict': # strict partitioning (do not equally split indices of median values)
            lhs_v = list(list(np.where(dfp < median))[0])
            rhs_v = list(list(np.where(dfp >= median))[0])
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
        elif mode=='relaxed': # exact median values are equally split between the two halves
            lhs_v = list(list(np.where(dfp < median))[0])
            rhs_v = list(list(np.where(dfp > median))[0])
            median_v = list(list(np.where(dfp == median))[0])
            lhs_p = [part[i] for i in lhs_v]
            rhs_p = [part[i] for i in rhs_v]
            median_p = [part[i] for i in median_v]
            diff = len(lhs_p)-len(rhs_p) # i need to have |lhs| = |rhs| +- 1
            if diff<0:
                med_lhs = np.random.choice(median_p, size=np.abs(diff), replace=False) # first even up |lhs_p| and |rhs_p|
                med_to_split = [i for i in median_p if i not in med_lhs] # prepare remaining indices for equal split
                lhs_p = np.concatenate((lhs_p,med_lhs))
            else: # same but |rhs_p| needs to be levelled up to |lhs_p|
                med_rhs = np.random.choice(median_p, size=np.abs(diff), replace=False)
                med_to_split = [i for i in median_p if i not in med_rhs]
                rhs_p = np.concatenate((rhs_p,med_rhs))
            med_lhs_1 = np.random.choice(med_to_split, size=(len(med_to_split)//2), replace=False) # split remaining median indices equally between lhs and rhs
            med_rhs_1 = [i for i in med_to_split if i not in med_lhs_1]
            lhs = np.concatenate((lhs_p,med_lhs_1))
            rhs = np.concatenate((rhs_p,med_rhs_1))
        else:
            lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
    return [int(x) for x in lhs], [int(x) for x in rhs]

# create k-anonymous equivalence classes
def partitioning(df,q, k, cat_indices, mode):

	final_partitions = []
	working_partitions = [[x for x in range(len(df))]] # start with full dataset
	
	while len(working_partitions) > 0: # while there is at least one working partition left
	
		partition = working_partitions[0] # take the first in the list
		working_partitions = working_partitions[1:] # remove it from list of working partitions
		
		if len(partition) < 2*k: # if it is not at least 2k long, i.e. if i cannot get any new acceptable partition pair, at least k-long each
			final_partitions.append(partition) # append it to final set of partitions
			# and skip to the next partition
		else:
			spans = colSpans(df,q, cat_indices, partition) # else, get spans of the feature columns restricted to this partition
			ordered_span_cols = sorted(spans.items(), key=lambda x:x[1], reverse=True) # sort col indices in descending order based on their span
			for dim, _ in ordered_span_cols: # select the largest first, then second largest, ...
				lhs, rhs = splitVal(df, dim, partition, cat_indices, mode) # try to split this partition
				if len(lhs) >= k and len(rhs) >= k: # if new partitions are not too small (<k items), this partitioning is okay
					working_partitions.append(lhs) 
					working_partitions.append(rhs) # re-append both new partitions to set of working partitions for further partitioning
					break # break for loop and go to next partition, if available          
			else: # if no column could provide an allowable partitioning
				final_partitions.append(partition) # add the whole partition to the list of final partitions
		
	return final_partitions

# print('Setting up partitioning...')

# # build k-anonymous equivalence classes
# k = int(arguments[2])
# if k > len(df):
#    print('Invalid input. k must not exceed dataset size. Setting k to default 10.')
#    k = 10
   
# modeArg = str(arguments[3])
# if modeArg not in ['s','r']:
#    print("Invalid input. Partitioning mode must be 'r' for relaxed or 's' for strict.")
#    print("Setting relaxed mode as default.")
# mode = 'relaxed'
# if modeArg == 's': mode = 'strict'
 
# equivalence_classes = partitioning(df, k, cat_indices, mode)
# sizes = []
# for part in equivalence_classes:
#    sizes.append(len(part))
# min_size = np.min(sizes)
# print('Partitioning completed.')
# print('{} equivalence classes were created. Minimum size is {}.'.format(len(equivalence_classes),min_size))

# generate the anonymised dataset
def anonymize_df(df,q, partitions, cat_indices, mode='range'):
  
    anon_df = []
    categorical = cat_indices
    #types = list(df.dtypes)
    for ip,p in enumerate(partitions):
        aggregate_values_for_partition = []
        partition = df[p]
        for column in range(q):
            if column in categorical:
                values = list(np.unique(partition[:,column]))
                #aggregate_values_for_partition.append(','.join(values))
                aggregate_values_for_partition.append(','.join(str(v) for v in values))
            else:
                if mode=='mean':
                    aggregate_values_for_partition.append(np.mean(partition[:,column]))
                else:
                    col_min = np.min(partition[:,column])
                    col_max = np.max(partition[:,column])
                    if col_min == col_max:
                        aggregate_values_for_partition.append(col_min)
                    else:
                        aggregate_values_for_partition.append('{}-{}'.format(col_min,col_max))
        for i in range(len(p)):
            anon_df.append([int(p[i])]+aggregate_values_for_partition)
  
    df_anon = pd.DataFrame(anon_df)
    dfn1 = df_anon.sort_values(df_anon.columns[0])
    dfn1 = dfn1.iloc[:,1:]
    return np.array(dfn1)

#print('Setting up anonymisation...')

# anonymise dataset
# aggregationArg = str(arguments[4])
# if aggregationArg not in ['m','r']:
#    print("Invalid input. Aggregation metrics must either be 'r' for range or 'm' for mean.")
#    print("Setting range metrics as default.")
# aggregation = 'range'
# if aggregationArg == 'm': aggregation = 'mean'
 
# dfn = anonymize_df(df, equivalence_classes, cat_indices, aggregation)
# np.savetxt('anon_df.csv', dfn, fmt='%s', delimiter=';')
# print('Anonymization completed.')
#sys.exit(1)


#%%

def mondrian_all(df,qid, cat_indices, k, part = 's', aggr = 'range'):
    #or df is only with qid or extra step to select the qid
    df = df.loc[:,qid]
    # remove NaNs
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.iloc[:,1:]
    # convert df to numpy array
    df = np.array(df)
    q=len(qid)
    print('Setting up partitioning...')
    if part not in ['s','r']:
       print("Invalid input. Partitioning mode must be 'r' for relaxed or 's' for strict.")
       print("Setting relaxed mode as default.")
    mode = 'strict'
    if part == 's': mode = 'strict'
    equivalence_classes = partitioning(df,q, k, cat_indices, mode)
    sizes = []
    for part in equivalence_classes:
       sizes.append(len(part))
    min_size = np.min(sizes)
    print('Partitioning completed.')
    print('{} equivalence classes were created. Minimum size is {}.'.format(len(equivalence_classes),min_size))
    print('Setting up anonymisation...')
    # anonymise dataset
    if aggr not in ['m','r']:
       print("Invalid input. Aggregation metrics must either be 'r' for range or 'm' for mean.")
       print("Setting range metrics as default.")
    aggregation = 'range'
    if aggr == 'm': aggregation = 'mean'
    dfn = anonymize_df(df,q, equivalence_classes, cat_indices, aggregation)
    #np.savetxt('anon_df.csv', dfn, fmt='%s', delimiter=';')
    print('Anonymization completed.')
    dfn=pd.DataFrame(dfn,columns=qid)
    return dfn



def calculate_metrics_mondrian(NICE_fit,clf,X,X_train, X_test, y_test,num,cat,feature_names,target_outcome,qid,dfn):
    ex_times_dataset=[]
    pureness_list_dataset=[]
    NCP_list_dataset=[]
    sel_frame_dataset=[]
    X_test_values=X.iloc[X_test].values
    trainingdata=X.iloc[X_train]
    trainingset=trainingdata[qid]
    if len(y_test)<=1000:
        n=len(y_test)
    else:
        n=1000
    for i in tqdm(range(n)):
        to_explain = X_test_values[i:i+1,:]
        if clf.predict(to_explain)!=target_outcome:
            CF = NICE_fit.explain(to_explain)
            instance = pd.Series(CF[0], index=feature_names)
            start=time.perf_counter()
            sel_frame=anom_instance(dfn,instance,qid,num,cat)
            #mondrian_cmc_cf_frame.append(sel_frame)
            NCP_dataset=calculate_solution_quality_dataset(trainingset,sel_frame,num, cat)
            pureness_dataset=calculate_pureness_dataset(sel_frame,instance,trainingset,feature_names,clf,target_outcome,num,cat,qid)
            end=time.perf_counter()
            ex_times_dataset.append(end-start)
            pureness_list_dataset.append(pureness_dataset)
            NCP_list_dataset.append(NCP_dataset)
            sel_frame_dataset.append(sel_frame)
    return  NCP_list_dataset,pureness_list_dataset,ex_times_dataset,sel_frame_dataset

# def anom_instance(dfn,instance,qid,num,cat):
#     sel_frame=pd.DataFrame(columns=qid)
#     #instance_eq=instance.copy()
#     for i in range(len(dfn)):
#         for  q in qid:
#             if qid.index(q) in num:
#                 if type(dfn.iloc[i][q])==str: 
#                     if dfn.iloc[i][q].count('-')>=2:
#                         temp = dfn.iloc[i][q].split('-')
#                         low, high = '-'.join(temp[:2]), '-'.join(temp[2:])
#                         if instance[q]>=float(low) and instance[q]<= float(high):
#                             continue
#                         else:
#                             break 
#                     elif (type(dfn.iloc[i][q])==str and '-' in dfn.iloc[i][q]):
#                         low,high=dfn.iloc[i][q].split('-')
#                         if instance[q]>=float(low) and instance[q]<= float(high):
#                             continue
#                         else:
#                             break 
#                 elif (dfn.iloc[i][q]==instance[q] or float(dfn.iloc[i][q])==instance[q]):
#                         continue
#                 else:
#                     break
#             if qid.index(q) in cat:
#                 if (type(dfn.iloc[i][q])==str and ',' in dfn.iloc[i][q]):
#                     lijst=dfn.iloc[i][q].split(',')
#                     try:
#                         lijst=[float(item) for item in lijst]
#                     except ValueError:
#                         lijst=lijst
#                     if instance[q] in lijst:
#                         continue
#                     else:
#                         break 
#                 else:
#                     try: 
#                         if float(dfn.iloc[i][q])==instance[q]:
#                             continue
#                     except ValueError:
#                         if dfn.iloc[i][q]==instance[q]: #or float(dfn.iloc[i][q])==instance[q]):
#                             continue
#                         else:
#                             break
#         else:
#             sel_frame=sel_frame.append(dfn.iloc[i])
#     if len(sel_frame)==0:
#         print('Error, no equivalence class found for this counterfactual instance')
#     return sel_frame

def anom_instance(dfn,instance,qid,num,cat):
    sel_frame=pd.DataFrame(columns=qid)
    #instance_eq=instance.copy()
    for i in range(len(dfn)):
        for  q in qid:
            if qid.index(q) in num:
                if type(dfn.iloc[i][q])==str: 
                    if dfn.iloc[i][q].count('-')>=2:
                        temp = dfn.iloc[i][q].split('-')
                        low, high = '-'.join(temp[:2]), '-'.join(temp[2:])
                        if instance[q]>=float(low) and instance[q]<= float(high):
                            continue
                        else:
                            break 
                    elif (type(dfn.iloc[i][q])==str and '-' in dfn.iloc[i][q]):
                        low,high=dfn.iloc[i][q].split('-')
                        if instance[q]>=float(low) and instance[q]<= float(high):
                            continue
                        else:
                            break 
                elif (dfn.iloc[i][q]==instance[q] or float(dfn.iloc[i][q])==instance[q]):
                       continue
                else:
                    break
            if qid.index(q) in cat:
                if (type(dfn.iloc[i][q])==str and ',' in dfn.iloc[i][q]):
                    lijst=dfn.iloc[i][q].split(',')
                    try:
                        lijst=[float(item) for item in lijst]
                    except ValueError:
                        lijst=lijst
                    if instance[q] in lijst:
                        continue
                    else:
                        try:
                            if float(instance[q]) in lijst:
                                continue
                            else:
                                break
                        except ValueError:
                            break 
                else:
                    if dfn.iloc[i][q]==instance[q]: #or float(dfn.iloc[i][q])==instance[q]):
                        continue
                    else:
                        try: 
                            if float(dfn.iloc[i][q])==instance[q]:
                                continue
                            else:
                                break
                        except ValueError:
                            break
        else:
            sel_frame=sel_frame.append(dfn.iloc[i])
    if len(sel_frame)==0:
        print('Error, no equivalence class found for this counterfactual instance')
    return sel_frame

def calculate_solution_quality_dataset(trainingset,selected,num, cat):  
    NCP_attribute={}
    for j in range(len(selected.columns)):
        if j in num:
            if type(selected.iloc[0][j])==str: 
                if selected.iloc[0][j].count('-')>=2:
                    temp = selected.iloc[0][j].split('-')
                    low, high = '-'.join(temp[:2]), '-'.join(temp[2:])
                elif (type(selected.iloc[0][j])==str and '-' in selected.iloc[0][j]):
                    low,high=selected.iloc[0][j].split('-')
                    NCP_attribute[j]= (float(high) - float(low))/(max(trainingset.iloc[:,j])-min(trainingset.iloc[:,j]))
            else:
                NCP_attribute[j]=0
        if j in cat:
            if (type(selected.iloc[0][j])==str and ',' in selected.iloc[0][j]):
                lijst=selected.iloc[0][j].split(',')
                NCP_attribute[j]=len(lijst)/trainingset.iloc[:,j].nunique()  
            else:
                NCP_attribute[j]=0   
    w=1/len(selected.columns)
    NCP=sum(NCP_attribute.values())*w
    return NCP

def calculate_pureness_dataset(sel_frame,instance,trainingset,feature_names,clf,target_outcome,num,cat,qid):
    combilist={}
    for i in range(len(instance)):
         feature=feature_names[i]
         combilist[feature]=[]
         if feature not in qid:
             combilist[feature]=instance[i]
         if feature in qid:
             f=qid.index(feature)
             if f in num:
                 if type(sel_frame.iloc[0][f])==str: 
                     if sel_frame.iloc[0][f].count('-')>=2:
                         temp = sel_frame.iloc[0][f].split('-')
                         low, high = '-'.join(temp[:2]), '-'.join(temp[2:])
                         vals=trainingset.iloc[:,f].unique()
                     elif (type(sel_frame.iloc[0][f])==str and '-' in sel_frame.iloc[0][f]):
                         low,high=sel_frame.iloc[0][f].split('-')
                         vals=trainingset.iloc[:,f].unique()
                     for val in vals:
                         if (val>=float(low)) and (val<=float(high)):
                             combilist[feature].append(val)
                 else:
                    combilist[feature].append(float(sel_frame.iloc[0][f]))
             if f in cat:
                 if (type(sel_frame.iloc[0][f])==str and ',' in sel_frame.iloc[0][f]):
                     lijst=sel_frame.iloc[0][f].split(',')
                     try:
                         lijst=[float(item) for item in lijst]
                     except ValueError:
                         lijst=[item for item in lijst]
                     combilist[feature]=lijst
                 else:
                     try:
                         combilist[feature].append(float(sel_frame.iloc[0][f]))
                     except ValueError:
                        combilist[feature].append(sel_frame.iloc[0][f])
    #sample 100 combinations
    sample_frame=pd.DataFrame(columns=feature_names,index=None)
    while len(sample_frame)<100:
        id_instance={}
        for key in combilist.keys():
            if type(combilist[key])!=list:
                id_instance[key]=combilist[key]
            else:
                id_instance[key]=random.choice(combilist[key])
        sample_frame=sample_frame.append(id_instance.copy(),ignore_index=True)
    #calculate pureness
    #teller=0
    #noemer=0
    #for i in range(len(sample_frame)):
        #noemer+=1
        #if clf.predict([sample_frame.iloc[i]])==target_outcome:
            #teller+=1
    #pureness=teller/noemer
    predictions=clf.predict(sample_frame.values)
    counts=Counter(predictions)
    pureness=counts[target_outcome]/len(sample_frame)
    return pureness

        
            
                
            
                     
                 
             
             
             
    

