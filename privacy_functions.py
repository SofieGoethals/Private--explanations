# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:01:01 2022

@author: SGoethals
"""

import operator, math,random,copy, time
from statistics import mean
import seaborn as sns
import numpy as np
from nice import NICE
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from collections import Counter


#%% modelling

def model(X,y,feature_names, cat_feat, num_feat, target_outcome):
    print('Modelling..')
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.4,random_state=0)
    X_train_values=X.iloc[X_train].values
    y_train_values=y.iloc[X_train].values
    clf = Pipeline([
        ('PP',ColumnTransformer([
                ('num',StandardScaler(),num_feat),
                ('cat',OneHotEncoder(handle_unknown = 'ignore'),cat_feat)])),
        ('RF',RandomForestClassifier(random_state=0))])
    pipe_params = {
    "RF__n_estimators": [ 10,50,100, 500, 1000,5000],
    "RF__max_leaf_nodes":[10,100,500,None],
    }
    grid=GridSearchCV(clf, pipe_params, cv=5,n_jobs=-1)
    grid.fit(X_train_values,y_train_values)
    clf=grid.best_estimator_
    predict_fn = lambda x: clf.predict_proba(x)
    NICE_fit = NICE(X_train= X_train_values,
                   predict_fn=predict_fn,
                   y_train = y_train_values,
                   cat_feat=cat_feat,
                   num_feat=num_feat,
                   distance_metric='HEOM',
                   num_normalization='minmax',
                   optimization='none',
                   justified_cf=False
                   )
    return X_train, y_train, X_test,y_test,clf,NICE_fit

def feature_importances(clf,feature_names,num_feat,cat_feat):
    num_names=[x for i,x in enumerate(feature_names) if i in num_feat]    
    cat_names=[x for i,x in enumerate(feature_names) if i in cat_feat]
    onehot_columns=clf.named_steps['PP'].named_transformers_['cat'].get_feature_names(input_features=cat_names)
    feat_after_pipeline=np.array(num_names+list(onehot_columns))
    plt.barh(feat_after_pipeline, clf['RF'].feature_importances_)


#%% GRASP

def grasp_results(discr_attr,X,y,X_train,X_test,y_test, y_train,target_outcome, k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf, NICE_fit):
    ex_times=[]
    pureness_list=[]
    NCP_list=[]
    discr_list=[]
    gen_list=[]
    instance_list=[]
    X_test_values=X.iloc[X_test].values
    y_test_values=y.iloc[X_test].values
    testdata = X.iloc[X_test]
    trainingdata=X.iloc[X_train]
    if len(y_test_values)<=1000:
        n=len(y_test_values)
    else:
        n=1000
    for i in tqdm(range(n)):
        #print(i)
        to_explain = X_test_values[i:i+1,:]
        #only explain the instances with a bad prediction
        if clf.predict(to_explain)!=target_outcome:
            print('instance'+str(i))
            CF = NICE_fit.explain(to_explain)
            #print(CF)
            instance=pd.Series(CF[0],index=feature_names)
            start=time.perf_counter()
            best_selected,best_instance,best_pureness,best_NCP=GRASP(trainingdata, y_train,target_outcome, instance, k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf)
            end=time.perf_counter()
            ex_times.append(end-start)
            pureness_list.append(best_pureness)
            NCP_list.append(best_NCP)
            gen_list.append(best_instance)
            instance_list.append(instance)
            discr_list.append(testdata[discr_attr].iloc[i])
    return ex_times, pureness_list, NCP_list, gen_list, discr_list,instance_list


def greedy_randomized_construction(trainingdata,y_train,target_outcome, instance, k, alfa,  num_feat, cat_feat,feature_names,qid,clf):
    trainingset=trainingdata[qid]
    testinstance=instance[qid]
    num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
    cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
    selected=calculate_eq_class(testinstance,trainingset,num,cat)
    generalized_instance=testinstance
    if check_k_anom(selected,k):
        NCP = calculate_solution_quality(trainingset,selected,num, cat)
        pureness=calculate_pureness_sampling(instance,generalized_instance,trainingset,feature_names,clf,target_outcome,num,cat,qid)
    else:
        while not check_k_anom(selected,k):
            neighbors= getNeighbors (trainingset, generalized_instance,y_train,target_outcome, alfa, cat, num) #cat_feat,num_feat)
            neighbor = SelectFromList(alfa, neighbors)
            generalized_instance=GeneralizeInstance(generalized_instance,neighbor)
            selected=calculate_eq_class(generalized_instance,trainingset,num_feat,cat_feat)
            identical=calculate_eq_class_all_attributes(instance,generalized_instance,qid,trainingdata,num_feat,cat_feat,feature_names)
        NCP = calculate_solution_quality(trainingset,selected,num, cat)
        pureness=calculate_pureness_sampling(instance,generalized_instance,trainingset,feature_names,clf,target_outcome,num,cat,qid)
    return pureness, NCP,selected,generalized_instance
    
    
def local_search(gen_instance,pureness, NCP, num_feat, cat_feat,trainingdata,instance,k,target_outcome,y_train,feature_names,qid,clf):
    trainingset=trainingdata[qid]
    testinstance=instance[qid]
    num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
    cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
    i=0
    sol=None
    timeout=time.time()+5
    while i<len(gen_instance) and time.time()<timeout: #first improving
        solution=copy.deepcopy(gen_instance)
        solution=np.array(solution,dtype=object).tolist()
        if i in cat:
            if type(gen_instance[i])!=list:
                r=random.randint(0,len(trainingset.iloc[:,i].unique())-2) #-2 because we remove an instance from the trainingset, and the indexing starts at 0
                rl=trainingset.iloc[:,i].unique().tolist()
                rl.remove(gen_instance[i])
                solution[i]=[gen_instance[i],rl[r]]
                if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                    break
            elif (type(gen_instance[i])==list) and (len(gen_instance[i])>1):
                if len(gen_instance[i])>2:
                    r=random.randint(0,len(gen_instance[i])-2)
                else:
                    r=0
                rl=copy.deepcopy(gen_instance[i])
                rl.remove(testinstance[i])
                solution[i].remove(rl[r])

                if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                    break
        elif i in num:
            if type(gen_instance[i])!=list:
                solution[i]=[gen_instance[i],gen_instance[i] - 1]
                if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                    break
                else:
                    solution=copy.deepcopy(gen_instance)
                    solution=np.array(solution,dtype=object).tolist()
                    solution[i]=[gen_instance[i],gen_instance[i] + 1]
                    if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                        break

            elif type(gen_instance[i])==list and len(gen_instance[i])>1:
               solution[i].sort()
               if testinstance[i]!= solution[i][0]:
                   solution[i].remove(solution[i][0])
                   if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                       break
               else:
                   solution=copy.deepcopy(gen_instance) 
                   solution=np.array(solution,dtype=object).tolist()
                   if testinstance[i]!= solution[i][-1]:
                       solution[i].remove(solution[i][-1])
                       if check_solution(solution,trainingdata,instance,num_feat,cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
                           break

        i+=1
    if sol==None:
        sol=copy.deepcopy(gen_instance)
    selected_sol=calculate_eq_class(sol,trainingset,num_feat,cat_feat)
    #identical=calculate_eq_class_all_attributes(instance,sol,qid,trainingdata,num_feat,cat_feat,feature_names)
    #pureness_sol, NCP_sol=calculate_solution_quality(trainingset, selected_sol, identical, y_train, target_outcome, num, cat)
    NCP_sol = calculate_solution_quality(trainingset,selected_sol,num, cat)
    #pureness_sol= calculate_pureness(instance, sol, qid, feature_names, num_feat, cat_feat, trainingdata,clf, target_outcome)
    pureness_sol=calculate_pureness_sampling(instance,sol,trainingset,feature_names,clf,target_outcome,num,cat,qid)
    return pureness_sol, NCP_sol,selected_sol, sol
    #calculate_solution_quality(trainingset, selected, y_train, target_outcome, num_feat, cat_feat)
                       
def check_solution(solution,trainingdata,instance,num_feat, cat_feat,y_train, target_outcome,pureness,NCP,feature_names,qid,k,clf):
    num=[i for i,v in enumerate(qid) if feature_names.index(v) in num_feat]
    cat=[i for i,v in enumerate(qid) if feature_names.index(v) in cat_feat]
    trainingset=trainingdata[qid]
    selected_sol=calculate_eq_class(solution,trainingset,num, cat)
    if check_k_anom(selected_sol, k):
        NCP_sol = calculate_solution_quality(trainingset,selected_sol,num, cat)
        #pureness_sol= calculate_pureness(instance, solution, qid, feature_names, num_feat, cat_feat, trainingdata,clf, target_outcome)
        pureness_sol=calculate_pureness_sampling(instance,solution,trainingset,feature_names,clf,target_outcome,num,cat,qid)
        if (pureness_sol>pureness or (pureness_sol==pureness and NCP_sol<NCP)):
             cond=True
             #print('better solution found in local neighborhood')
        else:
             cond=False
             #print('solution not better')
    else:
        cond=False
        #print('k-anonymity not satisfied')
    return cond

def GRASP(trainingdata, y_train,target_outcome, instance, k, alfa, qid, num_feat, cat_feat,feature_names,max_iterations,clf):
    best_pureness=0
    best_NCP=1 #the lower the better
    timeout=time.time()+30
    i=0
    #for i in range(max_iterations):
    while i<=max_iterations: #and time.time()<timeout:
        #print('iteratie '+str(i))
        pureness,NCP,selected,gen_instance=greedy_randomized_construction(trainingdata, y_train,target_outcome, instance, k, alfa,  num_feat, cat_feat,feature_names,qid,clf)
        #print('after construction: {}'.format(gen_instance))
        pureness, NCP,selected, gen_instance=local_search(gen_instance,pureness, NCP, num_feat, cat_feat,trainingdata, instance,k,target_outcome,y_train,feature_names,qid,clf)
        #print('after local search: {}'.format(gen_instance))
        if ((pureness>best_pureness) or (pureness==best_pureness and NCP<best_NCP)):
            best_selected=selected
            best_instance=gen_instance
            best_pureness=pureness
            best_NCP=NCP
        i+=1
            #print('solution is improved. New solution: {}, pureness:{}, NCP: {}'.format(best_instance,best_pureness,best_NCP))
    #print('Iterations finished. The best solution is {}, with equivalence class {}, pureness {} and NCP {}'.format(best_instance,best_selected,best_pureness,best_NCP))
    return best_selected,best_instance,best_pureness,best_NCP

#%% helper functions

def calculate_distance(testinstance,traininginstance,cat_feat,num_feat,trainingset):
    distance=0
    for i in range(len(testinstance)):
        if i in cat_feat:
            if type(testinstance[i])!=list:
                if testinstance[i]!=traininginstance.iloc[i]:
                    distance+=1
            if type(testinstance[i])==list:
                if traininginstance.iloc[i] not in testinstance[i]:
                    distance+=1
        if i in num_feat:
            width=trainingset.iloc[:,i].max(axis=0)-trainingset.iloc[:,i].min(axis=0)
            if type(testinstance[i])!=list:
                if testinstance[i]!=traininginstance.iloc[i]:
                    distance+=abs(testinstance[i]-traininginstance.iloc[i])/width
            if type(testinstance[i])==list:
                testinstance[i].sort()
                if not ((traininginstance.iloc[i] >= testinstance[i][0]) & (traininginstance.iloc[i] <= testinstance[i][-1])):
                    dist=min(abs(testinstance[i][0]-traininginstance.iloc[i]), abs(testinstance[i][-1]-traininginstance.iloc[i]))
                    distance+=dist/width
    return distance

def getNeighbors (trainingset, testinstance,y_train,target_outcome, l, cat_feat, num_feat):
    distances=[]
    #length = len(testinstance)-1
    for i in range(len(trainingset)):
        dist= calculate_distance(testinstance,trainingset.iloc[i],cat_feat,num_feat,trainingset)
        distances.append((trainingset.iloc[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    x=0
    while len(neighbors)!=l:
        index=distances[x][0].name
        if y_train[index]==target_outcome:
            neighbors.append(distances[x][0])
        x+=1
    return neighbors

def SelectFromList(l,neighbors):
    lst=list(range(0,l))
    i=random.choice(lst)
    neighbor=neighbors[i]
    return neighbor

def GeneralizeInstance(instance,neighbor):
    generalized_instance=[None]*len(instance)
    for i in range(len(instance)):
        if type(instance[i])!=list:
            if instance[i]==neighbor[i]:
                generalized_instance[i]=instance[i]
                continue
            elif instance[i]!= neighbor[i]:
                generalized_instance[i]=[instance[i],neighbor[i]]
                continue
        elif type(instance[i])==list:
            if neighbor[i] in instance[i]:
               generalized_instance[i]=instance[i]
               continue 
            else:
                generalized_instance[i]=instance[i]
                generalized_instance[i].append(neighbor[i])
    return generalized_instance
            
            
def calculate_eq_class(generalized_instance,trainingset,num, cat):
    selected=trainingset.copy()
    for i in range(len(generalized_instance)):
        if type(generalized_instance[i])!=list:
            selected=selected[(selected.iloc[:,i]==generalized_instance[i])]
        elif type(generalized_instance[i])==list:
            if i in num:
                generalized_instance[i].sort()
                selected=selected[(selected.iloc[:,i] >= generalized_instance[i][0]) & (selected.iloc[:,i] <= generalized_instance[i][-1])]
            elif i in cat:
                selected=selected[(selected.iloc[:,i].isin(generalized_instance[i]))]      
    return selected

def calculate_eq_class_all_attributes(instance,generalized_instance,qid,trainingdata,num_feat,cat_feat,feature_names):
#id_instance is the same as the generalized instance but with the other attribute values      
    id_instance=[None]*len(instance)
    for i in range(len(instance)):
        feature=feature_names[i]
        if feature in qid:
            f=qid.index(feature)
            id_instance[i]=generalized_instance[f]
        if feature not in qid:
            id_instance[i]=instance[i]
    identical=trainingdata.copy()
    for i in range(len(id_instance)):
        if type(id_instance[i])!=list:
            identical=identical[(identical.iloc[:,i]==id_instance[i])]
        elif type(id_instance[i])==list:
            if i in num_feat:
                id_instance[i].sort()
                identical=identical[(identical.iloc[:,i] >= id_instance[i][0]) & (identical.iloc[:,i] <= id_instance[i][-1])]
            elif i in cat_feat:
                identical=identical[(identical.iloc[:,i].isin(id_instance[i]))]      
    return identical
    
def check_k_anom(selected,k):
    if len(selected)>=k:
        return True
    else:
        return False
            

def calculate_solution_quality(trainingset,selected,num, cat):  
    NCP_attribute={}
    for j in range(len(selected.columns)):
        if j in num: 
            NCP_attribute[j]= (max(selected.iloc[:,j])-min(selected.iloc[:,j]))/(max(trainingset.iloc[:,j])-min(trainingset.iloc[:,j]))
        if j in cat:
            if selected.iloc[:,j].nunique()==1:
                NCP_attribute[j]=0
            else:
                NCP_attribute[j]=selected.iloc[:,j].nunique()/trainingset.iloc[:,j].nunique()      
    w=1/len(selected.columns)
    NCP=sum(NCP_attribute.values())*w
    return NCP


#just take 100 random samples to calculate for the purity
def calculate_pureness_sampling(instance,generalized_instance,trainingset,feature_names,clf,target_outcome,num,cat,qid):
    combilist={}
    for i in range(len(instance)):
         feature=feature_names[i]
         combilist[feature]=[]
         if feature not in qid:
             combilist[feature]=instance[i]
         if feature in qid:
             f=qid.index(feature)
             if f in num:
                 if type(generalized_instance[f])==list: 
                     low, high = generalized_instance[f][0], generalized_instance[f][-1]
                     vals=trainingset.iloc[:,f].unique()
                     for val in vals:
                         if (val>=float(low)) and (val<=float(high)):
                             combilist[feature].append(val)
                 else:
                    combilist[feature].append(float(generalized_instance[f]))
             if f in cat:
                 combilist[feature]=generalized_instance[f]
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
    #calculate pureness
    predictions=clf.predict(sample_frame.values)
    counts=Counter(predictions)
    pureness=counts[target_outcome]/len(sample_frame)
    return pureness

