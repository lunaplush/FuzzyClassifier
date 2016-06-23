# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import arff

import skfuzzy as fuzz


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score


    


def plot_res(x1,x2):
#    plt.plot(x[0,:],x[1,:])
    fig, ax = plt.subplots()

    #ax.scatter(x[0,:],x[1,:],c=t,s=25)      
    ax.scatter([p[x1] for p in x ],[p[x2] for p in x],c = t,s=5)    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('machine learning')
    
#number = 100  
#x = np.random.sample((2,number))
#t = np.random.randint(1,4,number)
    
#%% open data    
data_ps = arff.load(open("c:/Luna/Work/python/data/data_PS/ps_c7_a16_all.arff") )   

x = [xx[:-1] for xx in data_ps.get('data')]
t = [xx[-1] for xx in data_ps.get('data')]

#data_ps=pd.DataFrame(x,index = t,columns = [d[0] for d in data.get('attributes')][:-1])

classes={}
xclasses={}
i = 0
for ti in t:
    if (classes.get(ti) is None):
        classes[ti]=1
        xclasses[ti]=[x[i]]
    else: 
        classes[ti]=classes[ti]+1
        xclasses[ti].append(x[i])
    i=i+1
features_num = len(x[0])        
#%% create fuzzy rules
uni_min = np.array([2000000.0 for c in np.arange(float(features_num))])
uni_max = np.array([-1 for c in np.arange(float(features_num))])
feature = [[np.nan] for f in range(0, features_num)]
for key in xclasses.keys():
       for f in range(0,features_num):
            ff = [x[f] for x in  xclasses.get(key)]
            [f_min,f_max] = [min(ff),max(ff)]
            feature[f] = [f_min,f_max] TERM
            uni_min[f] = min([f_min,uni_min[f]]) 
            uni_max[f] = max([f_max,uni_max[f]])

uni_step = (uni_max - uni_min) / 1000
mfs = [[np.nan] for f in range(0, features_num)]        
mfs_uni = [[np.nan] for f in range(0, features_num)]
for lingvo in range(0, features_num):
    mfs_uni[lingvo] = np.arange(uni_min[lingvo],uni_max[lingvo],uni_step[lingvo])
    term = 0
    for term_key in xclasses.key():
         mfs[lingvo][term] = fuzz.trimf(mfs_uni[lingvp],[feature[x][0],feature[x][0]+(feature[x][1]-feature[x][0])/2,feature[x][1]])
         term=term+1
        
#mf1_universum=np.arange(universum_min,universum_max + universum_step,universum_step)

#mf_feature1 = fuzz.trimf(mf1_universum,[min(feature1),min(feature1)+((max(feature1)-min(feature1))/2),max(feature1)])
#mf_feature2 = fuzz.trimf(mf1_universum,[min(feature2),min(feature2)+((max(feature2)-min(feature2))/2),max(feature2)])

        
        

#%% visualization
fig, ax = plt.subplots()

x=0
plt.plot( mfs_uni[x],mfs,'r',mf1_universum,mf_feature2,'g')





#%% something next

        
#clf= DecisionTreeClassifier(random_state=0)
#cross_val_score(clf, x, t, cv=10)

#plot_res(1,2)
#plt.show()


 
#f = open("c:/Luna/Work/python/tmp.txt",'rb')
#f = Io.StringIO("Hello")
#data.get('data')
#x = [xx[1:3] for xx in data.get('data')]
#t = [xx[15] for xx in data.get('data')]
#
#data_wine = pd.read_csv("c:/Luna/Work/python/data/wine.data")
#A =data_wine.loc[(data_wine["1"] > 13) & (data_wine["2"] > 1),["1","2"]]



#def num_missing(x):
#    return sum(x*x)
    
#print("Missing values per column:")
#print(data_wine.apply(num_missing, axis=1).head())
#print(data_wine.apply(num_missing, axis=0).head()) 
#r=data_wine.apply(num_missing, axis=1).head()   




