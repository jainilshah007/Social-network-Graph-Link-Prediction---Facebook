#!/usr/bin/env python
# coding: utf-8

# In[50]:


import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd#pandas to create small dataframes 
import datetime #Convert to unix time
import time #Convert to unix time
import numpy as np#Do aritmetic operations on arrays
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns#Plots
from matplotlib import rcParams#Size of plots  
from sklearn.cluster import MiniBatchKMeans, KMeans#Clustering
import math
import pickle
import os

import xgboost as xgb

import warnings
import networkx as nx
import pdb
import pickle


# In[51]:


if not os.path.isfile(r"D:\data\train_woheader.csv"):
    traincsv=pd.read_csv(r"D:\data\train.csv")
    print(traincsv[traincsv.isna().any(1)])
    print(traincsv.info())
    print("Number of diplicate entries: ",sum(traincsv.duplicated()))
    traincsv.to_csv(r"D:\data\train_woheader.csv",header=False,index=False)
    print("saved the graph into file")
else :
    g=nx.read_edgelist(r"D:\data\train_woheader.csv", delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    print(nx.info(g))


# In[52]:


if not os.path.isfile('train_woheader_sample.csv'):
    pd.read_csv(r'D:\data\train.csv', nrows=50).to_csv(to_csv('train_woheader_sample.csv',header=False,index=False)

subgraph=nx.read_edgelist('train_woheader_sample.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)


pos=nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color='#A0CBE2',edge_color='#00bb5e',width=1,edge_cmap=plt.cm.Blues,with_labels=True)
plt.savefig("graph_sample.pdf")
print(nx.info(subgraph))                                                       


# In[ ]:


print("The number of unique persons",len(g.nodes()))


# In[ ]:


indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(indegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()


# In[ ]:


plt.boxplot(indegree_dist)
plt.ylabel('No Of Followers')
plt.show()


# In[ ]:


for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(indegree_dist,90+i))


# In[ ]:


for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(indegree_dist,99+(i/100)))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(indegree_dist, color='#16A085')
plt.xlabel('PDF of Indegree')
sns.despine()


# In[ ]:


# No of people each person is following


# In[ ]:


outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()


# In[ ]:


indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(outdegree_dist[0:1500000])
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following')
plt.show()


# In[ ]:


for i in range(0,11):
    print(90+i,'percentile value is',np.percentile(outdegree_dist,90+i))


# In[ ]:


for i in range(10,110,10):
    print(99+(i/100),'percentile value is',np.percentile(outdegree_dist,99+(i/100)))


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(outdegree_dist, color='#16A085')
plt.xlabel('PDF of Outdegree')
sns.despine()


# In[ ]:


print('No of persons those are not following anyone are' ,sum(np.array(outdegree_dist)==0),'and % is',
                                sum(np.array(outdegree_dist)==0)*100/len(outdegree_dist) )


# In[ ]:


count=0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0 :
        if len(list(g.successors(i)))==0:
            count+=1
print('No of persons those are not not following anyone and also not having any followers are',count)


# In[ ]:


#both followers + following


# In[ ]:


from collections import Counter
dict_in = dict(g.in_degree())
dict_out = dict(g.out_degree())
d = Counter(dict_in) + Counter(dict_out)
in_out_degree = np.array(list(d.values()))


# In[ ]:


in_out_degree_sort = sorted(in_out_degree)
plt.figure(figsize=(10,6))
plt.plot(in_out_degree_sort)
plt.xlabel('Index No')
plt.ylabel('No Of people each person is following + followers')
plt.show()


# In[ ]:


print('Min of no of followers + following is',in_out_degree.min())
print(np.sum(in_out_degree==in_out_degree.min()),' persons having minimum no of followers + following')


# In[ ]:


print('No of weakly connected components',len(list(nx.weakly_connected_components(g))))
count=0
for i in list(nx.weakly_connected_components(g)):
    if len(i)==2:
        count+=1
print('weakly connected components wit 2 nodes',count)


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "###generating bad edges from given graph\nimport random\nif not os.path.isfile(r'D:\\data\\missing_edges_final.p'):\n    #getting all set of edges\n    r = csv.reader(open(r'D:\\data\\train_woheader.csv','r'))\n    edges = dict()\n    for edge in r:\n        edges[(edge[0], edge[1])] = 1\n        \n        \n    missing_edges = set([])\n    while (len(missing_edges)<9437519):\n        a=random.randint(1, 1862220)\n        b=random.randint(1, 1862220)\n        tmp = edges.get((a,b),-1)\n        if tmp == -1 and a!=b:\n            try:\n                if nx.shortest_path_length(g,source=a,target=b) > 2: \n\n                    missing_edges.add((a,b))\n                else:\n                    continue  \n            except:  \n                    missing_edges.add((a,b))              \n        else:\n            continue\n    pickle.dump(missing_edges,open(r'D:\\data\\missing_edges_final.p','wb'))\nelse:\n    missing_edges = pickle.load(open(r'D:\\data\\missing_edges_final.p','rb'))")


# In[ ]:


from sklearn.model_selection import train_test_split
if (not os.path.isfile(r'D:\data\train_pos_after_eda.csv')) and (not os.path.isfile('data/after_eda/test_pos_after_eda.csv')):
    #reading total data df
    df_pos = pd.read_csv(r'D:\data\train.csv')
    df_neg = pd.DataFrame(list(missing_edges), columns=['source_node', 'destination_node'])
    
    print("Number of nodes in the graph with edges", df_pos.shape[0])
    print("Number of nodes in the graph without edges", df_neg.shape[0])
    
    #Trian test split 
    #Spiltted data into 80-20 
    #positive links and negative links seperatly because we need positive training data only for creating graph 
    #and for feature generation
    X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos,np.ones(len(df_pos)),test_size=0.2, random_state=9)
    X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg,np.zeros(len(df_neg)),test_size=0.2, random_state=9)
    
    print('='*60)
    print("Number of nodes in the train data graph with edges", X_train_pos.shape[0],"=",y_train_pos.shape[0])
    print("Number of nodes in the train data graph without edges", X_train_neg.shape[0],"=", y_train_neg.shape[0])
    print('='*60)
    print("Number of nodes in the test data graph with edges", X_test_pos.shape[0],"=",y_test_pos.shape[0])
    print("Number of nodes in the test data graph without edges", X_test_neg.shape[0],"=",y_test_neg.shape[0])

    #removing header and saving
    X_train_pos.to_csv(r'D:\data\train_pos_after_eda.csv',header=False, index=False)
    X_test_pos.to_csv(r'D:\data\test_pos_after_eda.csv',header=False, index=False)
    X_train_neg.to_csv(r'D:\data\train_neg_after_eda.csv',header=False, index=False)
    X_test_neg.to_csv(r'D:\data\test_neg_after_eda.csv',header=False, index=False)
else:
    #Graph from Traing data only 
    del missing_edges


# In[ ]:


if (os.path.isfile(r'D:\data\train_pos_after_eda.csv')) and (os.path.isfile(r'D:\data\test_pos_after_eda.csv')):        
    train_graph=nx.read_edgelist(r'D:\data\train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    test_graph=nx.read_edgelist(r'D:\data\test_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
    print(nx.info(train_graph))
    print(nx.info(test_graph))

    # finding the unique nodes in the both train and test graphs
    train_nodes_pos = set(train_graph.nodes())
    test_nodes_pos = set(test_graph.nodes())

    trY_teY = len(train_nodes_pos.intersection(test_nodes_pos))
    trY_teN = len(train_nodes_pos - test_nodes_pos)
    teY_trN = len(test_nodes_pos - train_nodes_pos)

    print('no of people common in train and test -- ',trY_teY)
    print('no of people present in train but not present in test -- ',trY_teN)

    print('no of people present in test but not present in train -- ',teY_trN)
    print(' % of people not there in Train but exist in Test in total Test data are {} %'.format(teY_trN/len(test_nodes_pos)*100))


# In[ ]:


#final train and test data sets
import os as my_aliased_module
import ntpath as path
if (not os.path.isfile(r'D:\data\train_after_eda.csv')) and (not os.path.isfile(r'D:\data\test_after_eda.csv')) and (not os.path.isfile(r'D:\data\train_y.csv')) and (not os.path.isfile(r'D:\data\test_y.csv')) and (os.path.isfile(r'D:\data\train_pos_after_eda.csv')) and (os.path.isfiler(r'D:\data\test_pos_after_eda.csv')) and (os.path.isfile(r'D:\data\train_neg_after_eda.csv')) and (os.path.isfile(r'D:\data\test_neg_after_eda.csv')):
    
    X_train_pos = pd.read_csv(r'D:\data\train_pos_after_eda.csv', names=['source_node', 'destination_node'])
    X_test_pos = pd.read_csv(r'D:\data\test_pos_after_eda.csv', names=['source_node', 'destination_node'])
    X_train_neg = pd.read_csv(r'D:\data\train_neg_after_eda.csv', names=['source_node', 'destination_node'])
    X_test_neg = pd.read_csv(r'D:\data\test_neg_after_eda.csv', names=['source_node', 'destination_node'])

    print('='*60)
    print("Number of nodes in the train data graph with edges", X_train_pos.shape[0])
    print("Number of nodes in the train data graph without edges", X_train_neg.shape[0])
    print('='*60)
    print("Number of nodes in the test data graph with edges", X_test_pos.shape[0])
    print("Number of nodes in the test data graph without edges", X_test_neg.shape[0])

    X_train = X_train_pos.append(X_train_neg,ignore_index=True)
    y_train = np.concatenate((y_train_pos,y_train_neg))
    X_test = X_test_pos.append(X_test_neg,ignore_index=True)
    y_test = np.concatenate((y_test_pos,y_test_neg)) 
    
    X_train.to_csv(r'D:\data\train_after_eda.csv',header=False,index=False)
    X_test.to_csv(r'D:\data\test_after_eda.csv',header=False,index=False)
    pd.DataFrame(y_train.astype(int)).to_csv(r'D:\data\train_y.csv',header=False,index=False)
    pd.DataFrame(y_test.astype(int)).to_csv(r'D:\data\test_y.csv',header=False,index=False)


# In[ ]:


print("Data points in train data",X_train.shape)
print("Data points in test data",X_test.shape)
print("Shape of traget variable in train",y_train.shape)
print("Shape of traget variable in test", y_test.shape)


# In[ ]:




