#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import random
from matplotlib import pyplot as plt
data=scipy.io.loadmat('samples.mat')


# In[2]:


points=data['AllSamples']
Dict={} #dictionary to store cluster points as values in specific cluster 'key'


# In[3]:


def euclidean(a,b): #find euclidean distance between two points
    return np.linalg.norm(a-b)


# In[4]:


def updatedict(centroids): #update dictionary clusters every time centroids change
    global Dict
    for i in range(len(centroids)):
        Dict[i+1]= [] #initialise dictionary with cluster number as keys and empty values
    for i in range (len(points)):
        D = [euclidean(points[i],centroid) for centroid in centroids] #find euclidean distance between point i and each centroid
        cluster=np.argmin(D) #store index of minimum distance of point i from a cluster centroid
        Dict[cluster+1].append(points[i]) #add point i to the cluster corresponding to maximum distance


# In[5]:


def getcentroid(k): #calculate new centroid of a cluster with key 'k'
    sumx=0
    sumy=0
    points_id = Dict[k] #iterate through values in k cluster
    for point in points_id:
        sumx+=point[0] #sum of x coordinate values
        sumy+=point[1] #sum of y coordinate values
    x=sumx/len(points_id) 
    y=sumy/len(points_id)
    A=[x,y] #centroid point
    return A


# In[6]:


def kmeans (strategy):
    objfunc=[] #list to store values of objective function for each value of i (number of clusters)
    for i in range (2,11):
        global Dict
        Dict={}
        obj=0 #values of objective function for each value of i (number of clusters)
        centroids=[]
        if(strategy==1): #random selection of i cluster centroids
            idxes = random.sample(set(range(len(points))),i)
            centroids = points[idxes]
        elif(strategy==2): #selection of ith centroid based on avg distance of points from already selected centroids
            distance=[] 
            centroids=[]
            chosen_points = [] #store index of already selected centroids to avoid repetition
            first_idx = random.randint(0,300) #first centroid chosen at random
            chosen_points.append(first_idx)
            centroids.append(points[first_idx])
            selected=1 #store number of already selected clusters
            while(selected < i): #until required number of clusters have been selected
                distance = []
                for f in range (len(points)): 
                    D=0
                    for s in range (selected):
                        D = D + euclidean(points[f],centroids[s]) #total distance of point f from all selected centroids
                    avg=D/selected #avg distance of point f from selected centroids
                    distance.append((avg,f)) #add to list, avg distance for point f, and the index f
                distance = sorted(distance,key=lambda x:x[0],reverse=True) #sort in descending order
                for dist in distance: #check for max distance 
                    if dist[1] not in chosen_points: #check if index of max distance is already chosen
                        index = dist[1]
                        centroids.append(points[index]) #add to centroids if not already added
                        chosen_points.append(index)
                        break #break as soon as max unrepeated index has been found
                selected+=1 #increment selectedd by 1 as one more centroid has been selected
        newarray=centroids[:] 
        oldarray=[]
        updatedict(newarray) #update dictionary according to selected centroids
        while True:
            oldarray=newarray[:]
            newarray=[]
            for key in Dict:
                c=getcentroid(key) #get new centroids for each cluster
                newarray.append(np.array(c)) #update newarray to store new centroids
            updatedict(newarray) #update dictionary according to newly selected centroids
            if np.array_equal(np.array(newarray),np.array(oldarray)): #if centroids don't change, convergence reached
                break #break when convergence reached
        for k in Dict:
            p= Dict[k] 
            for point in p:
                y=newarray[k-1]-point 
                obj=np.square(y[0])+np.square(y[1])+obj #find objective function  for each number of clusters

        objfunc.append(obj) #store value of objective function 
        
    x=[] #stores number of clusters
    for i in range (2,11):
        x.append(i)
    plt.plot(x,objfunc) #plot objective function value vs number of clusters k
    plt.xlabel("Number of clusters k")
    plt.ylabel("Objective function value")
    plt.show()
    


# In[7]:


kmeans(1) #for strategy 1 of selecting initial cluster centers


# In[8]:


kmeans(2) #for strategy 2 of selecting initial cluster centers


# In[ ]:




