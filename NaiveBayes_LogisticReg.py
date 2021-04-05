#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import math
import numpy as np
data=scipy.io.loadmat('mnist_data.mat')


# In[2]:


trX=data['trX'] #store training data
trY=data['trY'] #store labels of training data
tsX=data['tsX'] #store testing data
tsY=data['tsY'] #store labels of testing data


# In[3]:


meantrX=np.mean(trX,axis=1) #store mean of pixel values of images in training data
sdtrX=np.std(trX,axis=1) #store standard deviation of pixel values of images in training data
meantsX=np.mean(tsX,axis=1) #store mean of pixel values of images in testing data
sdtsX=np.std(tsX,axis=1) #store standard deviation of pixel values of images in testing data

tr1f1=[] #store mean (feature 1) of images labeled 1
tr0f1=[] #store mean of images labeled 0
tr1f2=[] #store standard deviation (feature 2) of images labeled 1
tr0f2=[] #store standard deviation of images labeled 0

for i in range (12116): #check label assigned to each image in training data
    if trY[0][i]==0: 
        tr0f1.append(meantrX[i])
        tr0f2.append(sdtrX[i])
    else:
        tr1f1.append(meantrX[i])
        tr1f2.append(sdtrX[i])


tr0f1mean=np.mean(tr0f1) #mean of feature 1, class 0
tr0f1sd=np.std(tr0f1) #standard deviation of feature 1, class 0

tr0f2mean=np.mean(tr0f2) #mean of feature 2, class 0
tr0f2sd=np.std(tr0f2) #standard deviation of feature 2, class 0

tr1f1mean=np.mean(tr1f1) #mean of feature 1, class 1
tr1f1sd=np.std(tr1f1) #standard deviation of feature 1, class 1

tr1f2mean=np.mean(tr1f2) #mean of feature 2, class 0
tr1f2sd=np.std(tr1f2) #standard deviation of feature 2, class 0

print(tr0f1mean)
print(tr0f1sd)
print(tr0f2mean)
print(tr0f2sd)
print(tr1f1mean)
print(tr1f1sd)
print(tr1f2mean)
print(tr1f2sd)


# In[4]:


def normaldist(x, m, sd): #function for normal distribution
    p1= 1/((np.sqrt(2*math.pi))*sd)
    p2=-(x-m)**2
    p3=2*sd*sd
    p4=p1*np.exp(p2/p3)
    return p4


# In[5]:


count0=0; #count total 0 labels in training data
for i in range (12116):
    if trY[0][i]==0:
        count0=count0+1

py0=count0/12116 #P(Y=0)
py1=1-py0 #P(Y=1)

test= []

# x1=feature 1
# x2=feature 2

for i in range (2002):
    px1y0=normaldist(meantsX[i],tr0f1mean,tr0f1sd) # P(X1|Y=0)
    px2y0=normaldist(sdtsX[i],tr0f2mean,tr0f2sd) # P(X2|Y=0)
    px1y1=normaldist(meantsX[i],tr1f1mean,tr1f1sd) # P(X1|Y=1)
    px2y1=normaldist(sdtsX[i],tr1f2mean,tr1f2sd) # P(X2|Y=1)
    p0=px1y0*px2y0*py0 #P(Y=0|X)=P(X1|Y=0)P(X2|Y=0)P(Y=0)
    p1=px1y1*px2y1*py1 #P(Y=1|X)=P(X1|Y=1)P(X2|Y=1)P(Y=1)
    if p0>p1: #If P(Y=0|X) if greater than P(Y=1|X), image will be classified as 0, else 1
        test.append(0)
    else:
        test.append(1)
    
c=0 #counter
for i in range (2002): #check if predicted labels of testing data match given labels
    if tsY[0][i]==test[i]:
        c=c+1

print(c/2002) #classification accuracy


# In[6]:


def sigmoid(w, x): #logistic sigmoid function
    return 1 / (1 + np.exp(-np.dot(x, w)))


# In[7]:


def classes(w,x): #assigns a class to the image based on results from sigmoid function
    value = sigmoid(w, x)
    return [1 if v >= 0.5 else 0 for v in value]


# In[8]:


def loglikelihood(w,x,y):
    y_ = sigmoid(w, x)
    ll = np.sum(y * np.log(y_) + (1-y) * np.log(1-y_)) / len(x)
    return ll


# In[9]:


def gradient(x,y,prediction): #calculate gradient of loglikelihood
    grad = np.dot(x.T, y - prediction) / len(x)
    return grad


# In[10]:


def gradientascent(x, y, w): 
    for i in range(10000):
        ll = loglikelihood(w, x, y)
        prediction = classes(w,x) #predicted classes of images
        grad = gradient(x, y, prediction)
        w += 0.01 * grad #learning rate is 0.01
    return w


# In[11]:


w = [1, 1, 1]
w = gradientascent(np.c_[meantrX, sdtrX, np.ones(12116)],trY[0],w)


# In[12]:


print(w) #this is the optimal w


# In[13]:


val=classes(w,np.c_[meantsX, sdtsX, np.ones(2002)]) #assign classes to testing images with optimal w


# In[14]:


c=0
for i in range (2002):
    if val[i]==tsY[0][i]: #compare predicted classes with actual classes
        c=c+1
print(c/2002) #classification accuracy
    

