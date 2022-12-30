# COMP 3105 A4
# Alexander Breeze 101 143 291
# Viktor Litzanov  101 143 028

import numpy as np
import scipy as sp
from scipy.cluster.vq import vq
import time
from sklearn.cluster import KMeans

'''
Trainer
'''

def learn(X1,Y1):
 k = 270
 max_iter = 75
 X = X1.reshape((X1.shape[0]*3, int(X1.shape[1]/3))) #separate each image into 3
 Y = np.reshape(Y1, (Y1.shape[0],1)) #convert y to 'matrix'
 y = np.concatenate((np.full((Y1.shape[0],1),-1),Y,Y), axis=1).flatten().reshape((X.shape[0],1)) #[-1,i,i] As per our explanation, label corresponds to either second or third digit

 km = KMeans(k, max_iter=max_iter, tol=1e-06, algorithm="elkan") #initialize scikit-learn kmeans algorithm
 km = km.fit(X, y) #run kmeans on training data
 Kcentroids = km.cluster_centers_ #get groups
 
 accuracy = -1
 bestAcc = -1
 bestModel = None
 s = -1 # flag for initial grouping
 while True: #find most common label for each group, prune groups with multiple common labels until max acc reached
  bestAcc = accuracy #reset best accuracy
  for i in range(s,k):
   result = vq(X, Kcentroids)[0] #labels for X
   
   groups = np.zeros((11,k)) #11th group for -1's, delete later
   result = result.reshape(y.shape) #convert vector to 'matrix'
   pos = np.concatenate((y, result), axis=1).astype(int) #assign label to each image
   if (i > -1): #if not initial run
    pos = pos[pos[:,1] < i] #remove values that are out of bounds
   np.add.at(groups,tuple(pos.T),1) #accumulate the number of each group present
   
   groups = groups[:-1] #deleting 11th row of groups, aka all -1's
   table=groups.argmax(0) #find most common label in each group
   model = {'centroids':Kcentroids, 'table':table} #make model to return
   
   accuracy = np.mean(np.equal(Y1, _cat(X,result,table))) #get accuracy of current model on training labels
   if accuracy > bestAcc: #if new model is best so far, keep it
    bestAcc = accuracy
    bestModel = model
   Kcentroids = np.concatenate((bestModel['centroids'][:i+1],bestModel['centroids'][i+2:]),axis=0) #remove group i
  s = 0 #disable initial run flag
  if accuracy == bestAcc: #if best accuracy has not changed
   return bestModel
  k -= 1 #next k

'''
Classifier
'''

def classify(Xtest,model):
 X = Xtest.reshape((Xtest.shape[0]*3, int(Xtest.shape[1]/3))) #separate each image into 3
 centroids = model['centroids'] #get model centroids
 table = model['table'] #get model table
 result = vq(X,centroids)[0] #sort input to groups
 return _cat(X,result,table) #assign label to digit and return result vector

def _cat(X,result,table):
 Y = np.empty(int(X.shape[0]/3), dtype=int) #fast create output vector
 for i in range(0,X.shape[0],3):
  try:
   if table[result[i]]<5: #lookup label for group of first digit
    Y[int(i/3)] = table[result[i+1]] #if <5 return second digit
   else:
    Y[int(i/3)] = table[result[i+2]] #if >=5 return third digit
  except:
   Y[int(i/3)] = 0 #fallback in case of exception
 return Y
