# COMP 3105 A4
# Alexander Breeze 101 143 291
# Viktor Litzanov  101 143 028

import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sys
from contextlib import redirect_stderr
from sklearn.cluster import KMeans

#takes image of 3 digits, returns 3 images of digit
def split(x):
 d=x.shape[0] #should be 28*28*3, but this handles any size input
 image1=x[:int(d/3)]
 image2=x[int(d/3):int(2*d/3)]
 image3=x[int(2*d/3):]
 return [image1,image2,image3]

#X nxd, y nx1, returns a model
def model(X1,Y1,K,max_iter):
 k=K
 #split all X values and alter Y accordingly
 X=[]
 y=[]
 for x in range(X1.shape[0]):
  X+=split(X1[x]) #splits X into 3 digit images
  y+=[-1,Y1[x],Y1[x]] #label corresponds to either second or third digit
 X=np.array(X)
 y=np.array(y)
 n,d=X.shape

 start = time.time()
 
 '''
 X = whiten(X) #make all axes have same scale
 centroids = kmeans(X, k, iter=max_iter, thresh=1e-05)[0] #run kmeans
 '''
 km = KMeans(K, max_iter=max_iter, tol=1e-05, algorithm="elkan") #initialize scikit-learn kmeans algorithm
 km = km.fit(X, y) #run kmeans on training data
 centroids = km.cluster_centers_ #get groups
 result = vq(X, centroids)[0] #get labels for X using model
 
 f = open("log.txt", "a")
 f.write(f" | KmeansT: {str(time.time() - start)}")
 f.close()
 
 groups = np.zeros((11,k)) #11th group for -1's, delete later
 y = y.reshape((n,1))
 result = result.reshape((n,1))
 pos = np.concatenate((y, result), axis=1).astype(int)
 np.add.at(groups,tuple(pos.T),1)
 
 groups=groups[:-1] #deleting 11th row of groups, aka all -1's
 table=groups.argmax(0) #find most common label in each group
 model={'centroids':centroids,'table':table} #make model to return
 accuracy=np.mean(Y1==classify(X1,model)) #get accuracy of current model on training labels

 bestModel = model
 while True: #prune groups with multiple common labels until max accuracy reached
  bestAcc = accuracy #if found better model, update
  model = bestModel
  
  for i in range(k):
   Kcentroids = np.concatenate((model['centroids'][:i],model['centroids'][i+1:]),axis=0) #remove group i
   result = vq(X, Kcentroids)[0] #plot training data into new groups
   
   groups = np.zeros((11,k)) #11th group for -1's, delete later
   result = result.reshape((n,1))
   pos = np.concatenate((y, result), axis=1).astype(int)
   pos = pos[pos[:,1] < i]
   np.add.at(groups,tuple(pos.T),1)
   
   groups = groups[:-1] #deleting 11th row of groups, aka all -1's
   table  = groups.argmax(0) #find most common label in each group
   table  = table[:-2] #to fix bad index error handler
   Bmodel = {'centroids':Kcentroids,'table':table} #new prospective model
   accuracy = np.mean(Y1==classify(X1,Bmodel)) #get accuracy of new model
   if accuracy > bestAcc: #if new model is best so far, keep it
    bestAcc = accuracy
    bestModel = Bmodel
  if accuracy == bestAcc: #if no group removal increased accuracy, at local minimum, return.
   break
  k -= 1
 return model


#Xtest mxd, model, returns yhat mx1
def classify(Xtest,model):
 #split all X values and alter Y accordingly
 X = []
 for x in range(Xtest.shape[0]):
  X  += split(Xtest[x]) #split each image into 3 digit images
 X    = np.array(X)
 n, d = X.shape

 centroids = model['centroids'] #extract parts of model
 table = model['table']
 result = vq(X,centroids)[0] #sort input into groups

 Y=[]
 for i in range(0,n,3):
  try:
   splitter=table[result[i]] #lookup label for group of first digit
   if splitter<5:
    Y.append(table[result[i+1]]) #return second digit
   else:
    Y.append(table[result[i+2]]) #return third digit
  except:
   Y.append(0) #handle exceptions without losing order
 return np.array(Y)

def processK(Xtrain,Ytrain,Xtest,Ytest, k_cand, itr, results):
 for k in k_cand:
  for i in itr:
   f = open("log.txt", "a")
   f.write(f"K: {str(k)} | MAX_ITER: {str(i)}")
   f.close()
   start = time.time()
   myModel = model(Xtrain, Ytrain, k, i)
   traint = time.time() - start
   accuracy = np.mean(Ytest == classify(Xtest,myModel)) * 100
   if accuracy>results[0]:
    results[0] = accuracy
    results[1] = k
    results[2] = i
   f = open("log.txt", "a")
   f.write(f" | ACCURACY: {str(accuracy)} | TRAIN_T: {str(traint)}s\n")
   f.close()

def plotImg(x):
 img = x.reshape((28, 28))
 plt.imshow(img, cmap='gray')
 plt.show()
 return     

def dataReader(fileName):
 X=[]
 Y=[]
 data = np.loadtxt(fileName, delimiter=",") #read in csv file
 for row in data:
  X.append(row[1:])
  Y.append(row[0])
 X=np.array(X)
 Y=np.array(Y)
 return X,Y

if __name__ == "__main__":

 Xtrain,Ytrain = dataReader("A4train.csv")
 Xtest,Ytest = dataReader("A4test.csv")

 k_cand = range(150,240,10)
 itr = range(15,121,15)
 num_threads = len(k_cand)
 
 results = mp.Array('d', 3)
 
 f = open("log.txt", "w")
 f.write("Results:\n")
 f.close()
 
 processK(Xtrain,Ytrain,Xtest,Ytest,k_cand,itr,results)
   
 f = open("log.txt", "a")
 f.write(f"Finished in {str(end - start)}s.\nOutput:\n{str(results[:])}")