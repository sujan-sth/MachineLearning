#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 4 Task 1

import sys
import numpy as np
import random
import math
import time

def hotVector(labels,list_of_labels):
    dict={}
    hotVect=[]
    for x in range (0,len(labels)):
        dict[labels[x]]=x+1
        
    for values in range(0,len(list_of_labels)):
        InList=[]
        for key in dict:
            if list_of_labels[values]==key:
                InList.append(1)
            else:
                InList.append(0)
        hotVect.append(InList)
    return hotVect 

def prop(BiasW,Weight,row,labels,hot,ita):
   Z=[]
   Z.append(list(row))
   a=[]
   for x in range(len(BiasW)):
       a1=[]
       z1=[]
       for y in range(len(BiasW[x])):
            sum=0
            for g in range(len(Weight[x][y])):
                #print(Z[x][g])
                sum=sum+Z[x][g]*Weight[x][y][g]   
            a1.append(BiasW[x][y]+sum)
            z1.append(1/(1+np.exp(-a1[y])))
       a.append(a1)    
       Z.append(z1)

   DeltaValue=[]
   for i in range(len(labels)):
       DeltaValue.append((Z[-1][i]-hot[i])*Z[-1][i]*(1-Z[-1][i]))
   for val in range(len(Z)-2,0,-1):
       newDel=[]
       for a in range(len(Z[val])):
           DelSum=0
           for  b in range(len(Z[val+1])):
               DelSum=DelSum+DeltaValue[0][b]*Weight[val][b][a]
           newDel.append(DelSum*Z[val][a]*(1-Z[val][a]))
       DeltaValue.insert(0,newDel)
   for x in range(0,len(BiasW)):
       for y in range(len(BiasW[x])):
           BiasW[x][y]=BiasW[x][y]-ita*DeltaValue[x][y]
           for z in range(len(Weight[x][y])):
               Weight[x][y][z]=Weight[x][y][z]-ita*DeltaValue[x][y]*Z[x][z]
   return BiasW,Weight             
train_file=np.loadtxt(sys.argv[1],dtype=str)
layers = int(sys.argv[3])
units_per_layer = int(sys.argv[4])
rounds = int(sys.argv[5])
list_of_labels=[values[-1] for values in train_file]
labels=np.unique(list_of_labels)
#print(len(labels))
hot=hotVector(labels,list_of_labels)
train_file1=[value[:-1] for value in train_file]
train_file1=np.array(train_file1).astype(float)
maxValue=max(np.array(train_file1).min(),np.array(train_file1).max(),key=abs)
normalize_train=train_file1/maxValue


BiasW=[]
Weight=[]
for valu in range (layers-2):
    b=[]
    weig=[]
    for y in range(units_per_layer):
        b.append(np.random.uniform(-0.5,0.5))
        w=[]
        if(y==0):
            for value in range(len(train_file1[0])):
                w.append(np.random.uniform(-0.5,0.5))
        else:
            for value in range(units_per_layer):
                w.append(np.random.uniform(-0.5,0.5))
        weig.append(w)
    BiasW.append(b)
    Weight.append(weig)
lastBias=[]
lastWeight=[]
for x in range(len(labels)):
    lastBias.append(np.random.uniform(-0.5,0.5))
    wVal=[]
    if layers>2:
        for x in range(units_per_layer):
            wVal.append(np.random.uniform(-0.5,0.5))
    else:
        for y in range(len(train_file1[0])):
            wVal.append(np.random.uniform(-0.5,0.5))
    lastWeight.append(wVal)
BiasW.append(lastBias)
Weight.append(lastWeight)

ita=1
for y in range(rounds):
    for i in range(len(normalize_train)):
        BiasW,Weight=prop(BiasW,Weight,normalize_train[i],labels,hot,ita)
    ita=ita*0.98
print(ita)

#test files
test_file=np.loadtxt(sys.argv[2],dtype=str)
test_file1=[value[:-1] for value in test_file]
test_file1=np.array(test_file1).astype(float)
testLabels=[values[-1] for values in test_file]
maxValu=max(np.array(test_file1).min(),np.array(test_file1).max(),key=abs)
normalize_test=test_file1/maxValu

counter=0
for i in range(len(normalize_test)):
    Z_test=[]
    aVal=[]
    Z_test.append(normalize_test[i])
    for x in range(len(BiasW)):
        a1Val=[]
        z1Val=[]
        for y in range(len(BiasW[x])):
            summing=0
            for g in range(len(Weight[x][y])):
                summing=summing+Z_test[x][g]*Weight[x][y][g]   
            a1Val.append(BiasW[x][y]+summing)
            z1Val.append(1/(1+np.exp(-a1Val[y])))
        aVal.append(a1Val)    
        Z_test.append(z1Val)
        checkif=0
        valuing=list_of_labels[Z_test[-1].index(max(Z_test[-1]))]
        if testLabels[i]==valuing:   
            counter=(counter+1)
            checkif=1
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f'%(i+1, valuing, testLabels[i], checkif))
print('Classification Accuracy=     ',(counter)/len(normalize_test))

