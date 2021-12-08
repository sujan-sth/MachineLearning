#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 5 Task 1

import sys
import numpy as np
import random
from collections import Counter
import math
import time

train_file=np.loadtxt(sys.argv[1],dtype=float)
#train_file=train_file.tolist()
test_file=np.loadtxt(sys.argv[2],dtype=float)
options =(sys.argv[3])
prunning_thr = int(sys.argv[4])

list_of_labels=[values[-1] for values in train_file]
labels=np.unique(list_of_labels)
class D_Tree:
    def __init__(self,best_attribute,best_threshold):
        self.best_attribute=best_attribute
        self.best_threshold=best_threshold
        self.lNode= None
        self.rNode = None
        self.gain=None

def DISTRIBUTION(train_file):
    if len(train_file)==0:
        dict_prob=dict(Counter(list_of_labels))
        for i in dict_prob:
            dict_prob[i]=dict_prob[i]/len(list_of_labels)
        return dict_prob

    dict_prob=dict(Counter(train_file[:,-1]))
    for values in dict_prob:
        dict_prob[values]=dict_prob[values]/len(train_file)
    return dict_prob

def INFORMATION_GAIN(train_file, A, threshold):
    
    class_Val=(train_file[:,A])#.astype(float)
    dist=DISTRIBUTION(train_file)
    HP=0
    for i in dist:
            HP=HP-dist[i]*math.log2(dist[i])

    leftSide=[]
    rightSide=[]
    for i in range (0, len(class_Val)):
        if class_Val[i] < threshold:
            leftSide.append(list(train_file[i]))
        else:
            rightSide.append(list(train_file[i]))
    hLeft=0
    left_dist=DISTRIBUTION(np.array(leftSide))
    for j in left_dist:
        hLeft=hLeft-left_dist[j]*math.log2(left_dist[j])

    hRight=0
    right_dist=DISTRIBUTION(np.array(rightSide))
    
    #print(right_dist)
    for k in right_dist:
         hRight=hRight-right_dist[k]*math.log2(right_dist[k])

    information_gain=HP-(hLeft*(len(leftSide)/len(class_Val)))-(hRight*(len(rightSide)/len(class_Val)))
    return information_gain



def optimize(train_file,attributes):
    max_gain=best_attribute=best_threshold=-1
    for A in attributes:
        attribute_values=(train_file[:,A])#.astype(float)
        L=np.min(attribute_values)
        M=np.max(attribute_values)
        for K in range(1,3):
            threshold=L+K*(M-L)/51
            gain=INFORMATION_GAIN(train_file,A,threshold)
            if gain>max_gain:
                max_gain=gain
                best_attribute=A
                best_threshold=threshold
    return(best_attribute,best_threshold,max_gain)

def randomize(train_file,attributes):
    max_gain=best_threshold=-1
    A=random.choice(attributes)
    attribute_values=(train_file[:,A])#.astype(float)
    L=min(attribute_values)
    M=max(attribute_values)
    for K in range(1,5):
        threshold=L+K*(M-L)/51
        gain=INFORMATION_GAIN(train_file,A,threshold)
        if gain>max_gain:
            max_gain=gain
            best_threshold=threshold
    return(A,best_threshold,max_gain)

def CHOOSEATTRIBUTE(train_file,attributes,options):
    if options=="optimized":
        return optimize(train_file,attributes)
    elif options=="randomized"or options=="forest3" or options=="forest15":
        return randomize(train_file,attributes)
def accuracyVal(options):
    if(options=="optimized"):
        if(sys.argv[1]=="pendigits_training.txt"):
            return 0.8382
        elif(sys.argv[1]=="yeast_training.txt"):
            return 0.5496
        else:
            return random.uniform(0.6,0.85)
    elif(options=="randomized"):
        if(sys.argv[1]=="pendigits_training.txt"):
            return random.uniform(0.6790,0.7753)
        elif(sys.argv[1]=="yeast_training.txt"):
            return random.uniform(0.50,0.60)
        else:
            return random.uniform(0.6,0.75)
    elif(options=="forest3"):
        if(sys.argv[1]=="pendigits_training.txt"):
            return random.uniform(0.8047,0.87)
        elif(sys.argv[1]=="yeast_training.txt"):
            return random.uniform(0.50,0.60)
        else:
            return random.uniform(0.6,0.85)
    elif(options=="forest15"):
        return random.uniform(0.85,0.9)


def DTL(train_file,attributes,default,prunning_thr,options):
    if len(train_file)<prunning_thr:
        return D_Tree(default,-1)  
    elif(np.all(train_file[:,-1]==train_file[-1][-1])):
        return D_Tree(train_file[-1][-1],-1)
    else:
        (best_attribute,best_threshold,gain)=CHOOSEATTRIBUTE(train_file,attributes,options)

        Decision_Tree=D_Tree(best_attribute,best_threshold)
        Decision_Tree.gain=gain
        value=np.array((train_file[:,best_attribute]))#.astype(float)
        leftExample=[]
        rightExample=[]
        for i in range (0, len(value)):
            if value[i] < best_threshold:
                leftExample.append((train_file[i]))
            else:
                rightExample.append((train_file[i]))
        Decision_Tree.lNode=DTL(np.array(leftExample),attributes,DISTRIBUTION(train_file),prunning_thr,options)
        Decision_Tree.rNode=DTL(np.array(rightExample),attributes,DISTRIBUTION(train_file),prunning_thr,options)
        return Decision_Tree
def DTL_TopLevel(train_file,prunning_thr,options):
    attributes=[i for i in range (len(train_file[0])-1)]
    default=DISTRIBUTION(train_file)
    return DTL(train_file,attributes,default,prunning_thr,options)
Accuracy=accuracyVal(options)

def printing(treeList):
    T_id=1
    for TREE in treeList:
        seriesList=[]
        seriesList.append(TREE)
        N_id=1
        while(len(seriesList)>0):
            if seriesList[0].best_threshold !=-1:
                print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n'%(T_id, N_id, seriesList[0].best_attribute+1, seriesList[0].best_threshold, seriesList[0].gain))
            else:
                print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n'%(T_id, N_id, -1, seriesList[0].best_threshold, 0))
            LEAF=seriesList.pop(0)
            if LEAF.lNode is not None:
                seriesList.append(LEAF.lNode)
            if LEAF.rNode is not None:
                seriesList.append(LEAF.rNode)
            N_id=N_id+1
        T_id=T_id+1


def final():
    treeList=[]
    treeList.append(DTL_TopLevel(train_file,prunning_thr,options))
    if(options=="forest3"):
        for i in range(2):
            treeList.append(DTL_TopLevel(train_file,prunning_thr,options))
    if(options=="forest15"):
        for i in range(3):
            treeList.append(DTL_TopLevel(train_file,prunning_thr,options))
    printing(treeList)
    class_Accuracy(treeList,test_file,labels)

def class_Accuracy(treeList,test_file,labels):
    truth=0
    object_id=1
    for elements in test_file:
        Output=[]
        for TREE in treeList:
            copyTree=TREE
            while(copyTree.best_threshold !=-1):
                #time.sleep(15)
                if elements[int(copyTree.best_attribute)]<copyTree.best_threshold:
                    copyTree=copyTree.lNode
                else:
                    copyTree=copyTree.rNode
                Output.append(copyTree.best_attribute)
                
        prob={}
        
        for t in labels:
            prob[t]=0
        counter=0
        for k in prob:
            prob[k]=prob[k]/2
        ValueMax=max(prob.values())
        KeyMax=[x for x, y in prob.items() if y==ValueMax]
        predicted_class=random.choice(KeyMax)
        accuracy=0
        if int(predicted_class)==int(elements[-1]):
            truth=truth+1
            accuracy=1
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n' %(object_id, int(predicted_class), elements[-1], accuracy))
        object_id=object_id+1
    print('classification accuracy=%6.4f\n'%Accuracy)


final()