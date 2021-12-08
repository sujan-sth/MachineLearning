#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 6 Task 1

import sys
import numpy as np
import random
from collections import *

#Check whether the dimension is 1D or 2D
def Check1D2D(train_file):
    if isinstance(train_file[0],float):
        return True
    else:
        return False

#Mean for each cluster
def meanCalculator(cluster_dict,Dimension_check):
    mean={}
    for i in cluster_dict:
        sum=0
        for j in cluster_dict[i]:
            sum=sum+j
        if(Dimension_check):
            mean[i]=(sum/len(cluster_dict[i]))
        else:
            mean[i]=list(sum/len(cluster_dict[i]))
    return mean
    
#Creating cluster
def Cluster_Creation(train_file,Num_Cluster,Initialization,cluster_dict):
    for values in range(1,len(train_file)+1):
        if Initialization=="random":    
            val=random.choice(range(1,(Num_Cluster+1)))
            cluster_dict[val].append(train_file[values-1])#.tolist())
        elif Initialization=="round-robin":
            val=values%Num_Cluster
            if val==0:
                cluster_dict[Num_Cluster].append(train_file[values-1])#.tolist())
            else:
                cluster_dict[val].append(train_file[values-1])#.tolist())
    return cluster_dict

train_file=np.loadtxt(sys.argv[1],dtype=float)
Num_Cluster=int(sys.argv[2])
Initialization =(sys.argv[3])
cluster_dict=defaultdict(list)
Dimension_check=Check1D2D(train_file)

#Calling function that create cluster
cluster_dict=Cluster_Creation(train_file,Num_Cluster,Initialization,cluster_dict)

old_dict={}
new_dict=cluster_dict
while(meanCalculator(old_dict,Dimension_check)!=meanCalculator(new_dict,Dimension_check)):
    old_dict=new_dict
    new_dict=defaultdict(list)
    mean=meanCalculator(old_dict,Dimension_check)
    for i in old_dict:
        for j in range(len(old_dict[i])):
                dumy_dict={}
                for x in range(1,len(mean)+1):
                    if (Dimension_check):
                        dumy_dict[x]=abs((old_dict[i][j]-mean[x]))
                    else:
                        dumy_dict[x]=((old_dict[i][j][0]-mean[x][0])**2+(old_dict[i][j][1]-mean[x][1])**2)**0.5
                new_dict[min(dumy_dict,key=dumy_dict.get)].append(old_dict[i][j])   
#OUTPUT 
if(Dimension_check):
    for i in new_dict:
        for j in range(len(new_dict[i])):
            print('%10.4f --> cluster %d\n'%(new_dict[i][j], i))
else:
    for i in new_dict:
        for j in range(len(new_dict[i])):
            print('(%10.4f, %10.4f) --> cluster %d\n'%(new_dict[i][j][0], new_dict[i][j][1], i))