#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 3 Task 1

import sys
import numpy as np
import math
import random

def matrix_creator(a,degree):
    degree_matrix=[]
    for value in range(len(a)):
        row=[]
        row.append(1)
        for values in a[value]:
            for num in range(degree):
                if degree==1:
                    row.append(values)
                else:
                    row.append(values**(num+1))
        degree_matrix.append(row)
    return degree_matrix

a=np.loadtxt(sys.argv[1])
list_of_labels=[values[-1] for values in a]
a=[value[:-1] for value in a]
degree=int(sys.argv[3])
lamda=int(sys.argv[4])
DM_train=np.array(matrix_creator(a,degree))
idt=np.identity(len(DM_train[0]))
weight=np.matmul(np.matmul(np.linalg.pinv(lamda*idt+np.matmul(np.array(DM_train).T,DM_train)),np.array(DM_train).T),list_of_labels)
#print(weight)
testData=np.loadtxt(sys.argv[2])
test_label=[lab[-1] for lab in testData]
testData=[vall[:-1] for vall in testData]
DM_test=matrix_creator(testData,degree)
for x in range(len(weight)):
    print('w'+str(x)+"=%.4f"%weight[x])
for i in range(len(DM_test)):
    val=np.matmul(DM_test[i],weight)
    
    squared_error=(val-test_label[i])**2
    print("ID=%5d,"%(i+1), "output=%14.4f,"%val, "target value = %10.4f,"%test_label[i], "squared error = %.4f"%squared_error)
#print(np.matmul(DM_test[0],weight))

