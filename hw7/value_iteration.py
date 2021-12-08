#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 7 Task 1

import copy
import sys
import numpy as np

#Reading and Storing the values from command line argument
envFile=np.loadtxt(sys.argv[1],dtype=str,delimiter=",")
nonTerminalReward=np.double(sys.argv[2])
gamma=np.double(sys.argv[3])
k=int(sys.argv[4])

def initialList(envFile):
    utilityTable=[]
    newL=[]
    for x in range(len(envFile)+2):
        newL.append([0,0,0,0])   #outer layer covering all the accessible from the given elements
    utilityTable.append(newL)
    for i in range(len(envFile)):
        subUtilityTable=[]
        subUtilityTable.append([0,0,0,0])  #outer layer covering the provided value
        for j in range(len(envFile[i])):
            if envFile[i][j]=='.':
                    subUtilityTable.append([0,1,nonTerminalReward,0])  # [utilityValue, can be accessed or not, reward_value, Terminal or not]
            elif envFile[i][j]=='X':
                subUtilityTable.append([0,0,0,0])
            else:
                subUtilityTable.append([0,1,float(envFile[i][j]),1])
        subUtilityTable.append([0,0,0,0])
        utilityTable.append(subUtilityTable)
    utilityTable.append(newL)
    return utilityTable

def actions_max(uT,i,j):
    leftVal=0
    rightVal=0
    upVal=0
    collect=[]
     
    #FOR LEFT CASE
    if uT[i][j-1][1]: #0.8 probability of going to left
        leftVal+= 0.8*uT[i][j-1][0]
    if not(uT[i][j-1][1]): # if cannot go to left stays in the same place
        leftVal+=0.8*uT[i][j][0]
    
    if uT[i+1][j][1]: #0.1 probability of going down
        leftVal+=0.1*uT[i+1][j][0]
    if not(uT[i+1][j][1]): #if going down is not possible in left case
        leftVal += 0.1*uT[i][j][0]
 
    if uT[i-1][j][1]: #0.1 probability of going up
        leftVal+=0.1*uT[i-1][j][0]
    if not(uT[i-1][j][1]):  #if going up as up box is not accessible
        leftVal +=0.1*uT[i][j][0]
    collect.append(leftVal)

    #FOR RIGHT CASE
    if uT[i][j+1][1]: #0.8 probability of going  to right
        rightVal+= 0.8*uT[i][j+1][0]
    if not(uT[i][j+1][1]): # if cannot go to right stays in the same place
        rightVal+=0.8*uT[i][j][0]
    
    if uT[i+1][j][1]: #0.1 probability of going down
        rightVal+=0.1*uT[i+1][j][0]
    if not(uT[i+1][j][1]): #if going down is not possible in left case
        rightVal+= 0.1*uT[i][j][0]

    if uT[i-1][j][1]: #0.1 probability of going up
        rightVal+=0.1*uT[i-1][j][0]
    if not(uT[i-1][j][1]):  #if going up as up box is not accessible
        rightVal+=0.1*uT[i][j][0]
    collect.append(rightVal)


    #FOR UP CASE
    if uT[i-1][j][1]: #0.8 probability of going up
        upVal+= 0.8*uT[i-1][j][0]
    if not(uT[i-1][j][1]): # if cannot go to left stays in the same place
        upVal+= 0.8*uT[i][j][0]
    
    if uT[i][j-1][1]: #0.1 probability of going left
        upVal+= 0.1*uT[i][j-1][0]
    if not(uT[i][j-1][1]): #if going left is not possible in up case
        upVal+= 0.1*uT[i][j][0]

    if uT[i][j+1][1]: #0.1 probability of going right
        upVal+=0.1*uT[i][j+1][0]
    if not(uT[i][j+1][1]):  #if going right as right box is not accessible
        upVal+=0.1*uT[i][j][0]
    collect.append(upVal)
    
    downVal=0
    #FOR DOWN CASE
    if uT[i+1][j][1]: #0.8 probability of going down
        downVal += 0.8*uT[i+1][j][0]
    if not(uT[i+1][j][1]): #if going down is not possible
        downVal += 0.8*uT[i][j][0]

    if uT[i][j+1][1]: #0.1 probability of going right
        downVal += 0.1*uT[i][j+1][0]
    if not(uT[i][j+1][1]):
        downVal += 0.1*uT[i][j][0]

    if uT[i][j-1][1]: #0.1 probability of going left
        downVal += 0.1*uT[i][j-1][0]
    if not(uT[i][j-1][1]):
        downVal += 0.1*uT[i][j][0]

    collect.append(downVal)
    maxVal=max(collect)
    ind=collect.index(maxVal)
    return maxVal,ind
indValue=[]
def valueIteration():
    u_Dash=initialList(envFile)
    for x in range(k):
        u=copy.deepcopy(u_Dash)
        for i in range(1,len(u_Dash)-1):
            for j in range(1,len(u_Dash[i])-1):
                if not(u_Dash[i][j][1]) or u_Dash[i][j][3]:
                    u_Dash[i][j][0]=u_Dash[i][j][2]
                    if not(u_Dash[i][j][1]):
                        indValue.append("X")
                    else:
                        indValue.append("o")
                else:
                    maximumValue ,ind= actions_max(u,i,j)
                    u_Dash[i][j][0]=u_Dash[i][j][2]+maximumValue*gamma
                    if ind==0:
                        indValue.append("<")
                    elif ind==1:
                        indValue.append(">")
                    elif ind==2:
                        indValue.append("^")
                    else:
                        ind==3
                        indValue.append("v")
    return(u_Dash)

iteratedValue=valueIteration()
Dimension=len(envFile)*len(envFile[0])
policyValue=indValue[-Dimension:]
print("\nutilities:")
for i in range (1,len(iteratedValue)-1):
    for j in range (1, len(iteratedValue[i])-1):
        print(' {:6.3f}'.format(iteratedValue[i][j][0]),end="")
    print()

print("\npolicy:")
col=len(envFile[0])
print('     ',end="")
print ("\n     ".join(["      ".join(policyValue[i:i+col]) for i in range(0,len(policyValue),col)]))

