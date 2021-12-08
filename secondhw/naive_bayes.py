#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 2 Task 5
import sys
import numpy as np
import math
import random

#Gaussian formula
def Gauss(sd,mean,value): 
    numer=(math.exp(-(math.pow(((value-mean)/((math.sqrt(2))*sd)),2))))
    Denom= (math.sqrt((2*math.pi))*sd)
    return numer/Denom 

trainFile=open(sys.argv[1],"r+")   #reading training file
listofFile=[]
numberOfLines=0 #counting the rows in file

#read the line and strip '\n
for line in trainFile.readlines():
    numberOfLines+=1
    listofFile.append([line.rstrip()])
floatFile=[]

#converting the string into float and sorted as per the class label
for value in listofFile:
    floatFile.append([float(a) for a in value[0].split()])
sortedFloatFile=sorted(floatFile,key=lambda l:l[-1])
classDict={}
classList=[]

# creating dictionary that holds the class as key and its all attributes as list of list
for items in sortedFloatFile:
    if items[-1] in classDict:
        classDict[int(items[-1])].append(items[:-1])
    else:
        classDict[int(items[-1])]=[items[:-1]]
        classList.append(items[-1])

#2 dictionary holding class as key and mean and stadndard deviation for all attributes as list
meanDict={}
SDDict={}
classProbability=dict()    #list that hold prior probability P(Ci)
for classValue in classDict:
    meanofColumn=[]
    SDofColumn=[]
    length=len(classDict[classValue])
    classProbability[classValue]=length/numberOfLines
    for values in range(0,len(classDict[classValue][0])):
        listed=np.array(classDict[classValue])[:,values]
        mean=sum(listed)/length
        SD=(sum((at-mean)**2 for at in listed)/float((length)-1))**(1/2)
        meanofColumn.append(mean)
        SDofColumn.append(SD) 
    meanDict[classValue]=list(meanofColumn)
    SDDict[classValue]=list(SDofColumn)
# loop to print the training data information
for classValue in meanDict:
    for values in range(1,len(meanDict[classValue])+1):
        if SDDict[classValue][values-1]<0.01:
            SDDict[classValue][values-1]=0.01
        print("Class %d,"%classValue, "attribute %d,"%values, "mean = %.2f,"%(meanDict[classValue][values-1]), "std = %.2f"%SDDict[classValue][values-1])
#reading the test file 
testData=np.loadtxt(sys.argv[2])
ln=len(testData)

# finding the P(C|x)
id=0
accuracy_List=[]
for x in testData:
    sum=0
    eachClass=[]
    for label in classList:
        Product=1
        for dataVal in range(len(x[:-1])):
            sd=SDDict[label][dataVal]
            mean=meanDict[label][dataVal]
            value=x[dataVal]
            GaussianValue = Gauss(sd,mean,value) 
            Product=GaussianValue*Product   #P(x|ci)
        eachClass.append(Product*classProbability[int(label)]) 
    sum=math.fsum(eachClass)                #P(x)
    div=[num/sum for num in eachClass]      #P(Cj|x)   
    largest=max(div)
    counter=0
    #if there are multiple maximum number
    if largest in div:
        counter=counter+1
    real=(x[-1])                         #true class  
    #inititating accuracy=0
    acc=0
    #finding indices of maximum value
    indices = [index for index, val in enumerate(div) if val == largest]
    pred=classList[random.choice(indices)]   #choosing random predicted value if there are more than 2 maximum
    #for finfding value of accuracy value
    if pred==real:
        if counter>=1:
            acc=1/counter
    #listing all the value of accuracy
    accuracy_List.append(acc)
    id=id+1
    print("ID=%5d,"%id, "predicted=%3d,"%pred, "probability = %.4f,"%largest, "true=%3d,"%real, "accuracy=%4.2f"%acc)
print("classification accuracy=%6.4f"%np.mean(accuracy_List))
