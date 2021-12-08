#Name:Sujan Shrestha
#ID:1001752468
#OS: Unix
#Assignment 1 Task 9

import os
def file_stats(pathname):
    fileRead=open(pathname,"r+")
    listofFloats=[]
    for line in fileRead.readlines():
        listofFloats.append(float(line))
    #number of values in file
    length=len(listofFloats)
    #using formula to calculate the average
    average=sum(listofFloats)/length
    #Using formula of standard deviation to calculate the value
    SD=(sum((values-average)**2 for values in listofFloats)/(length-1))**(1/2)
    return average,SD

pathname=os.path.join("numbers1.txt")        
(avg,stdev)=file_stats(pathname)
#print(file_stats(pathname))
