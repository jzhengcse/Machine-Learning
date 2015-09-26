import csv
import calendar
import numpy as np
import matplotlib.pyplot as plt
import string
import collections
from collections import OrderedDict
from sklearn import linear_model
from sklearn import preprocessing
from numpy import genfromtxt
import pandas as pd
from sklearn import tree

def bikes():
    # train model
    with open('train.csv', 'rb') as f:
        training_examples = []
        csv_f = csv.reader(f)
        feature_names = csv_f.next()
        feature_names=["year","month","weekday","hour","season","holiday","workingday","weather","temp","atemp","humidity","windspeed","casual","registered","count"]
        size=len(feature_names)
        for row in csv_f:
            training_examples.append(row)
    data=np.array(training_examples)
    # process data
    X=processData(data)
    yindex=-1
    y=data[:,yindex]
    plotAll(X,y,feature_names,yindex)
    clf=tree.DecisionTreeRegressor()
    clf2=tree.DecisionTreeRegressor()
    #clf=linear_model.SGDRegressor()
    newX=genFeature(X)
   
    level2=np.zeros((np.size(y),2))
    y=data[:,-3]
    y=np.array(y,np.float)
    print "size of x: %s" %np.size(X,1)
    clf.fit(newX,y)
    
    # print and plot error 
    #OriRe=clf.decision_function(X) 
    level2[:,0]=clf.predict(newX) 

    y=data[:,-2]
    y=np.array(y,np.float)
    clf.fit(newX,y)
    level2[:,1]=clf.predict(newX)

    
    clf2.fit(level2,data[:,-1])
    print data[:,(-3,-2)]
    OriRe=clf2.predict(level2)

    error=np.subtract(OriRe,y)
    meanSquareError=np.dot(error,error)/np.size(error)
    print meanSquareError
    count=range(0,np.size(OriRe))
    plotData(count,error,"data point","error")   
    for i in range(12):
        x=X[:,i]
        plotData(x,error,feature_names[i],"error")   
    
    
   
    # test data
    data=[]
    with open('test.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            data.append(row)
    data=np.array(data)
    time=data[:,0]
    time=np.copy(time)
    # process and clean data
    X=processData(data)
    X=genFeature(X)
    #re=clf.decision_function(X)
    level2=np.zeros((np.size(X,0),2))
    level2[:,0]=clf.predict(X) 
    level2[:,1]=clf.predict(X)
    re=clf2.predict(level2)
    #kzero=np.zeros(np.size(re))
    #re=np.maximum(re,kzero)

    with open('sampleSubmission.csv','wb') as f:
        writer=csv.writer(f)
        writer.writerow(("datetime","count"))
        for i in range(np.size(time)):
            writer.writerow(("%s"%(time[i]),"%.2f"%re[i]))
def plotAll(X,y,feature_names,yindex):
    # plot count against category data
    for i in range(0,8):
        x=X[:,i]
        od=dealWithCategory(x,y)
        plotData(od.keys(),od.values(),feature_names[i],feature_names[yindex])
    # plot count against quantitative data
    for i in range(8,12):
        x=X[:,i]
        plotData(x,y,feature_names[i],feature_names[yindex])   
    plt.close('all')
    
def processData(data):
    newFea=processTimes(data[:,0]);
    data=np.hstack((newFea,data))
    X=cleanData(data) 
    return X

def cleanData(data):
    X=data[:,0:13]
    X=np.delete(X,4,1)  #delete datatime
    #X=np.delete(X,8,1)  #delete atemp
    X=np.array(X,np.float)
    return X
def standardize(X):
    #normalizer = preprocessing.Normalizer().fit(X)
    #X=normalizer.transform(X)
    scaler=preprocessing.StandardScaler().fit(X)
    X=scaler.transform(X)
    return X
def genFeature(X):
    X=standardize(X)
    newX=X
    len=np.size(X,1)
    for i in range(0,len):
        for j in range(i+1,len):
            newFea=(X[:,i])*(X[:,j])
            newX=np.hstack((newX,newFea[:,None]))
    newX=standardize(newX)
    for i in range(0,len):
        for j in range(i+1,len):
            for k in range(j+1,len):
                newFea=X[:,i]*X[:,j]*[:,k]
                newX=np.hstack((newX,newFea[:,None]))
    newX=standardize(newX)
    return newX
def processTimes(times):
    newFea=np.zeros((np.size(times),4))
    for i in range(np.size(times)):
        newFea[i,:]=processTime(times[i])
    return newFea
        
def processTime(time):
    newFea=np.zeros(4)
    indexYear=time.index("-")
    newFea[0]=float(time[0:indexYear])
    indexMonth=time.index("-",indexYear+1)
    newFea[1]=float(time[indexYear+1:indexMonth])
    date=float(time[indexMonth+1:indexMonth+3])
    weekday=calendar.weekday(int(newFea[0]),int(newFea[1]),int(date))+1
    newFea[2]=weekday
    indexHour=time.index(":")
    newFea[3]=float(time[indexHour-2:indexHour])
    return newFea
def dealWithCategory(x,y):
    category=set(x)
    size=len(category)
    dict={}
    dict2={}
    for i in category:
        dict[i]=0
        dict2[i]=0
    for i in range(len(y)):
        dict[x[i]]+=float(y[i])
        dict2[x[i]]+=1
    # get avg of each category data
    for i in category:
        dict[i]/=dict2[i]
    
    return collections.OrderedDict(sorted(dict.items()))
def plotData(X,y,xtitle,ytitle):
    fig=plt.figure()
    plt.scatter(X,y,s=20,marker='+',c='r')
    title=ytitle+'VS'+xtitle
    fig.suptitle(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(0)
    fig.savefig(title+'.jpg')


if __name__ == "__main__":    
    bikes()
