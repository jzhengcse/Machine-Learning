import csv
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
        size=len(feature_names)
        for row in csv_f:
            training_examples.append(row)
    data=np.array(training_examples)
    # change from datatime to hour
    data[:,0]=processTimes(data[:,0]);
    data=np.array(data,np.float)

    # plot count against category data
    for i in range(0,5):
        x=data[:,i]
        y=data[:,-1]
        od=dealWithCategory(x,y)
        plotData(od.keys(),od.values(),feature_names[i],feature_names[-1])
    # plot count against quantitative data
    for i in range(5,9):
        x=data[:,i]
        y=data[:,-1]
        plotData(x,y,feature_names[i],feature_names[-1])   
    
    X=cleanData(data) 
    clf=tree.DecisionTreeRegressor()
    #clf=linear_model.SGDRegressor()
    X=genFeature(X)
    
    y=data[:,-1]
    print "size of x: %s" %np.size(X,1)
    clf.fit(X,y)
    
    # print and plot error 
    #OriRe=clf.decision_function(X) 
    OriRe=clf.predict(X) 
    OriRe=np.subtract(OriRe,y)
    #for i in range(300):
    #    print OriRe[i]
    count=range(0,np.size(OriRe))
    plotData(count,OriRe,"data point","error")   
    
   
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
    # change from datatime to hour
    data[:,0]=processTimes(data[:,0]);
    data=np.array(data,np.float)
    X=cleanData(data)
    X=genFeature(X)
    #re=clf.decision_function(X)
    re=clf.predict(X)
    kzero=np.zeros(np.size(re))
    re=np.maximum(re,kzero)

    with open('sampleSubmission.csv','wb') as f:
        writer=csv.writer(f)
        writer.writerow(("datetime","count"))
        for i in range(np.size(time)):
            writer.writerow(("%s"%(time[i]),"%.2f"%re[i]))
def cleanData(data):
    X=data[:,0:9]
    X=np.delete(X,6,1)
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
    #normalizer = preprocessing.Normalizer().fit(X)
    #X=normalizer.transform(X)
    newX=standardize(newX)
    return newX
def processTimes(times):
    for i in range(np.size(times)):
        times[i]=processTime(times[i])

    return times
        
def processTime(time):
    index=time.index(":")
    return time[index-2:index]
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

