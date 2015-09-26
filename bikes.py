import csv
import numpy as np
import matplotlib.pyplot as plt
import string
import collections
from collections import OrderedDict
def bikes():
    with open('train.csv', 'rb') as f:
        training_examples = []
        csv_f = csv.reader(f)
        feature_names = csv_f.next()
        size=len(feature_names)
        for row in csv_f:
            training_examples.append(row)
        print "Feature names:"
        print feature_names

        print "\n\nSome training examples:"
        #for i in xrange(20):
        #    print training_examples[i]
        data=np.array(training_examples)
        for i in range(np.size(data,0)):
            data[i,0]=processTime(data[i,0]);
        #for i in xrange(20):
        #    print data[i,0]
        for i in range(0,5):
            x=data[:,i]
            y=data[:,-1]
            dealWithCategory(x,y)
        for i in range(0,9):
            x=data[:,i]
            y=data[:,-1]
            plotData(x,y,feature_names[i],feature_names[-1])        
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
        dict[x[i]]+=int(y[i])
        dict2[x[i]]+=int(1)

    for i in category:
        dict[i]/=dict2[i]
    print collections.OrderedDict(sorted(dict.items()))
data=[]
global a
def descent():
    with open('ex1data1.csv','rb') as f:
        csv_f=csv.reader(f)
        for line in csv_f:
            data.append(line)
        for i in xrange(20):
            print data[i];
def processData():
        a= np.genfromtxt("ex1data1.csv",delimiter=",")
        a=np.mat(a)
        X=a[:,0]
        y=a[:,1]
        m=np.size(X)
        plotData(X,y)
        listA=np.ones((m,1))
        X=np.hstack((listA,X))
        theta=np.zeros((2,1));
        iterations=1500
        alpha=.01
        print computeCost(X,y,theta)

        
        theta=gradientDescent(X,y,theta,alpha,iterations)
        print theta

        plt.plot(X[:,1],X*theta)
        #plt.show()
        predict1=np.array([1,3.5])*theta;
        predict2=np.array([1,7])*theta;
        print predict1;
        print predict2;
def processData2():
     a= np.genfromtxt("ex1data2.txt",delimiter=",")
     a=np.mat(a)
     X=a[:,0:2]
     y=a[:,2]
     m=np.size(X)
     X, mu, sigma=featureNormalize(X);
     listA=np.ones((m,1))
     listA.resize(X.shape)
     X=np.hstack((listA,X))
     X=X[:,[0,2,3]];
    
     alpha=.01
     num_iters=400
     theta=np.zeros((3,1));
   #  print computeCost(X,y,theta)


     theta=gradientDescent(X,y,theta,alpha,num_iters)
     print theta

   #  plt.plot(X[:,1],X*theta)
   #  #plt.show()
   #  predict1=np.array([1,3.5])*theta;
   #  predict2=np.array([1,7])*theta;
   #  print predict1;
   #  print predict2;

def featureNormalize(X):
    X_norm=X;
    xdim2=np.size(X,1);
    print xdim2
    mu=np.zeros((1,xdim2));
    sigma=np.zeros((1,xdim2));
    for i in range(xdim2):
        mu[0,i]=np.mean(X[:,i]);
        sigma[0,i]=np.std(X[:,i]);
        X_norm[:,i]=(X[:,i]-mu[0,i])/sigma[0,i];
    return (X,mu,sigma)

def plotData(X,y,xtitle,ytitle):
    fig=plt.figure()
    plt.scatter(X,y,s=20,marker='+',c='r')
    title=ytitle+'VS'+xtitle
    fig.suptitle(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    fig.savefig(title+'.jpg')

def computeCost(X,y,theta):
    m=np.size(y)
    h=X*theta
    J=1.0/(2*m)*(h-y).T*(h-y)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m=np.size(y)
    J_history=np.zeros((num_iters,1))
    for i in range(num_iters):
        k=1.0/m*X.T*(X*theta-y);
        theta=theta-alpha*k;
        J_history[i]=computeCost(X,y,theta)
    return theta
#a=np.array(training_examples);
#w=a[:,5];
#w=list(w);
#w=map(string.atof,w);
#c=a[:,-1];
#c=list(c);
#c=map(string.atof,c);
#plt.plot(w,c);

if __name__ == "__main__":    
    bikes()
    #processData2()
