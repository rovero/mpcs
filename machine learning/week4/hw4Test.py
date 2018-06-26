import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

class ANN():
    h = None # num of hidden layers
    s = None # num of units in hidden layer
    k = None # num of class
    weights = None
    mean = None
    std = None
    def __init__(self,h,s):
        self.h = h #num of hidden layer
        self.s = s #num of nodes in each hidden layer
    # X is m*n matrix, y is m*1 vector, alpha is training parameter
    #t is number of iterations
    def fit(self,X,y,alpha,t):
        m = X.shape[0]  # number of examples
        n = X.shape[1]  # number of features
        #normalization
        self.mean = np.mean(X,0)
        self.std = np.std(X,0)
        for col in range(n):
            X[:, col] = X[:, col] - self.mean[col]
            if self.std[col]!=0:
                X[:, col]= X[:, col] / self.std[col]
        Cost = []
        m = X.shape[0] #number of examples
        n = X.shape[1] #number of features
        #Xavier Initialization
        self.weights = [] #length h+1
        self.weights.append(np.random.uniform(low=-math.sqrt(6/(n+self.s)), high=math.sqrt(6/(n+1+self.s)), size=(self.s, n+1)))
        for i in range(self.h-1):
            self.weights.append(np.random.uniform(low=-math.sqrt(6/(2*self.s)), high=math.sqrt(6/(2*self.s)), size=(self.s, self.s+1)))
        if len(set(y))==2:
            self.k = 1
        elif len(set(y))>=3:
            self.k = len(set(y))
        else:
            print("cannot have less than 2 classes")
            return
        self.k = 10 #For using some sample data to train the network, can comment this if using the whole dataset
        Y = np.array([[0]*self.k]*m)
        for i in range(m):
            Y[i][y[i]]=1
        self.weights.append(np.random.uniform(low=-math.sqrt(6/(self.k+self.s)), high=math.sqrt(6/(self.k+self.s)), size=(self.k, self.s+1)))
        #David's Test statistics
        #a = [[-1.14282988, -0.45253739, -0.16098125, -0.08067616],
        #     [0.22379589, -0.20108552, -0.23830988, -0.10435417],
        #     [1.07941935, 0.23856118, -0.41676105, -0.84241045],
        #     [-1.55562539, -0.48465322, 0.68205681, -0.08747013],
        #     [0.05029518, -0.60252568, 0.61336393, 0.15432761]]
        #b = [[0.28503219, 0.51830356, -0.38052089, -0.11475257, -0.638776, -0.46862776],
        #     [-0.28988183, -0.5736774, -0.07588861, 0.12875949, 0.64619578, -0.63165241],
        #     [-2.15749797, 0.2970735, -0.33831311, -0.52670153, 0.02932394, 0.64146576],
        #     [0.21010699, 0.55335698, 0.40126154, -0.04065573, -0.09815836, 0.52545504]]
        #self.weights.append(np.array(a))
        #self.weights.append(np.array(b))
        for it in range(t):
            for row in range(m):
                #change the form of y value in an example from integer to array
                a = [] #list of matrices of input in each layer
                currA = X[row] #input layer, add bias term
                #for i in range(X.shape[1]): #iterate all columns in an example
                #    currA.append(X.iloc[row][i])
                currA = np.hstack([[1], currA])
                a.append(np.array(currA))
                input = [] # Store input in each layer
                input.append(currA[1:])
                for l in range(self.h+1):
                    input.append(np.matmul(self.weights[l],currA))
                    currA = sigmoid(input[l+1])
                    currA = np.hstack([[1],currA]) #add bias term in current a
                    a.append(currA)
                delta = []
                #delta.append(np.multiply(sigmoidPrime(a[self.h+1][1:]),Y[row]-a[self.h+1][1:,]))
                delta.append(sigmoidPrime(a[self.h + 1][1:])*(Y[row] - a[self.h + 1][1:, ]))
                for l in range(self.h,-1,-1):
                    sum = np.matmul(self.weights[l].transpose(),delta[-1])[1:,]
                    #delta.insert(0,np.multiply(sigmoidPrime(a[l][1:]),sum))
                    #delta.append(np.multiply(sigmoidPrime(a[l][1:]), sum))
                    delta.append(sigmoidPrime(a[l][1:])*sum)
                for l in range(self.h+1):
                    #self.weights[l] = self.weights[l] + alpha*np.outer(delta[l+1],a[l])
                    self.weights[l] = self.weights[l] + alpha * np.outer(delta[self.h - l], a[l])
        return Y
    def predict(self,T):
        ypred = []
        m = T.shape[0]
        n = T.shape[1]
        for col in range(n):
            T[:, col] = T[:, col] - self.mean[col]
            if self.std[col] != 0:
                T[:, col] = T[:, col] / self.std[col]
        for row in range(m):
            a = [1] #input layer, add bias term
            for i in range(T.shape[1]):
                a.append(T[row][i])
            a = np.array(a)
            for l in range(self.h+1):
                input = np.matmul(self.weights[l],a)
                a = sigmoid(input)
                a = np.hstack([[1],a]) # add bias term
            ypred.append(a[1:,])
        return np.array(ypred) # return a numpy array
    def print(self):
        for l in range(len(self.weights)):
            print("Weights of Layer "+str(l+2))
            printWeight(self.weights[l].transpose())

def sigmoid(input):
    return 1/(1+np.exp(-input))
def sigmoidPrime(input):
    #return np.multiply(input,1-input)
    return input*(1-input)
def printWeight(weight):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            print(weight[i,j],end = " ")
        print("")
def computeAccuracy(ypred,y):
    cnt = 0
    m = y.shape[0]
    for row in range(m):
        if np.argmax(y[row])==np.argmax(ypred[row]):
            cnt += 1
    return cnt/m
#read MNIST data
data = pd.read_csv("train.csv")
X = np.array(data[:].drop("label",1))
y = np.array(data["label"])
h = 1
s = 3
alpha = 0.01
t = 1
model = ANN(h,s)
Y = model.fit(X,y,alpha,t)
ypred = model.predict(X)
#Estimate generalization error
Loss = np.mean((Y-ypred)**2)/2
print(Loss)
#Compute accuracy
print(computeAccuracy(ypred,Y))
model.print()

#David's Test case
#x = pd.DataFrame(np.arange(18).reshape(6,3))
#y = [0,1,2,3,0,1]
#nn = ANN(1,5)
#nn.fit(x,y,0.001,1)
#nn.print()

