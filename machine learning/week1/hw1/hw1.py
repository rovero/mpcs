import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

class TreeNode():
    left = None
    right = None
    attribute = None
    splitter = None
    depth = None
    examples = None
    label = None
    type = None    #distinguish between binary variables and continuous variables
    def __init__(self,depth,examples):
        self.depth = depth
        self.examples = examples

class DecisionTree():
    max_depth = 0
    tree = None
    def __init__(self,max_depth=3):
        self.max_depth = max_depth

    def fit(self,X,y):
        self.tree = None
        if len(X)!=len(y):
            print("Length does not match")
            return
        #replace NAN with most common value of the column
        X = X.apply(lambda x: x.fillna(x.value_counts().idxmax()),axis=0)
        self.tree = treeGenerator(X,y,1,self.max_depth)
        return self

    def predict(self,T):
        if self.tree==None:
            print("No decision tree available at this time ")
            return
        y_predicted = []
        for i in range(0,len(T)):
            curr = self.tree
            while curr.label==None:
                if T.iloc[i][curr.attribute]<curr.splitter:
                    curr = curr.left
                else:
                    curr = curr.right
            y_predicted.append(curr.label)
        return y_predicted

    def print(self):  #print in BFS order
        if self.tree==None:
            print("No decision tree available at this time ")
        queue = deque([self.tree])
        while queue:
            curr = queue.popleft()
            print("depth: "+str(curr.depth))
            if curr.label!=None:
                print("type: leaf")
                print("class: "+str(curr.label))
            else:
                print("type: internal")
            if curr.attribute!=None:
                if curr.type == 1:
                    print("column "+str(curr.attribute)+"<1?")
                elif curr.type == 0:
                    print("column "+str(curr.attribute)+"<"+str(curr.splitter)+"?")
            if curr.left!=None:
                queue.append(curr.left)
            if curr.right!=None:
                queue.append(curr.right)

#grow tree that fits the training set
def treeGenerator(X,y,depth,max_depth):
    if len(set(y))==1 or depth==max_depth:
        leaf = TreeNode(depth,X)
        leaf.label = y.value_counts().idxmax() ##the most common value in y
        return leaf
    else:
        root = TreeNode(depth,X)
        X_left,X_right,root.attribute,root.splitter,root.type = findBestSplit(X,y)
        y_left = y[X_left.index.values]
        y_right = y[X_right.index.values]
        root.left = treeGenerator(X_left,y_left,depth+1,max_depth)
        root.right = treeGenerator(X_right,y_right,depth+1,max_depth)
        return root

def findBestSplit(X,y):
    min = float('inf')
    selectedAttr = X.columns[0]
    splitter = 1 #splitter for binary variable, false 0 left, true 1 right
    type = 0
    for attr in X.columns.values:
        X_sort = X.sort_values([attr])
        l = len(X_sort)
        #check binary column
        if set(X_sort[attr])<={0,1}: #it is a subset of {0,1}
            #left false
            left = X_sort[X_sort.iloc[:,attr]==0]
            #right true
            right = X_sort[X_sort.iloc[:,attr]==1]
            leftSet = pd.value_counts(y[left.index])
            rightSet = pd.value_counts(y[right.index])
            Entropy = computeEntropy(leftSet,rightSet)
            if Entropy < min:
                min = Entropy
                selectedAttr = attr
                X_left = left
                X_right = right
                splitter = 1
                type = 1  #1 represents binary variable
        else:
            prev = None
            for i in range(1, l):
                if X_sort.iloc[i][attr]==prev: #no need to recalculate the same value
                    continue
                prev = X_sort.iloc[i][attr]
                left = X_sort.iloc[:i]
                right = X_sort.iloc[i:]
                leftSet = pd.value_counts(y[left.index])
                rightSet = pd.value_counts(y[right.index])
                Entropy = computeEntropy(leftSet,rightSet)
                if Entropy < min:
                    min = Entropy
                    selectedAttr = attr
                    splitter = X_sort.iloc[i][attr]
                    X_left = left
                    X_right = right
                    type = 0
    return X_left, X_right,selectedAttr,splitter,type


def validation_curve():
    path = "arrhythmia.csv" #data file path
    n_attr = 8 # number of attribute used
    size = 452 #size of data used
    data = pd.read_csv(path, header=None)
    #Assign Binary values
    data = data.drop(range(n_attr, 279), 1)
    data = data.iloc[0:size]
    data_clean = data.replace('?', np.nan)  # replace '?' by NAN
    # replace NAN with most common value of the column
    data_clean  = data_clean.apply(lambda x: x.fillna(x.value_counts().idxmax()), axis=0)
    # randomly shuffle the dataset
    data_shuffle = data_clean.reindex(np.random.permutation(data_clean.index))
    length = len(data_shuffle)
    # partition 1
    data_1 = data_shuffle[:int(length / 3)]
    # partition 2
    data_2 = data_shuffle[int(length / 3):int(2 * length / 3)]
    # partition 3
    data_3 = data_shuffle[int(2 * length / 3):]
    data_train1 = data_1.append(data_2)  # use data1 & 2 as training set
    X_train1 = data_train1.iloc[:, :-1].copy()
    y_train1 = data_train1.iloc[:, -1].copy()
    X_test1 = data_3.iloc[:, :-1].copy()
    y_test1 = data_3.iloc[:, -1].copy()
    data_train2 = data_1.append(data_3)  # use data 1 & 3 as training set
    X_train2 = data_train2.iloc[:, :-1].copy()
    y_train2 = data_train2.iloc[:, -1].copy()
    X_test2 = data_2.iloc[:, :-1].copy()
    y_test2 = data_2.iloc[:, -1].copy()
    data_train3 = data_2.append(data_3)  # use data 2 & 3 as training set
    X_train3 = data_train3.iloc[:, :-1].copy()
    y_train3 = data_train3.iloc[:, -1].copy()
    X_test3 = data_1.iloc[:, :-1].copy()
    y_test3 = data_1.iloc[:, -1].copy()
    testAcc = []
    trainAcc = []
    for i in range(2, 21, 2):  # use different max depth
        testavgacc = 0
        trainavgacc = 0
        dtree1 = DecisionTree(i)
        dtree1 = dtree1.fit(X_train1, y_train1)
        print("Decision Tree 1")
        dtree1.print()
        print("")
        y_train_pred1 = dtree1.predict(X_train1)
        y_test_pred1 = dtree1.predict(X_test1)
        #calculate the accuracy of tree1
        testavgacc += computeAccuracy(y_test_pred1, y_test1)
        trainavgacc += computeAccuracy(y_train_pred1,y_train1)
        dtree2 = DecisionTree(i)
        dtree2 = dtree2.fit(X_train2, y_train2)
        print("Decision Tree 2")
        dtree2.print()
        #print("")
        y_train_pred2 = dtree2.predict(X_train2)
        y_test_pred2 = dtree2.predict(X_test2)
        #calculate the accuracy of tree2
        testavgacc += computeAccuracy(y_test_pred2, y_test2)
        trainavgacc += computeAccuracy(y_train_pred2,y_train2)
        dtree3 = DecisionTree(i)
        dtree3 = dtree3.fit(X_train3, y_train3)
        print("Decision Tree 3")
        dtree3.print()
        print("")
        y_train_pred3 = dtree3.predict(X_train3)
        y_test_pred3 = dtree3.predict(X_test3)
        #calculate the accuracy of tree3
        testavgacc += computeAccuracy(y_test_pred3, y_test3)
        trainavgacc += computeAccuracy(y_train_pred3,y_train3)
        testavgacc /= 3
        testAcc.append(testavgacc)
        trainavgacc /= 3
        trainAcc.append(trainavgacc)

    # plot cvAcc
    plt.plot(range(2, 21, 2), testAcc)
    plt.plot(range(2, 21, 2), trainAcc)
    plt.xlabel('Max-depth')
    plt.ylabel('Accuracy')
    plt.legend(['Testing Accuracy', 'Training Accuracy'], loc = 'best')
    plt.savefig('validation.pdf')
    plt.show()
    plt.close()

def computeEntropy(leftSet,rightSet):
    leftEntropy = 0
    rightEntropy = 0
    sumLeft = sum(leftSet)
    sumRight = sum(rightSet)
    l = sumLeft + sumRight
    for val in leftSet:
        p = val / sumLeft
        leftEntropy -= p * np.log2(p)
    for val in rightSet:
        p = val / sumRight
        rightEntropy -= p * np.log2(p)
    Entropy = (sumLeft / l) * leftEntropy + (sumRight / l) * rightEntropy
    return Entropy

def computeAccuracy(pred, real):
    a = 0
    l = len(real)
    if len(pred) != l:
        print("Prediction does not have same length as real data")
    for i in range(l):
        if pred[i] == real.iloc[i]:
            a = a + 1
    return a / l

validation_curve()