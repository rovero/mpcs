import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
mpl.rc('figure', figsize=[12,8])  #set the default figure size

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class KNNRegressor(sklearn.base.RegressorMixin):
    def __init__(self, k):
        self.k = k
        super().__init__()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X.reshape(-1, 1))

    def predict(self, T):
        predictions = []
        _, neighbors = self.nn.kneighbors(T)
        regressor = LinearRegression()
        for i in range(T.shape[0]):
            regressor.fit(self.X[neighbors[i]], self.y[neighbors[i]])
            predictions.append(regressor.predict([T[i]]))
        return np.asarray(predictions)

def f_func(x):
        return 3.0 + 4.0 * x - 0.05 * x ** 2

def generate_data(size=200):
    X = np.sort(np.random.random(size) * 100)
    y = f_func(X) + (np.random.random(size) - 0.5) * 50
    return (X, y)


class LWRegressor():
    # kernel is a function
    def __init__(self, gamma):
        # self.k = k
        self.gamma = gamma

    # X should be of shape (m,n)
    def fit(self, X, y):
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.X_M = np.hstack([[[1], ] * self.m, X])
        self.X = X
        self.y = y
        # self.nn = NearestNeighbors(n_neighbors=self.k)
        # self.nn.fit(X.reshape(-1,1))

    def predict(self, T):
        predictions = []
        for i in range(T.shape[0]):
            print("fit example " + str(i))
            weight_opt = self.findWeight(i,T[i], 1e-2, 1e-5)
            xq = np.hstack([[1], T[i]])
            # print(weight_opt)
            predictions.append(np.matmul(xq, weight_opt))
        return predictions

    # X_M is the matrix that add a column of 1s
    def gradient(self, gamma, w, xq, X, X_M, y):
        xq = np.array([xq] * self.m)
        M_right = np.exp(-gamma * np.sum((xq - X) ** 2, 1).reshape(-1, 1)) * (y.reshape(-1, 1) - np.matmul(X_M, w))
        return np.matmul(X_M.T, M_right)

    def loss(self, w, xq, X, X_M, y):
        xq = np.array([xq] * self.m)
        a = np.exp(-self.gamma * np.sum((xq - X) ** 2, 1).reshape(-1, 1))
        b = (y.reshape(-1, 1) - np.matmul(X_M, w)) ** 2
        return np.sum(a * b)

    # do gradient descent to find the best weight for a query xq
    def findWeight(self,index, xq, alpha, epsilon):
        w = np.zeros((self.n + 1, 1))
        xq = np.hstack([[1], xq])
        loss = []
        l = self.loss(w, xq, self.X, self.X_M, self.y)
        loss.append(l)
        i = 0
        while (True):
            #print(l_prev)
            w += alpha / self.m * self.gradient(self.gamma, w, xq, self.X, self.X_M, self.y)
            l = self.loss(w, xq, self.X, self.X_M, self.y)
            loss.append(l)
            i += 1
            #print(l)
            if (abs(loss[i] - loss[i-1])) <= epsilon:
                break
        plt.plot(range(len(loss)),loss)
        plt.savefig("loss_"+str(index)+".png")
        return w

u = np.linspace(20,40,60)
f = f_func(u)
X, y = generate_data()
X = X.reshape(-1,1)
lwr = LWRegressor(1/40)
lwr.fit(X,y)
predictions = lwr.predict(u.reshape(-1,1))
#plt.plot(u,f, 'r', label='underlying function')
plt.scatter(X, y, s=10, color='b', alpha=0.5, label='data')
plt.plot(u,predictions, color='g', label='locally weighted regressor')
plt.legend()