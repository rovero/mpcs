import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
mpl.rc('figure', figsize=[12,8])  #set the default figure size

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

#3(a)
class LWRegressor():
    # kernel is a function
    def __init__(self, gamma=1/40):
        # self.k = k
        self.gamma = gamma

    # X should be of shape (m,n)
    def fit(self, X, y):
        self.X = X
        self.y = y
        # self.nn = NearestNeighbors(n_neighbors=self.k)
        # self.nn.fit(X.reshape(-1,1))

    def predict(self, T):
        predictions = []
        for i in range(T.shape[0]):
            xq = T[i]
            regressor = LinearRegression()
            w = np.exp(-self.gamma*np.sum((self.X-xq)**2,1))
            regressor.fit(self.X,self.y,w)
            predictions.append(regressor.predict([xq]))
        return predictions

    def set_params(self,gamma):
        self.gamma = gamma

class KNNRegressor(sklearn.base.RegressorMixin):
    def __init__(self, k=2):
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
    def set_params(self, k):
        self.k = k

def f_func(x):
        return 3.0 + 4.0 * x - 0.05 * x ** 2

def generate_data(size=200):
    X = np.sort(np.random.random(size) * 100)
    y = f_func(X) + (np.random.random(size) - 0.5) * 50
    return (X, y)

#3(b)

u = np.linspace(0,100,300)
f = f_func(u)
X, y = generate_data()
knn_reg = KNNRegressor(5)
knn_reg.fit(X.reshape(-1,1), y)
predictions = knn_reg.predict(u.reshape(-1,1))

plt.plot(u,f, 'r', label='underlying function')
plt.scatter(X, y, s=10, color='b', alpha=0.5, label='data')
plt.plot(u,predictions, color='g', label='knn linear regression')

X = X.reshape(-1,1)
lwr = LWRegressor(1/40)
lwr.fit(X,y)
predictions = lwr.predict(u.reshape(-1,1))
plt.plot(u,predictions, color='y', label='locally weighted regressor')

plt.legend()
plt.show()


#3(c) cross validation

from sklearn.model_selection import validation_curve

train_scores1,test_scores1 = validation_curve(KNNRegressor(),X,y,"k",range(10,100),cv=5,scoring='neg_mean_squared_error')


train_scores_mean1 = np.mean(train_scores1, axis=1)
train_scores_std1 = np.std(train_scores1, axis=1)
test_scores_mean1 = np.mean(test_scores1, axis=1)
test_scores_std1 = np.std(test_scores1, axis=1)

plt.title("Validation Curve with KNN")
plt.plot(range(10,100),train_scores_mean1,'r')
plt.plot(range(10,100),test_scores_mean1,'g')
plt.show()

print("The best k is: ",end=" ")
print(range(10,100)[np.argmax(test_scores_mean1)])

train_scores2,test_scores2 = validation_curve(LWRegressor(),X,y,"gamma",np.arange(0.001,0.2,0.001),cv=5,scoring='neg_mean_squared_error')

train_scores_mean2 = np.mean(train_scores2, axis=1)
train_scores_std2 = np.std(train_scores2, axis=1)
test_scores_mean2 = np.mean(test_scores2, axis=1)
test_scores_std2 = np.std(test_scores2, axis=1)

plt.title("Validation Curve with LWR")
plt.plot(np.arange(0.001,0.2,0.001),train_scores_mean2,'r')
plt.plot(np.arange(0.001,0.2,0.001),test_scores_mean2,'g')
plt.show()

print("The best gamma is: ",end=" ")
print(np.arange(0.001,0.2,0.001)[np.argmax(test_scores_mean2)])