import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression

#Problme 5
#X is a numpy matrix initialized by np.matrix()
#y is a numpy array
def gradient_descent(X,y,alpha,T):
    m = X.shape[0] #sample size
    n = X.shape[1] #number of features
    c = np.array([[1],]*m) #column of 1s that will be add to X
    X_new = np.hstack([c,X]) # m*(n+1) matrix
    if(X_new.shape[1]!=n+1):
        print("Fail to add a new column to X")
    theta = np.matrix(np.zeros(n+1)).transpose() # (n+1)*1 matrix
    #print("theta: "+str(theta.shape[0])+" "+str(theta.shape[1]))
    it = 0
    y_new = np.matrix(y).transpose() # tranfer y from np array to m*1 np matrix
    J_hist = np.zeros(T+1) #includes the initial J
    while(it<=T):
        h = 1/(1+np.exp(-np.matmul(X_new,theta))) # m * 1 matrix
        diff = h-y_new # m * 1 matrix
        a = np.array(y_new)[:,0].dot(np.array(np.log(h))[:,0])
        b = np.array(1-y_new)[:,0].dot(np.array(np.log(1-h))[:,0])
        J = -(a+b)/m
        #J = np.matmul(theta.transpose(),theta)[0,0]/2 + log
        J_hist[it] = J
        if it==T:
            break
        temp = np.matmul(X_new.transpose(),-diff)*alpha
        theta = theta + temp
        it += 1
    plt.plot(range(0,T+1),J_hist)
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function J(theta)")
    plt.savefig("Cost_vs_Iteration_num.png")
    plt.show()
    return theta

#Problem 6
df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
df_sample = df.iloc[0:30].drop('class',1)
y = df_sample['color']
X = df_sample.drop('color',1)
X = X.as_matrix() # X is now a np matrix
y = np.array(y) # y is now a np array
theta = gradient_descent(X,y,1e-15,100)

#Problem 7
#This choice of C leads to almost the same theta. Not sure why.
logR = LogisticRegression(C = 1e-13)
logR.fit(X,y)
logR.intercept_
logR.coef_

#Problem 8
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')

c1 = 'mradius'
c2 = 'mtexture'

df_new = pd.concat([df[c1],df[c2],df[c1]*df[c2],df[c1]**2,df[c2]**2,(df[c1]**2)*df[c2],
                    df[c1]*(df[c2]**2),df[c1]**3,df[c2]**3],axis=1,keys=[c1,c2,c1+"*"+c2,
                                                                         c1+"^2",c2+"^2",c1+"^2*"+c2,c2+"^2*"+c1,
                                                                         c1+"^3",c2+"^3"])
clf = LogisticRegression()
clf.fit(df_new, df['color'])

plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

clf = LogisticRegression()
clf.fit(df_new, df['color'])

plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

x = np.linspace(df[c1].min(), df[c1].max(), 1000)
y = np.linspace(df[c2].min(), df[c2].max(), 1000)
xx, yy = np.meshgrid(x,y)
predictions = clf.predict(
        np.hstack(
            (xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1), (xx*yy).ravel().reshape(-1,1), (xx**2).ravel().reshape(-1,1),
            (yy**2).ravel().reshape(-1,1),((xx**2)*yy).ravel().reshape(-1,1),(xx*(yy**2)).ravel().reshape(-1,1),(xx**3).ravel().reshape(-1,1),
        (yy**3).ravel().reshape(-1,1))
        ))
predictions = predictions.reshape(xx.shape)

plt.contour(xx, yy, predictions, [0.0],colors='blue')
plt.savefig('LogisticRegression.png')
plt.show()

