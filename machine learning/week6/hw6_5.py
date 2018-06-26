from sklearn.svm import SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#(b)
x = np.array([[1.5,6],[2,1],[3,4],[4,4],[3.5,2],[4.5,4.5]])

y = np.array([1,1,1,0,0,0])

clf1 = SVC(C=100,kernel='linear')

clf1.fit(x,y)

xx = np.linspace(1.5,4.5)
a = -clf1.coef_[0][0]/clf1.coef_[0][1]
b = -clf1.intercept_/clf1.coef_[0][1]
yy = a*xx+b
my_color_map = mpl.colors.ListedColormap(['blue', 'red'], 'mycolormap')
plt.scatter(x[:,0],x[:,1],c=y,cmap=my_color_map)
plt.plot(xx,yy)
plt.show()

#(c)
x = np.array([[1.5,6],[2,1],[3,4],[4,4],[5,2],[4.5,4.5]])
clf2 = SVC(C=100,kernel='linear')
clf2.fit(x,y)
xx = np.linspace(1.5,4.5)
plt.scatter(x[:,0],x[:,1],c=y,cmap=my_color_map)
plt.axvline(x=-clf2.intercept_/clf2.coef_[0][0])
plt.show()

#(d)
x = np.array([[1.5,6],[2,1],[3,4],[4,4],[3.5,2],[4.5,4.5]])
y[5] = 1-y[5]
for c in [0.1,1,10]:
    clf3 = SVC(C=c,kernel='linear')

    clf3.fit(x,y)

    xx = np.linspace(1.5,4.5)
    a = -clf3.coef_[0][0]/clf3.coef_[0][1]
    b = -clf3.intercept_/clf3.coef_[0][1]
    yy = a*xx+b
    my_color_map = mpl.colors.ListedColormap(['blue', 'red'], 'mycolormap')
    plt.scatter(x[:,0],x[:,1],c=y,cmap=my_color_map)
    plt.plot(xx,yy)
    plt.show()

#(e)

h = .02
for C in [0.1,1,10]:
    rbf_svc = SVC(kernel='rbf', gamma=2, C=C).fit(x, y)
    poly_svc = SVC(kernel='poly', degree = 10,C=C).fit(x, y)

    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']


    for i, clf in enumerate((rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()