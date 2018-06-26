import numpy as np
import pandas as pd
import math


class CNN():
    ConvLayers = None
    conv = None  # number of convolutional layer
    FCLayers = None  # FCLayers is also a list of CNNLayer()
    fc = None  # number of fully-connected layer
    nfilters_conv = None  # list number of filters in each conv layer
    nfilters_fc = None  # list number of filters in each fc layer
    fshape_conv = None  # list of filter shape in each layer,contains height&width e.g [(h,w),...]
    fshape_fc = None  # list of filter shape in each layer,only contains depth e.g [d,...]

    def __init__(self, conv, fc, nfilters_conv, nfilters_fc, fshape_conv, fshape_fc):
        self.conv = conv
        self.fc = fc
        self.ConvLayers = []
        self.FCLayers = []
        if len(nfilters_conv) != conv or len(nfilters_fc) != fc:
            print("nfilters: length does not match!")
            return
        if len(fshape_conv) != conv or len(fshape_fc) != fc:
            print("fshape: length does not match!")
            return
        self.nfilters_conv = nfilters_conv
        self.nfilters_fc = nfilters_fc
        self.fshape_conv = fshape_conv
        self.fshape_fc = fshape_fc
        for i in range(conv):
            self.ConvLayers.append(CNNLayer(nfilters_conv[i], fshape_conv[i] + (0,)))
        for j in range(fc):
            self.FCLayers.append(CNNLayer(nfilters_fc[j], (0, 0) + (fshape_fc[j],)))

    # Assume X is 3-d array, y is 1-d array
    def fit(self, X, y, alpha, t):
        m = X.shape[0]
        for it in range(t):
            input = X
            # forward propagation
            i = 0
            for cl in self.ConvLayers:
                cl.filter_shape = (cl.filter_shape[0], cl.filter_shape[1], input.shape[3])
                input = cl.forward_step(input)
                print("forward complete: conv" + str(i))
                i += 1
            i = 0
            for fc in self.FCLayers:
                # print("c"+str(input.shape[1]))
                # print("c"+str(input.shape[2]))
                fc.filter_shape = input.shape[1:]
                input = fc.forward_step(input)
                print("forward complete: fc" + str(i))
                i += 1
            # Compute gradient of Hinge Loss
            out_delta = []
            lastl = self.FCLayers[self.fc - 1]
            for i in range(m):
                # for each example, compute delta
                out_delta.append(
                    activPrime(lastl.convOut_no_act[i], lastl.activation) * gradientHingeLoss(y[i], lastl.convOut[i]))
            out_delta = np.array(out_delta)
            # backward propagation
            for l in range(self.fc - 1, -1, -1):  # starts from second last layer
                out_delta = self.FCLayers[l].backward_step(out_delta)
                print("backward complete: fc" + str(l))
            for l in range(self.conv - 1, -1, -1):
                out_delta = self.ConvLayers[l].backward_step(out_delta)
                print("backward complete: conv" + str(l))
            # update weights in conv layers
            for l in range(0, self.conv):
                self.ConvLayers[l].update(alpha)
            # update weights in fc layers
            for l in range(0, self.fc):
                self.FCLayers[l].update(alpha)
        return self

    def predict(self, T):
        input = T
        y_predict = []
        # conv layer forward propagation
        for cl in self.ConvLayers:
            input = cl.forward_step(input, 0)
        # TODO: fc layer forward propagation
        for fc in self.FCLayers:
            input = fc.forward_step(input)
        for i in range(input.shape[0]):
            y_predict.append(input[i].argmax())
        return np.array(y_predict)


# gradientHingeLoss for a single example
def gradientHingeLoss(y, yhat):  # y is a label, yhab is an array of floats
    n = 0
    delta = np.zeros((1, 1, yhat.shape[2]))  # depth is number of labels
    for i in range(yhat.shape[2]):
        if i != y:
            if (yhat[:, :, i] < yhat[:, :, y]):
                delta[:, :, i] = 0
            else:
                delta[:, :, i] = 1
                n += 1
    delta[:, :, y] = -n
    return delta


class CNNLayer():
    nfilter = None
    filters = None
    bias = None
    filter_shape = None
    stride = None
    activation = None
    delta = None
    out_delta = None
    X = None
    convOut = None
    convOut_no_act = None
    delta = None
    pad_in = 0
    pad_out = 0
    h_in = 0
    w_in = 0
    d_in = 0
    padX = None

    # filter_shape is (h,w,d)
    # stride is assumed to be 1
    def __init__(self, n, filter_shape, stride=1, activation="no_act"):
        self.nfilter = n
        self.filter_shape = filter_shape  # 3-d (height,width,depth)
        self.stride = stride
        self.activation = activation  # relu or no_act
        # TODO: initialize filters and bias
        # s = self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[2]
        # self.bias = []
        # self.bias = np.random.randn(self.nfilter,1)*(2/s)
        # for i in range(self.nfilter):
        #    w = np.random.randn(s) * math.sqrt(2.0/s)
        #    self.filters.append(w.reshape(self.filter_shape[0],self.filter_shape[1],self.filter_shape[2]))

    # X has shape (N,h,w,d)
    # return the output
    def forward_step(self, X, pad=0):
        self.X = X
        self.pad_in = pad
        # if filters is not initialized by set_filters(), I am going to initialized them with small random numbers
        n = self.filter_shape[0] * self.filter_shape[1] * self.filter_shape[2]
        if (self.filters == None):
            self.filters = []
            # TODO: initialize filters
            for i in range(self.nfilter):
                w = np.random.randn(n) * math.sqrt(2.0 / n)
                self.filters.append(w.reshape(self.filter_shape[0], self.filter_shape[1], self.filter_shape[2]))
        self.bias = np.random.randn(self.nfilter, 1) * (2 / n)
        output = []
        self.X = X
        N = X.shape[0]
        h = X.shape[1]
        self.h_in = h
        w = X.shape[2]
        self.w_in = w
        d = X.shape[3]
        self.d_in = d
        nfilter = self.nfilter  # Number of filters
        fsize = self.filter_shape[1]
        outsize = int((w - fsize + 2 * pad) / self.stride + 1)
        convOut = []
        convOut_no_act = []
        self.padX = []
        for example in range(N):
            # pading
            padExample = np.lib.pad(X[example], ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            self.padX.append(padExample)
            h_new = h + 2 * pad
            w_new = w + 2 * pad
            X_col = []  # matrix of inputs
            W_row = []  # matrix of filters
            for i in range(fsize - 1, h_new, self.stride):
                for j in range(fsize - 1, w_new, self.stride):
                    X_col.append(padExample[(i - fsize + 1):(i + 1), (j - fsize + 1):(j + 1), :].flatten())
            X_col = np.array(X_col)
            X_col = X_col.T
            for fIndex in range(nfilter):
                W_row.append(self.filters[fIndex].flatten())
            W_row = np.array(W_row)
            out = np.dot(W_row, X_col) + self.bias
            out = out.reshape((self.nfilter, outsize, outsize)).transpose((1, 2, 0))
            convOut.append(activate(out, self.activation))
            convOut_no_act.append(out)
            # convOut.append(activate(out,self.activation))
        # return output of shape (N,h,w,d)
        self.convOut_no_act = convOut_no_act
        self.convOut = convOut
        return np.array(convOut)
        # return X_col,W_row,convOut

    def backward_step(self, out_delta):
        # out_delta_sum = sum(out_delta)
        self.out_delta = out_delta.copy()
        h = out_delta.shape[1]
        w = out_delta.shape[2]
        d = out_delta.shape[3]
        # reconstruct filters into d filters and flip each of them
        filters_new = []
        for d in range(self.filter_shape[2]):
            temp = np.zeros((self.filter_shape[0], self.filter_shape[1], self.nfilter))
            for i in range(self.nfilter):
                temp[:, :, i] = self.filters[i][:, :, d][::-1, ::-1]
            filters_new.append(temp)
        fsize_new = filters_new[0].shape[0]
        p = ((self.w_in + 2 * self.pad_in - 1) * self.stride + self.filter_shape[1] - out_delta.shape[2]) / 2
        if p != int(p):
            print("p is not an integer")
        p = int(p)
        self.pad_out = p
        h_new = h + 2 * p
        w_new = w + 2 * p
        self.delta = []
        for example in range(out_delta.shape[0]):
            padExample = np.lib.pad(out_delta[example], ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)
            X_col = []
            W_row = []
            for i in range(fsize_new - 1, h_new, self.stride):
                for j in range(fsize_new - 1, w_new, self.stride):
                    X_col.append(padExample[(i - fsize_new + 1):(i + 1), (j - fsize_new + 1):(j + 1), :].flatten())
            X_col = np.array(X_col)
            X_col = X_col.T
            for fIndex in range(len(filters_new)):
                W_row.append(filters_new[fIndex].flatten())
            W_row = np.array(W_row)
            out = np.dot(W_row, X_col)
            out = out.reshape((len(filters_new), self.w_in + 2 * self.pad_in, self.w_in + 2 * self.pad_in)).transpose(
                (1, 2, 0))
            gPrime = np.vectorize(activPrime)
            gPrimeIn = gPrime(self.padX[example], self.activation)
            self.testOut = out
            self.testgPrimeIn = gPrimeIn
            if self.activation != 'no_act':
                out = out * gPrimeIn
            self.delta.append(out)
        self.delta = np.array(self.delta)
        return self.delta

    # assume filters is a list of filters where each filter is list(bias,weight_matrix)
    # eg. A filter is like:
    # f1 = [[1],np.array([[[-1,-1,0],[0,-1,0],[0,0,-1]],[[0,-1,1],[-1,0,-1],[-1,1,1]],[[-1,0,0],[0,0,0],[1,-1,1]]])]
    def update(self, alpha):
        # out_delta_sum = sum(out_delta)
        h = self.h_in
        w = self.w_in
        d = self.d_in
        out_delta = self.out_delta
        # initialize f_delta, should finally have same size as filters
        f_delta_shape = (self.nfilter,) + self.filter_shape
        f_delta = np.zeros(f_delta_shape)
        X = self.X
        h_new = X.shape[1]
        w_new = X.shape[2]
        for example in range(out_delta.shape[0]):
            for d in range(out_delta.shape[3]):  # set each delta depth layer as a filter
                f = np.dstack([out_delta[example][:, :, d]] * self.d_in)
                fsize = f.shape[0]
                h_i = 0
                for i in range(fsize - 1, h_new, self.stride):
                    w_i = 0
                    for j in range(fsize - 1, w_new, self.stride):
                        # testd = delta[example][(i-fsize+1):(i+1),(j-fsize+1):(j+1),:]
                        p = X[example][(i - fsize + 1):(i + 1), (j - fsize + 1):(j + 1), :] * f
                        f_delta[d][h_i, w_i, :] += np.sum(np.sum(p, 0), 0)
                        w_i += 1
                    h_i += 1
        self.filters += alpha / out_delta.shape[0] * f_delta
        return self.filters

    # TODO: PRINT filters
    def print(self):
        for i in range(self.filters.shape[0]):
            print("Filter " + str(i) + " is:(print by depth layer)")
            self.printAfilter(self.filters[i])
        return

    def printAfilter(self, filter):
        for d in range(filter.shape[2]):
            print("layer of depth " + str(d))
            for h in range(filter.shape[0]):
                for w in range(filter.shape[1]):
                    print(filter[h, w, d], end=" ")
                print("")

    def set_filters(self, filters):
        self.filters = []
        self.bias = []
        if (len(filters) != self.nfilter):
            print("number of filters does not match!")
            return
        if (np.all(filters[0][1].shape != self.filter_shape)):
            print("shape of filters does not match")
            return
        for i in range(self.nfilter):
            self.filters.append(filters[i][1])
            self.bias.append(filters[i][0])
        self.filters = np.array(self.filters)


# activation function
def activate(output, activation):
    if activation == 'relu':
        return np.maximum(output, 0)
    elif activation == 'no_act':
        return output


def activPrime(x, activation):
    if activation == 'relu':
        return reluPrime(x)
    elif activation == 'no_act':
        return 1


def reluPrime(x):
    if x >= 0:
        return 1
    else:
        return 0

#These codes are used for debugging forward_step()
X = np.array([np.array([[[1,0,1],[1,2,0],[1,0,1],[2,0,0],[2,0,2]],[[0,0,1],[0,0,0],[1,2,2],[1,1,1],[1,1,2]],
              [[2,2,1],[2,2,1],[1,0,1],[0,0,0],[2,0,1]],[[0,0,2],[2,0,1],[1,1,0],[1,1,1],[2,0,0]],
              [[2,1,2],[1,0,2],[0,1,1],[1,1,0],[2,1,0]]])])
cl = CNNLayer(2,(3,3,3),2)
filters = []
#filters are in this format
f1 = [[1],np.array([[[-1,-1,0],[0,-1,0],[0,0,-1]],[[0,-1,1],[-1,0,-1],[-1,1,1]],[[-1,0,0],[0,0,0],[1,-1,1]]])]
f2 = [[0],np.array([[[0,0,-1],[-1,0,-1],[1,0,0]],[[1,1,-1],[1,-1,-1],[1,-1,-1]],[[0,-1,0],[1,0,1],[-1,1,1]]])]
filters.append(f1)
filters.append(f2)
cl.set_filters(filters)
cl.forward_step(X,1)

a = np.array([[ 1.,  0.,  1.,  0.,  1.],
         [ 0.,  1.,  0.,  1.,  0.],
         [ 1.,  1.,  1.,  1.,  1.],
         [ 0.,  0.,  0.,  0.,  0.],
         [ 1.,  1.,  1.,  0.,  0.]])
input = np.zeros((5,5,1))

input[:,:,0] = a

conFilter = np.zeros((3,3,1))

conFilter[:,:,0] = np.array([[0,1,0],[0,1,0],[1,0,0]])

filter = [[0],conFilter]

cl = CNNLayer(1,(3,3,1))

filters = []

filters.append(filter)

cl.set_filters(filters)
X = []
X.append(input)
X = np.array(X)
out = cl.forward_step(X,0)

out_delta = np.zeros((1,3,3,1))

out_delta[0][:,:,0] = np.array([[ 34.,  34.,  34.],
                                [ 34.,  34.,  34.],
                                [ 34.,  34.,  34.]])

cl.backward_step(out_delta)[0][:,:,0]

cl.update(0.001)
cl.print()

mnist = pd.read_csv("train.csv")

X = []
examples = mnist.drop('label',1)
y = mnist['label']
m = examples.shape[0]
n = examples.shape[1]

for i in range(m):
    #convert each example into a 3-d array
    x = np.zeros((28,28,1))
    e = examples.iloc[i].reshape((28,28))
    x[:,:,0] = e
    X.append(x)
X = np.array(X)

#create cnn model
cnn = CNN(2, 1, [32,32],[10],[(5,5),(5,5)],[10])
#fit cnn model
cnn = cnn.fit(X,y,0.001,1)
#make prediction
print(cnn.predict(X))