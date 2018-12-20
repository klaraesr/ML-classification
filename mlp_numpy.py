import numpy as np


class MLP(object):
    """
    Multi Layer Perceptron model.
    """
    dim_in = 2
    hidden_nodes = 100
    dim_out = 1
    eta = 0.01

    # Initialize random weights
    w = np.random.randn(dim_in, hidden_nodes)
    print(w.shape)
    v = np.random.randn(hidden_nodes, dim_out)
    print(v.shape)

    # Initialize sizes if input, hidden and output layer
    #x = np.zeros(dim_in)
    #h = np.zeros(hidden_nodes)
    #y = np.zeros(dim_out)

    def __init__(self):
        super(MLP, self).__init__()
        

    def fit(self, X, y):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        y: Targets. A numpy array of dimensions (number of samples,).
        :return: Nothing.
        """

        for i in range(100):
        #for n in range(len(y)):
            
            y_pred = self.forward(X)
            self.backward(X, y_pred, y)


    def sig_derivative(self, x):
        return x * (1-x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.x = X #1000, 2

        self.a = np.dot(self.x, self.w)
        self.h = self.sigmoid(self.a) #1000,100
        self.b = np.dot(self.h, self.v)
        self.y = self.sigmoid(self.b) #1000,1
        return self.y


    def backward(self, x, y_pred, y):
        d_y = y_pred - y
        d_b = self.sig_derivative(self.y)
        d_v = self.h.T

        d_out = np.dot(d_v, (d_y * d_b))

        d_h = np.dot((d_y * d_b), self.v.T)
        d_x = self.sig_derivative(self.h)
        d_w = x.T

        d_hid = np.dot(d_w, (d_h * d_x))
        

        self.w = self.w - (self.eta * d_hid)
        self.v = self.v - (self.eta * d_out)



    def predict(self, X):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        :return: Predicted targets. A numpy array of dimensions (number of samples,)
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        #for n in range(N):
        y_pred = self.forward(X)

        return np.round(y_pred).flatten()
        