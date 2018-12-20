import numpy as np
#import torch
#import clustering
#import matplotlib.pyplot as plt
import mlp_numpy
#import nn_pytorch


def main():
    """
    The purpose of main.py is to provide an running example of classes and functions.
    This file is not considered in the evaluation process.
    """

    # data generated from 3 mixture components
    # dim : n_samples X n_features
    X = np.random.normal([[[1], [1]], [[2], [2]], [[3], [3]]], 0.2, (3, 2, 100)).swapaxes(1, 2).reshape(300, 2) # 300 samples, 2 features
    Xc = np.random.normal([[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]], 0.2, (3, 3, 1000)).swapaxes(1, 2).reshape(3000, 3) # 3000 samples, 3 features

    # # K_means
    # model = clustering.K_means()
    # model.fit(X, 3)
    # centers = model.get_centers()
    # EM
    # model = clustering.EM()
    # model.fit(X, 3)
    # pis, mus, sigmas = model.get_params()

    # # get samples from data/nn_train.txt and data/nn_test.txt
    # # X_train = ...
    # # y_train = ...
    # # X_test = ...
    file_train = "data/nn_train.txt"
    file_test = "data/nn_test.txt"

    data_train = np.loadtxt(file_train)
    data_test = np.loadtxt(file_test)

    X_train = data_train[:,1:]
    y_train = data_train[:,0:1]
    X_test = data_test

    # MLP
    model = mlp_numpy.MLP()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_pred)
    
    # # NN


    # #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # X_train = torch.from_numpy(data_train[:,1:]) # convert to pytorch Tensors
    # print(X_train.shape)
    # y_train = torch.from_numpy(data_train[:,0:1])
    # print(y_train.shape)
    # X_test = torch.from_numpy(data_test)

    # model = nn_pytorch.NN()
    # nn_pytorch.train(model, X_train, y_train)
    # y_pred = nn_pytorch.test(model, X_test)
    

if __name__ == "__main__":
    main()