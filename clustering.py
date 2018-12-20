import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


class K_means(object):
    """
    K means clustering algorithm.
    """
    centers = []

    def __init__(self):
        super(K_means, self).__init__()

    def fit(self, X, n_centers):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        n_centers: The number of centers. Integer.
        :return: Nothing.
        """
        plt.scatter(X[:,0], X[:,1], c='blue')
        centers = self.init_centers(X, n_centers)
        done = False
        while not done:
            nc = self.nearest_center(X, centers)
            centers, done = self.move_centers(X, nc, centers)
            print("centers: ")
            print(centers)
            print(done)
        plt.scatter(centers[:,0], centers[:,1], c = 'black')
        self.centers = centers
        plt.show()

    def get_centers(self):
        """
        :return: Centers. A numpy array of dimensions (number of centers, number of features).
        """
        
        return self.centers
    
    def init_centers(self, X, k):
        var = np.var(X)
        mean = np.mean(X)
        shape = X.shape
        rand_points = np.random.sample(shape) * var + mean
        # plt.scatter(rand_points[:,0], rand_points[:,1], rand_points[:,2], c = 'blue')
        centers = rand_points[:k]
        print(centers.shape)
        # plt.scatter(centers[:,0], centers[:,1], centers[:,2], c = 'black')
        # plt.show()
        return centers
        
    def nearest_center(self, points, centroids):
        nc_arr = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            min_dist = 10000000000
            centroid = 0
            for j in range(centroids.shape[0]):
                dist = np.linalg.norm(points[i] - centroids[j]) # euclidian distance
                if dist < min_dist:
                    min_dist = dist
                    centroid = j
            nc_arr[i] = centroid
        return nc_arr

    def move_centers(self, points, nc, centroids):
        new_centroids = centroids.copy()
        done = False
        print(nc.shape)
        for i in range(centroids.shape[0]):
        #get all points where nearest cluster is i, and find mean of each feature
            #mean = points[np.where(nc == i)[0]].mean(axis=0)
            mean = np.mean(points[nc == i], axis=0)
            new_centroids[i] = mean
        if np.allclose(centroids, new_centroids):
            done = True
        print(new_centroids.shape)
        return new_centroids, done

class EM(object):
    """
    Expectation Maximization algorithm.
    """
    def __init__(self):
        super(EM, self).__init__()

    def fit(self, X, n_centers):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        n_centers: The number of centers. Integer.
        :return: Nothing.
        """

        log_like = 0

        N, n_features = X.shape
        self.pis = np.abs(np.random.rand(n_centers))
        self.pis /= self.pis.sum()
        self.mus = X[np.random.choice(N, n_centers)]
        self.sigmas = np.zeros((n_centers, n_features, n_features))
        for k in range(n_centers):
            random_sigmas_k = np.random.rand(n_features, n_features)
            self.sigmas[k] = np.dot(random_sigmas_k, random_sigmas_k.T)

        print("number of features: ", n_features)
        print("shape pis: ", self.pis.shape)
        print("shape mus: ", self.mus.shape)
        print("shape sigmas: ", self.sigmas.shape)

        for i in range(100):
            # E-step
            gamma = self.e_step(X, N)

            # M-step
            self.m_step(gamma, X, N)
            print(i)

            # Compute log likelihood to see if either parameters converges before 100th iteration
            log_like_new = 0
            for n in range(N):
                sum = 0
                for k in range(n_features):
                    sum += self.pis[k] * mvn(mean=self.mus[k], cov=self.sigmas[k]).pdf(X[n])
                log_like_new += np.log(sum)

            if np.abs(log_like - log_like_new) < 0.01:
                break
            log_like = log_like_new
        



    def e_step(self, X, N):
        K = len(self.pis)
        gamma = np.zeros((K, N))

        for k in range(K):
            for n in range(N):
                gamma[k][n] = self.pis[k] * mvn(mean=self.mus[k], cov=self.sigmas[k]).pdf(X[n])
        gamma /= np.sum(gamma, axis=0)
        print("N = ", N, "K = ", K)
        print("Gamma shape: ", gamma.shape)
        print("X shape: ", X.shape)
        return gamma

    def m_step(self, gamma, X, N):
        K = len(self.pis)

        gamma_row_sum = np.zeros(K)
        for k in range(K):
            gamma_row_sum[k] = np.sum(gamma[k])
        

        print("mus before: ", self.mus.shape)
        self.mus.fill(0)
        for k in range(K):
            for n in range(N):
                self.mus[k] += gamma[k][n] * X[n]
            self.mus[k] /= gamma_row_sum[k]
        print("mus after: ", self.mus.shape)

        a = 0
        print("sigmas before: ", self.sigmas.shape)
        self.sigmas.fill(0)
        for k in range(K):
            for n in range(N):
                diff = (X[n] - self.mus[k]).reshape((2, 1))
                if k == 0 and n == 0:
                    print("Shape Xn: ", X[n].shape, " shape musk: ", self.mus[k].shape)
                    print("diff shape: ", diff.shape)
                self.sigmas[k] += gamma[k][n] * np.dot(diff, diff.T)
            self.sigmas[k] /= gamma_row_sum[k]
        self.sigmas = np.nan_to_num(self.sigmas)
        print("sigmas after: ", self.sigmas)

        self.pis.fill(0)
        for k in range(K):
            self.pis[k] = gamma_row_sum[k] / N



    def get_params(self):
        """
        :return: 3 gaussian mixture model parameters, piss, mus, sigmass.
                piss : A numpy array of dimensions (number of centers,).
                mus : A numpy array of dimensions (number of centers, number of features).
                sigmass : A numpy array of dimensions (number of centers, number of features, number of features).
        """

        return (self.pis, self.mus, self.sigmas)