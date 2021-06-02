#This code is for PCA analysis

#Importing the module numpy and others from it
import numpy as np
from numpy import linalg as LA # importing linear algebra

#class for pca
class d_PCA:
    
    #taking number of components
    def __init__(self, num_of_components):
        self.num_of_components = num_of_components
    
    #building fit transform model which takes in correlation logic and data
    def fit_transform(self, x, corr_logic):
        column_mean = x.mean(axis=0) #calculating the mean of column
        column_mean_stacked = np.tile(column_mean, reps=(x.shape[0], 1)) #stacking the mean values
        x_mean_centered = x - column_mean_stacked #finding the mean center

        #for correlation logic
        # use mean_centered data or standardized mean_centered data
        if not corr_logic:
            data_for_pca = x_mean_centered
        else:
            column_sd = np.std(x, axis=0)
            column_sd_stacked = np.tile(column_sd, reps=(x.shape[0], 1))
            data_for_pca = x / column_sd_stacked

        # get covariance matrix of the data
        covariance_matrix = np.cov(data_for_pca, rowvar=False)

        # eigendecomposition of the covariance matrix
        w, v = LA.eig(covariance_matrix)

        w = w.real
        v = v.real

        # sort eigenvalues in descending order
        II = w.argsort()[::-1]
        all_eigenvalues = w[II]
        all_eigenvectors = v[:, II]

        # get percent variance
        percent_variance_explained = all_eigenvalues / sum(all_eigenvalues) * 100

        # get scores
        pca_scores = np.matmul(data_for_pca, all_eigenvectors)

        # collect PCA results
        pca_results = {'data': x,
                      'mean_centered_data': x_mean_centered,
                      'percent_variance_explained': percent_variance_explained,
                      'loadings': all_eigenvectors,
                      'scores': pca_scores,
                      'data_after_pretreatment': data_for_pca}
        #return the result
        return pca_results