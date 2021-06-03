#This is complete code for PCA analysis

# import module operating system
import os
#for manipulate different parts of the Python runtime environment
import sys

#importing the modules from python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the library sklearn for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# set up paths
# --------------------------------------------------------------------------
# get the directory path of the running script
# working_dir_absolute_path = os.path.dirname(os.path.realpath(__file__))
#
# toolbox_absolute_path = os.path.join(working_dir_absolute_path, "ML_toolbox")
# data_absolute_path = os.path.join(working_dir_absolute_path, "data")
#
# sys.path.append(toolbox_absolute_path)
# sys.path.append(data_absolute_path)

#From the ml toolbox importing the pca class
from ML_toolbox.d_PCA import d_PCA
import config


example_index = 6
# example_index = 1: linearly correlated toy data with two variables
# example_index = 2: pure random toy data with two variables
# example_index = 3: toy data with three variables
# example_index = 4: real data
# example_index = 5: use sklearn
# example_index = 6: use correlation

#setting the correlation to true/false
use_corr = False

#writing the main function
def main():
    x1 = np.arange(start=0, stop=20, step=0.1)
    num_of_samples = len(x1)

    if example_index == 1:
        # ----------------------------------------------------------------
        # A first glimpse at PCA
        # ----------------------------------------------------------------

        standard_deviation = [0.5, 4.0]

        for i in range(len(standard_deviation)):
            cur_scale = standard_deviation[i]

            # 1. generate the raw data
            x2 = 2 * x1 + np.random.normal(loc=0, scale=cur_scale, size=num_of_samples)

            # 2. visualize the raw data
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x1, x2, color='blue')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_aspect('equal', 'box')
            fig.show()

            # 3. do PCA
            dataForAnalysis = np.column_stack((x1, x2))

            num_of_components = min(dataForAnalysis.shape[0], dataForAnalysis.shape[1])

            #use the pca class from ml toolbox
            object_pca = d_PCA(num_of_components=num_of_components)
            pca_results = object_pca.fit_transform(x=dataForAnalysis, corr_logic=use_corr)#using fit transform 

            # scree plot
            fig, ax = plt.subplots()
            ax.scatter(range(len(pca_results['percent_variance_explained'])), pca_results['percent_variance_explained'], color='blue')
            ax.set_title('scree plot')
            ax.set_xlabel('PC index')
            ax.set_ylabel('percent variance explained')
            ax.set_ylim((-10.0, 110.0))
            fig.show()

            # scores plot
            fig, ax = plt.subplots()
            ax.scatter(pca_results['scores'][:, 0], pca_results['scores'][:, 1], color='blue')
            ax.set_title('scores plot')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            fig.show()

            fig, ax = plt.subplots()
            ax.scatter(pca_results['scores'][:, 0], pca_results['scores'][:, 1], color='blue')
            ax.set_title('scores plot')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_ylim((2*min(pca_results['scores'][:, 1]), max(x2)))
            fig.show()

            # loadings plot
            fig, ax = plt.subplots()
            ax.scatter(pca_results['loadings'][:, 0], pca_results['loadings'][:, 1], color='blue')
            ax.set_title('loadings plot')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            for i in range(pca_results['loadings'].shape[0]):
                ax.text(pca_results['loadings'][i, 0], pca_results['loadings'][i, 1], 'x'+str(i+1))
            fig.show()

            # PCA in the context of the raw data
            fig, ax = plt.subplots()
            ax.scatter(x1, x2, color='blue')
            ax.plot([0, -20*pca_results['loadings'][0, 0]], [0, -20*pca_results['loadings'][1, 0]],
                    color='red', linewidth=3)
            ax.plot([0, 20 * pca_results['loadings'][0, 1]], [0, 20 * pca_results['loadings'][1, 1]],
                    color='red', linewidth=3)
            ax.set_title('raw data and PC axis')
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()

            # keep only the first dimension
            data_reconstructed = np.matmul(pca_results['scores'][:, 0].reshape((200, 1)), pca_results['loadings'][:, 0].reshape((1, 2)))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title('reconstructed data using PC1')
            ax.scatter(data_reconstructed[:, 0], data_reconstructed[:, 1], color='blue')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()

    elif example_index == 2:
        # ----------------------------------------------------------------
        # PCA on completely random data
        # ----------------------------------------------------------------
        # 1. generate raw data
        x1 = np.random.normal(loc=0, scale=0.5, size=num_of_samples)
        x2 = np.random.normal(loc=0, scale=0.5, size=num_of_samples)

        # 2. visualize the raw data
        fig, ax = plt.subplots()
        ax.scatter(x1, x2, color='blue')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        fig.show()

        # 3. do PCA
        data_for_analysis = np.column_stack((x1, x2))

        object_pca = d_PCA(num_of_components=min(data_for_analysis.shape[0], data_for_analysis.shape[1]))
        pca_results = object_pca.fit_transform(x=data_for_analysis, corr_logic=use_corr)

        # scree plot
        fig, ax = plt.subplots()
        ax.set_title('scree plot')
        ax.scatter(range(len(pca_results['percent_variance_explained'])), pca_results['percent_variance_explained'], color='blue')
        ax.set_ylim((-10.0, 110.0))
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance')
        fig.show()

        # scores plot
        fig, ax = plt.subplots()
        ax.set_title('scores plot')
        ax.scatter(pca_results['scores'][:, 0], pca_results['scores'][:, 1], color='blue')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('loadings plot')
        ax.scatter(pca_results['loadings'][:, 0], pca_results['loadings'][:, 1], color='blue')
        for i in range(pca_results['loadings'].shape[0]):
            ax.text(pca_results['loadings'][i, 0], pca_results['loadings'][i, 1], 'x' + str(i + 1))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # PCA in the context of the raw data
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('raw data and PC axis')
        ax.scatter(x1, x2, color='blue')
        k=3
        ax.plot([0, (-1)*k*pca_results['loadings'][0,0]], [0, (-1)*k*pca_results['loadings'][1,0]],
                color='red', linewidth=3)
        ax.plot([0, k * pca_results['loadings'][0, 1]], [0, k * pca_results['loadings'][1, 1]],
                color='red',linewidth=3)
        ax.set_aspect('equal', 'box')
        fig.show()
        plt.close('all')

    elif example_index == 3:
        # ----------------------------------------------------------------
        # PCA on toy data
        # ----------------------------------------------------------------
        # 1. get the raw data
        in_file_name = "../data/dataset_1.csv"
        dataIn = pd.read_csv(in_file_name)

        # 2. visualize the raw data: x vs y
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("raw data")
        ax.scatter(dataIn['x1'], dataIn['x2'], color='blue')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        fig.show()

        # x vs z
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("raw data")
        ax.scatter(dataIn['x1'], dataIn['x3'], color='blue')
        ax.set_xlabel('x1')
        ax.set_ylabel('x3')
        fig.show()

        # 3. do PCA
        data_for_analysis = dataIn.to_numpy()

        object_pca = d_PCA(num_of_components=min(data_for_analysis.shape[0], data_for_analysis.shape[1]))
        pca_results = object_pca.fit_transform(x=data_for_analysis, corr_logic=use_corr)

        # scree plot
        fig, ax = plt.subplots()
        ax.set_title('scree plot')
        ax.scatter(range(len(pca_results['percent_variance_explained'])), pca_results['percent_variance_explained'], color='blue')
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        ax.set_ylim((-10.0, 110.0))
        fig.show()

        # scores plot
        fig, ax = plt.subplots()
        ax.set_title('scores plot')
        ax.scatter(pca_results['scores'][:, 0], pca_results['scores'][:, 1], color='blue')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig, ax = plt.subplots()
        ax.set_title('loadings plot')
        ax.scatter(pca_results['loadings'][:, 0], pca_results['loadings'][:, 1], color='blue')
        for i in range(pca_results['loadings'].shape[0]):
            ax.text(pca_results['loadings'][i, 0], pca_results['loadings'][i, 1], 'x' + str(i + 1))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()
        plt.close('all')

    elif example_index == 4:

        # ----------------------------------------------------------------
        # PCA on real data
        # ----------------------------------------------------------------
        in_file_name = '../data/SCLC_study_output_filtered_2.csv'
        dataIn = pd.read_csv(in_file_name, header=0, index_col=0)

        data_for_analysis = dataIn.to_numpy()

        object_pca = d_PCA(num_of_components=min(data_for_analysis.shape[0], data_for_analysis.shape[1]))
        pca_results = object_pca.fit_transform(x=data_for_analysis, corr_logic=use_corr)

        # scree plot
        fig, ax = plt.subplots()
        ax.set_title('scree plot')
        ax.scatter(range(len(pca_results['percent_variance_explained'])), pca_results['percent_variance_explained'], color='blue')
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        ax.set_ylim((-10.0, 110.0))
        fig.show()

        # scores plot
        fig, ax = plt.subplots()
        ax.set_title('scores plot')
        ax.scatter(pca_results['scores'][:, 0], pca_results['scores'][:, 1], color='blue')
        ax.scatter(pca_results['scores'][0:20, 0], pca_results['scores'][0:20, 1], color='red')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig, ax = plt.subplots()
        ax.set_title('loadings plot')
        ax.scatter(pca_results['loadings'][:, 0], pca_results['loadings'][:, 1], color='blue')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()
        plt.close('all')

    elif example_index == 5:
        # ----------------------------------------------------------------
        # use sklearn
        # ----------------------------------------------------------------
        in_file_name = '../data/SCLC_study_output_filtered_2.csv'
        dataIn = pd.read_csv(in_file_name, header=0, index_col=0)

        num_of_samples = dataIn.shape[0]
        num_of_variables = dataIn.shape[1]
        sample_names = dataIn.index.values
        variable_names = dataIn.columns.values

        # data pre-processing
        # standardize each variable by computing its z-score
        data_for_analysis_standardized = StandardScaler(with_mean=True, with_std=True).fit_transform(dataIn)

        num_of_components = min(num_of_samples, num_of_variables)

        object_PCA = PCA(n_components=num_of_components)

        PCA_fit_results = object_PCA.fit(data_for_analysis_standardized)

        # scree plot
        fig, ax = plt.subplots()
        ax.scatter(range(len(PCA_fit_results.explained_variance_ratio_)),
                   PCA_fit_results.explained_variance_ratio_,
                   color='blue', s=config.marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("principal component index")
        ax.set_ylabel("explained variance ratio")
        ax.set_title("scree plot")
        fig.show()
        out_file_name = "../results/PCA_scree_plot.pdf"
        fig.savefig(out_file_name)

        PCA_scores = object_PCA.fit_transform(data_for_analysis_standardized)

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(PCA_scores[0:20, 0], PCA_scores[0:20, 1], color='blue', label='NSCLC')
        ax.scatter(PCA_scores[20:40, 0], PCA_scores[20:40, 1], color='red', label='SCLC')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('scores plot')

        for i in range(PCA_scores.shape[0]):
            ax.text(PCA_scores[i, 0], PCA_scores[i, 1], sample_names[i])

        ax.legend()
        fig.show()
        out_file_name = "../results/PCA_scores_plot.pdf"
        fig.savefig(out_file_name)

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(PCA_fit_results.components_[0, :], PCA_fit_results.components_[1, :],
                   color='blue', s=config.marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('loadings plot')

        for i in range(num_of_variables):
            ax.text(PCA_fit_results.components_[0, i], PCA_fit_results.components_[1, i], variable_names[i])

        fig.show()
        out_file_name = "../results/PCA_loadings_plot.pdf"
        fig.savefig(out_file_name)

        #export loadings
        PCA_loadings = pd.DataFrame((PCA_fit_results.components_.T)[:, 0:2],
                                    index=variable_names,
                                    columns=['PC1', 'PC2'])
        out_file_name = "../results/PCA_loadings.xlsx"
        PCA_loadings.to_excel(out_file_name)

    elif example_index == 6:
        # 1. get the raw data
        in_file_name = "../data/dataset_1.csv"
        data_in = pd.read_csv(in_file_name)

        # 2. prepare data
        data_for_analysis = data_in.loc[:, ['x1', 'x2']]
        data_for_analysis = data_for_analysis.to_numpy()

        data_for_analysis_standardized = StandardScaler(with_mean=True, with_std=True).fit_transform(data_for_analysis)

        num_of_samples = data_for_analysis.shape[0]
        num_of_variables = data_for_analysis.shape[1]
        num_of_components = min(num_of_samples, num_of_variables)

        # 3 do PCA using d_PCA using raw data
        object_pca = d_PCA(num_of_components=num_of_components)
        pca_results = object_pca.fit_transform(x=data_for_analysis, corr_logic=use_corr)

        k = 5
        fig, ax = plt.subplots()
        ax.scatter(data_for_analysis[:, 0], data_for_analysis[:, 1], color='blue', label='raw')
        ax.scatter(data_for_analysis_standardized[:, 0], data_for_analysis_standardized[:, 1], color='green', label='z_score')
        ax.plot([0, -k * pca_results['loadings'][0, 0]], \
                [0, -k * pca_results['loadings'][1, 0]],
                color='red', linewidth=3)
        ax.plot([0, k * pca_results['loadings'][0, 1]], \
                [0, k * pca_results['loadings'][1, 1]],
                color='red', linewidth=3, label='PC axis')
        ax.set_title('raw data and PC axis')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        fig.show()

        # 4. do PCA using d_PCA using standardized data
        object_pca = d_PCA(num_of_components=num_of_components)
        pca_results = object_pca.fit_transform(x=data_for_analysis_standardized, corr_logic=use_corr)

        k = 5
        fig, ax = plt.subplots()
        ax.scatter(data_for_analysis[:, 0], data_for_analysis[:, 1], color='blue', label='raw')
        ax.scatter(data_for_analysis_standardized[:, 0], data_for_analysis_standardized[:, 1], color='green', label='z_score')
        ax.plot([0, k * pca_results['loadings'][0, 0]], \
                [0, k * pca_results['loadings'][1, 0]],
                color='red', linewidth=3)
        ax.plot([0, k * pca_results['loadings'][0, 1]], \
                [0, k * pca_results['loadings'][1, 1]],
                color='red', linewidth=3, label='PC axis')
        ax.set_title('raw data and PC axis')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        fig.show()

        # 5. do PCA using sklearn using raw data
        object_pca = PCA(n_components=num_of_components)
        pca_fit_results = object_pca.fit(X=data_for_analysis)

        k = 5
        fig, ax = plt.subplots()
        ax.scatter(data_for_analysis[:, 0], data_for_analysis[:, 1], color='blue', label='raw')
        ax.scatter(data_for_analysis_standardized[:, 0], data_for_analysis_standardized[:, 1], color='green', label='z_score')
        ax.plot([0, -k * pca_fit_results.components_[0, 0]], [0, -k * pca_fit_results.components_[1, 0]],
                color='red', linewidth=3)
        ax.plot([0, k * pca_fit_results.components_[0, 1]], [0, k * pca_fit_results.components_[1, 1]],
                color='red', linewidth=3, label='PC axis')
        ax.set_title('raw data and PC axis')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        fig.show()

        # 6. do PCA using sklearn using standardized data
        object_pca = PCA(n_components=num_of_components)
        pca_fit_results = object_pca.fit(X=data_for_analysis_standardized)

        # scores plot
        pca_scores = object_pca.fit_transform(data_for_analysis_standardized)

        # PCA in the context of the raw data
        k = 5
        fig, ax = plt.subplots()
        ax.scatter(data_in['x1'], data_in['x2'], color='blue', label='raw')
        ax.scatter(data_for_analysis_standardized[:, 0], data_for_analysis_standardized[:, 1], color='green', label='z_score')
        ax.plot([0, -k * pca_fit_results.components_[0, 0]], [0, -k * pca_fit_results.components_[1, 0]],
                color='red', linewidth=3)
        ax.plot([0, k * pca_fit_results.components_[0, 1]], [0, k * pca_fit_results.components_[1, 1]],
                color='red', linewidth=3, label='PC axis')
        ax.set_title('raw data and PC axis')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        fig.show()

    else:
        print('Unknown example_index!')

#this helps in calling the method main and run the code
if __name__ == '__main__':
    main()
