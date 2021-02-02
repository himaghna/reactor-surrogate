"""Explore reactor data using unsupervised learning"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# set global plot params
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

class Dataset:
    def __init__(self, descriptor_src, response_src=None, response_cols=None):
        """
        Parameters
        ----------
        descriptor_src: str
            Path of csv file containing descriptors.
        response_src: str
            Path of csv file containing responses. Default is None.
        response_cols: list(str)
            Column names corresponding to responses in response_src csv file.
        
        """
        self.feature_dataframe = pd.read_csv(descriptor_src)
        self.resonse_dataframe = None
        if response_src is not None:
            self.resonse_dataframe = pd.read_csv(response_src)
        if response_cols is not None:
            self.response_dataframe = self.response_dataframe(response_cols)
        self.pca_transformed_dataset = None
        self.data_scaler = None
    
    def _apply_pca_transformation(self):
        """ Applies PCA transformation to the dataset
        Note
        ----
        Sets pca_transformed_dataset and data_scaler attributes

        """
        self.data_scaler = StandardScaler()
        scaled_X = self.data_scaler.fit_transform(
                                              self.feature_dataframe.to_numpy())
        pca = PCA()
        self.pca_transformed_dataset = pca.fit_transform(scaled_X)

    def project_in_principal_components(self,
                                        projection_axis_1=0, 
                                        projection_axis_2=1, **kwargs):
        """ Project the input data into two principal components 
        denoted by projection_axis_1 and projection_axis_1.

        Parameters
        ----------
        projection_axis_1: int
            First projection axis. 0-indexed i.e. supplying the value of 0 
            corresponds to selecting 1st principal component, 1 is second 
            principal componenet etc. Default is 0.
        projection_axis_2: int
            Second projection axis. 0-indexed i.e. supplying the value of 0 
            corresponds to selecting 1st principal component, 1 is second 
            principal component etc. Default is 1.
        **kwargs
            Key word arguments for modifying plotting behavior.
        
        """
        plot_params = {'alpha': 0.2,
                       's': 5,
                       'c': 'pink',
                       }
        if kwargs is not None:
            plot_params.update(kwargs)

        if self.pca_transformed_dataset is None:
            self._apply_pca_transformation()
        plt.scatter(self.pca_transformed_dataset[:, projection_axis_1],
                    self.pca_transformed_dataset[:, projection_axis_2],
                    **plot_params)
        plt.xlabel(f'Principal Component {projection_axis_1+1}', fontsize=20)
        plt.ylabel(f'Principal Component {projection_axis_2+1}', fontsize=20)
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-ds', '--descriptor_src', 
                        help='Path of descriptor filepath')
    parser.add_argument('-rs', '--response_src', default=None, 
                        help='Path of response filepath. Default is None.')
    parser.add_argument('-rc', '--response_column', default=None, 
                        help='Column in response csv file corresponding to' 
                              'response. Default is None.')
    args = parser.parse_args()
    data_set = Dataset(descriptor_src=args.descriptor_src, 
                       response_src=args.response_src, 
                       response_cols=args.response_column)
    data_set.project_in_principal_components(projection_axis_1=0, 
                                            projection_axis_2=1,)


