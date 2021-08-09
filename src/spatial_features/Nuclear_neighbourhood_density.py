# -*- coding: utf-8 -*-
"""
Library for computing local density of each nucleus.

Available Functions:
    distance_to_k_nneigh: Obtain the distance to the kth nearest neighbour for each nucleus
    num_neigbours_in_Radius: Obtain the number of neighbours in a given Radius

"""

from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import numpy as np 
import scipy.spatial as ss


def distance_to_k_nneigh(features, k_values):
    """Function to obtain the distance to the kth nearest neighbour for each nucleus
        Args:
            Features: dataframe of nuclear properties [labels and centroid]
            k_values: list containing the K nearest neighbours
    """
    cords=np.column_stack((features['centroid-0'],features['centroid-1']))
    nbrs = NearestNeighbors(n_neighbors=(max(k_values)+1), algorithm='ball_tree').fit(cords)
    distances, indices = nbrs.kneighbors(cords)
    
    distances = pd.DataFrame(distances)[k_values]
    distances.columns = [str(col) + '_NN' for col in distances.columns]

    distances = pd.concat([distances.reset_index(drop=True),
                          pd.DataFrame(features['label'], columns=["label"])], axis=1)
    return distances

def num_neigbours_in_Radius(features, R_values):
    """Function to obtain the number of neighbours in a given Radius
        Args:
            Features: dataframe of nuclear properties [labels and centroid]
            R_values: list containing the R radii
    """
    cords=np.column_stack((features['centroid-0'],features['centroid-1']))
    #obtain the distance matrix 
    dist_matrix=ss.distance.squareform(ss.distance.pdist(cords, 'euclidean'))
    
    
    #Defining neighbourhood radius "R" and counting the number of nuclei in "R"
    num_neigh = features['label']
    for R in R_values:
        mask=((dist_matrix<R) & (dist_matrix >0)).astype(float)
        mask[mask==0]= np.NaN
        neighbors=np.nansum(mask, axis=1)
        num_neigh=np.column_stack((num_neigh,neighbors))
    
    num_neigh = pd.DataFrame(num_neigh)
    num_neigh.columns= ['label']+['num_neigh_' + str(R) for R in R_values]

    return num_neigh