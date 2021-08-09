# -*- coding: utf-8 -*-
from tifffile import imread
import pandas as pd 
from skimage import measure
import src.spatial_features.Nuclear_neighbourhood_density as NND

def local_nuclear_density(img_dir):
    """
    This function calculates the local density of neighbours for each nucleus in a segmented image.
    
    Args:
        img_dir: Path to segmented image. 
    """
    
    # Read in the image
    img = imread(img_dir)

    #Get features for the individual nuclei in the image
    feat = measure.regionprops_table(img,properties = ('label','centroid'))
    
    #Compute the features
    knn_dist = NND.distance_to_k_nneigh(feat,[1,3,5,10,20])
    num_neigh_rad = NND.num_neigbours_in_Radius(feat, [20,50,100,150,200])
    all_density_features = pd.merge(num_neigh_rad,knn_dist, on="label", how="outer")
    
    all_density_features['Image'] = img_dir.rsplit('/', 1)[-1][:-4]
    return all_density_features
