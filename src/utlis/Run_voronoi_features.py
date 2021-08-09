# -*- coding: utf-8 -*-
import src.spatial_features.Voronoi_tessellation as VT
from tifffile import imread
from skimage import measure
import pandas as pd
import numpy as np

def extract_voronoi_features(img_path, show_plot=False):
    """Function that accepts an path to a segmented image and extract voronoi features
    """
    # Read in the image
    img= imread(img_path)
    image_width=img.shape[1]
    image_height=img.shape[0]

    #measure nuclear positions
    features=measure.regionprops_table(img,properties=('label','centroid'))
    #get voronoi map
    vor_image=VT.get_voronoi_map(centroids=np.stack((features['centroid-0'],features['centroid-1']),axis=1),
                   labels=features['label'],img_height=image_height,img_width=image_height)
    
    # plot if asked
    if show_plot:
        VT.plot_voronoi_map(img,vor_image)
    #extract features
    features=VT.extract_voronoi_cell_features(vor_image)
     
    features['Image'] = img_path.rsplit('/', 1)[-1][:-4]

    return features