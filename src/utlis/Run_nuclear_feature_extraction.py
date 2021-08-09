# -*- coding: utf-8 -*-
from tifffile import imread
import pandas as pd 
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import src.nuclear_features.Boundary_local_curvature as BLC
import src.nuclear_features.Boundary_global as BG
import src.nuclear_features.Int_dist_features as IDF
import src.nuclear_features.Img_texture as IT
from tqdm.notebook import tqdm

def run_nuclear_chromatin_feat_ext(raw_image_path, labelled_image_path,output_dir):
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features. 
    Note this has been used only for DAPI stained images
    Args:
        raw_image_path: path pointing to the raw image
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
    """
    labelled_image = imread(labelled_image_path)
    raw_image = imread(raw_image_path)
    
    # normalize images
    raw_image = ((raw_image-np.min(raw_image))/(np.max(raw_image)-np.min(raw_image)))*255
    raw_image = raw_image.astype(int)    
    
    #Get features for the individual nuclei in the image
    props = measure.regionprops(labelled_image,raw_image)

    #Measure scikit's built in features
    propstable = measure.regionprops_table(labelled_image,raw_image,cache=True,
                                   properties=['label', 'area','perimeter','bbox','bbox_area','convex_area',
                                               'equivalent_diameter','major_axis_length','minor_axis_length',
                                               'eccentricity','orientation',
                                                'centroid','weighted_centroid',
                                               'weighted_moments','weighted_moments_normalized',
                                               'weighted_moments_central','weighted_moments_hu',
                                                'moments','moments_normalized','moments_central','moments_hu'])
    propstable = pd.DataFrame(propstable)

    #measure other inhouse features
    all_features = pd.DataFrame()
    
    for i in tqdm(range(len(props))):
        temp = pd.concat([BLC.curvature_features(props[i].image,step=5).reset_index(drop=True),
                              BG.boundary_features(props[i].image,centroids=props[i].local_centroid).reset_index(drop=True),
                              IDF.intensity_features(props[i].image,props[i].intensity_image).reset_index(drop=True),
                              IT.texture_features(props[i].image,props[i].intensity_image,props[i].local_centroid),
                                 pd.DataFrame([i+1],columns=['label'])], axis=1)
        all_features = pd.concat([all_features, temp], ignore_index=True, axis=0)
    
    # Add in other related features for good measure   
    features=pd.merge(propstable,all_features, on="label")
    features['Concavity']=(features['convex_area']-features['area'])/features['convex_area']
    features['Solidity']=features['area']/features['convex_area']
    features['A_R']=features['minor_axis_length']/features['major_axis_length']
    features['Shape_Factor']=(features['perimeter']**2)/(4*np.pi*features['area'])
    features['Area_bbArea']=features['area']/features['bbox_area']
    features['Center_Mismatch']=np.sqrt((features['weighted_centroid-0']-features['centroid-0'])**2+
                                 (features['weighted_centroid-1']-features['centroid-1'])**2)
    features['Smallest_largest_Calliper']=features['Min_Calliper']/features['Max_Calliper']
    features['Frac_Peri_w_posi_curvature']=features['Len_posi_Curvature']/features['perimeter']
    features['Frac_Peri_w_neg_curvature']=features['Len_neg_Curvature'].replace(to_replace ="NA",value =0)/features['perimeter']
    features['Frac_Peri_w_polarity_changes']=features['nPolarity_changes']/features['perimeter']

    features['Image'] = labelled_image_path.rsplit('/', 1)[-1][:-4]
    # save the output
    features.to_csv(output_dir+"/"+labelled_image_path.rsplit('/', 1)[-1][:-4]+".csv")

    return features