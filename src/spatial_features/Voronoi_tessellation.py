# -*- coding: utf-8 -*-
"""
Library to obtain the voronoi tesselation for a given set of nuclear centroids

This module provides functions that one can use to obtain, plot and extract geometric features of the voronoi cells

Available Functions:
-get_voronoi_map:Obtain the voronoi diagram
-plot_voronoi_map:Function to plot the voronoi diagram : Describing centroid to boundary distances
-extract_voronoi_cell_features:Function to extract geometric features from voronoi cells
"""
import numpy as np
import pandas as pd 
from skimage import measure
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
from skimage.draw import ellipse, polygon, polygon_perimeter
import cv2 as cv
import pandas as pd

def get_voronoi_map(centroids,labels,img_height,img_width):
    """Function to obtain the voronoi diagram given the centroids and labels
    Args: 
        centroids
        labels
        image_height
        image_width
    """
    num_nuclei=len(labels)
    # obtain voronoi tesellation
    vor=Voronoi(centroids)
    #initialise  and build voronoi diagram
    image_voromap = np.zeros([img_height, img_width])
    for nuclear_index in range(0,num_nuclei):
        # gets the index of a voronoi region
        regions_ind = vor.point_region[nuclear_index]
        # gets the voronoi region
        region = vor.regions[regions_ind]
        # calculates the polygon of the region
        poly = vor.vertices[region]
        c0 = poly.min()
        c1 = poly[:,0].max()
        c2 = poly[:,1].max()

        if (c0>0) & (c2<img_width) & (c1<img_height):
            #  build the x and y vectors of the x and y polygon coordinates
            x = poly[:,0]
            y = poly[:,1]
     
            # Polygon clipping is throwing errors at times: Not sure why: for now if there are errors skip filling cells. 
            try: 
                # fills with values index of the nucleus
                rr, cc = polygon(x, y)
                rrp,ccp = polygon_perimeter(x,y)
                image_voromap[rr-1, cc-1] = labels[nuclear_index]
                image_voromap[rrp-1, ccp-1] = 0
            except:
                print("An exception occurred in Vornoi Tesselation: polygon clipping") 
    return image_voromap.astype(int)

def plot_voronoi_map(nuc_seg_image,voronoi_image):
    """Function to plot the voronoi diagram 
    Args: 
        nuc_seg_image
        voronoi_image
        centroids: nuclear centroids
    """
    #measure nuclear positions
    centroids=measure.regionprops_table(nuc_seg_image,properties=('label','centroid'))
    
    # plot the voronoi plot 
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(nuc_seg_image,aspect='auto',cmap="jet") 
    plt.imshow(voronoi_image, cmap='jet', alpha=0.5) # interpolation='none'
    plt.scatter(centroids['centroid-1'],centroids['centroid-0'], c="white",s=10)

def extract_voronoi_cell_features(voronoi_image):
    """ Function to extract geometric features from voronoi cells
    """
    #Extract features
    vor_features=measure.regionprops_table(voronoi_image, voronoi_image, properties=('label','centroid','orientation',
                                                             'area','perimeter',
                                                             'equivalent_diameter','major_axis_length',
                                                             'minor_axis_length','eccentricity'))
    vor_features = pd.DataFrame(vor_features)
    return vor_features