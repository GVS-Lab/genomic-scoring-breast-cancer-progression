# -*- coding: utf-8 -*-
from tifffile import imread
import pandas as pd 
from skimage import measure
import src.spatial_features.mechanical_coupled_regions_clustering as MCR


def mechanically_coupled_regions(img_dir,output_dir,
                                 dbscn_length=400,dbscn_min_size=15,
                                 plot_vis_cluster=False):
    """
    This function calculates the degree of mechanically coupled nuclei in a image.
    It performs density based spatial clustering analysis using the orientation and centroid cordinates of the ellongated nuclei. 
    The orientation computation, ellongation filtering and cluster membership are visualised and the plots is saved in the output_dir. 
    It returns features of the clustering. See below for feature description 
    """
    #Read in the image that is segmented/labelled for nuclei
    img=imread(img_dir)

    #Get features for the individual nuclei in the image
    feat=measure.regionprops_table(img,properties = ('label','orientation','centroid','area','eccentricity'))
    
    #Perform clustering
    df=MCR.clustering_mech_coupled_regions(feat,dbscn_length,dbscn_min_size)
    
    #Save output
    df.to_csv(output_dir+"/"+img_dir.rsplit('/', 1)[-1][:-4]+"_"+str(dbscn_length)+"_"+ str(dbscn_min_size)+".csv")
    
    #Plot clustering if asked  
    if plot_vis_cluster:
        MCR.vis_mechanically_coupled_regions(img_dir,output_dir,df,dbscn_length,dbscn_min_size)
    
    #Sumarise the clustering results
    cluster_summary= MCR.clusterability(feat, df)
    
    #characterise the number, area and density of nuclei in a cluster
    membership = MCR.cluster_membership_occupancy(df)
    
    #characterise the nuclear angular distributions within a tissue and identified clusters
    tissue_ang_dist = MCR.tissue_angular_homogenity(feat)
    cluster_ang_dist = MCR.cluster_angular_homogenity(df)
     
    #characterise the relative positioning of clusters within a tissue
    cluster_positioning = MCR.cluster_spatial_positioning(df)
    
    all_features = pd.concat([cluster_summary.reset_index(drop=True),
                              membership.reset_index(drop=True),tissue_ang_dist.reset_index(drop=True),
                              cluster_ang_dist.reset_index(drop=True),
                                 cluster_positioning], axis=1)
    return all_features