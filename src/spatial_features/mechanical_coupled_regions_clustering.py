# -*- coding: utf-8 -*-
"""
Library of functions used for identifying mechanically coupled nuclei in tissues


Available Functions:
vis_mechanically_coupled_regions: Plot the clusters
clustering_mech_coupled_regions: Spatially cluster ellongated nuclei.

clusterability:Summarise the clustering results
cluster_membership_occupancy: Characterise the number, area and density of nuclei in a cluster
cluster_angular_homogenity:Characterise the dispersion of angles within clusters
tissue_angular_homogenity: Characterise the dispersion of angles within clusters
cluster_spatial_positioning:Characterise relative distances between identified clusters

"""

# import libraries
from sklearn.cluster import DBSCAN
from skimage import measure
from math import degrees, sqrt
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from statistics import median
from tifffile import imread
import pandas as pd
import sklearn.metrics as sklm
import scipy.spatial as ss

class Clusterability_Features:
    def __init__(self,dbscn_output = np.zeros(shape=(6,))):
        self.Number_nuclei_in_tissue = dbscn_output[0]
        self.Number_of_ellongated_nuclei_in_tissue = dbscn_output[1]
        self.Number_of_clusters = dbscn_output[2]
        self.Number_of_ellongated_nuclei_unclusted = dbscn_output[3]
        self.Number_of_ellongated_nuclei_clustered = dbscn_output[4]
        self.Silohhouette_score = dbscn_output[5]

class Cluster_Membership_Features:
    def __init__(self,membership = np.zeros(shape=(8,))):
        self.Median_num_cluster_members = membership[0]
        self.Min_num_cluster_members = membership[1]
        self.Max_num_cluster_members = membership[2]
        self.StdDev_num_cluster_members = membership[3]
        self.CV_num_cluster_members = membership[4]
        self.CD_num_cluster_members = membership[5]
        self.IQR_num_cluster_members = membership[6]
        self.Q_CD_num_cluster_members = membership[7]

class Cluster_Area_Features:
    def __init__(self,chull_ad = np.zeros(shape=(9,))):
        self.Total_cluster_area = chull_ad[0]
        self.Median_cluster_area = chull_ad[1]
        self.Min_cluster_area = chull_ad[2]
        self.Max_cluster_area = chull_ad[3]
        self.StdDev_cluster_area = chull_ad[4]
        self.CV_cluster_area = chull_ad[5]
        self.CD_cluster_area = chull_ad[6]
        self.IQR_cluster_area = chull_ad[7]
        self.Q_CD_cluster_area = chull_ad[8]

class Cluster_Density_Features:
    def __init__(self,chull_ad = np.zeros(shape=(8,))):
        self.Median_cluster_dens = chull_ad[0]
        self.Min_cluster_dens = chull_ad[1]
        self.Max_cluster_dens = chull_ad[2]
        self.StdDev_cluster_dens = chull_ad[3]
        self.CV_cluster_dens = chull_ad[4]
        self.CD_cluster_dens = chull_ad[5]
        self.IQR_cluster_dens = chull_ad[6]
        self.Q_CD_cluster_dens = chull_ad[7]

class Cluster_Angular_Homogeneity:
    def __init__(self,cangl = np.zeros(shape=(15,))):
        self.Med_StdDev_angles_cluster = cangl[0]
        self.Med_CV_angles_cluster = cangl[1]
        self.Med_CD_angles_cluster = cangl[2]
        self.Med_IQR_angles_cluster = cangl[3]
        self.Med_Q_CD_angles_cluster = cangl[4]
        self.SD_StdDev_angles_cluster = cangl[5]
        self.SD_CV_angles_cluster = cangl[6]
        self.SD_CD_angles_cluster = cangl[7]
        self.SD_IQR_angles_cluster = cangl[8]
        self.SD_Q_CD_angles_cluster = cangl[9]
        self.StdDev_angles_noise = cangl[10]
        self.CV_angles_noise = cangl[11]
        self.CD_angles_noise = cangl[12]
        self.IQR_angles_noise = cangl[13]
        self.Q_CD_angles_noise = cangl[14]

class Tissue_Angular_Homogeneity:
    def __init__(self,tangl = np.zeros(shape=(10,))):
        self.StdDev_nuc_angles = tangl[0]
        self.CV_nuc_angles = tangl[1]
        self.CD_nuc_angles = tangl[2]
        self.IQR_nuc_angles = tangl[3]
        self.Q_CD_nuc_angles = tangl[4]
        self.StdDev_elg_nuc_angles = tangl[5]
        self.CV_elg_nuc_angles = tangl[6]
        self.CD_elg_nuc_angles = tangl[7]
        self.IQR_elg_nuc_angles = tangl[8]
        self.Q_CD_elg_nuc_angles = tangl[9]

class Cluster_Relative_Distances:
    def __init__(self, cdist = np.zeros(shape=(16,))):
        self.Median_bb_cluster_dist = cdist[0]
        self.Min_bb_cluster_dist = cdist[1]
        self.Max_bb_cluster_dist = cdist[2]
        self.StdDev_bb_cluster_dist = cdist[3]
        self.CV_bb_cluster_dist = cdist[4]
        self.CD_bb_cluster_dist = cdist[5]
        self.IQR_bb_cluster_dist = cdist[6]
        self.Q_CD_bb_cluster_dist = cdist[7]

        self.Median_cc_cluster_dist = cdist[8]
        self.Min_cc_cluster_dist = cdist[9]
        self.Max_cc_cluster_dist = cdist[10]
        self.StdDev_cc_cluster_dist = cdist[11]
        self.CV_cc_cluster_dist = cdist[12]
        self.CD_cc_cluster_dist = cdist[13]
        self.IQR_cc_cluster_dist = cdist[14]
        self.Q_CD_cc_cluster_dist = cdist[15]

def distribution_statistics(feat):
    """
    Function takes in an array and returns some central and dispersion measures
    Outputs in order 1.median, 2.min, 3.max, 4.standard deviation, 5.Coefficient of variation (Std/Median), 
    6.Coefficient of dispersion (Var/Median),7.Interquantile range and 8.Quartile coeefficient of dispersion
    """
    
    return [np.median(feat),np.min(feat),np.max(feat),np.std(feat),
            np.std(feat)/abs(np.median(feat)), (np.var(feat))/abs(np.median(feat)),
            np.subtract(*np.percentile(feat, [75, 25])), 
           np.subtract(*np.percentile(feat, [75, 25]))/np.add(*np.percentile(feat, [75, 25]))]



def clusterability(features,data):
    """Function to summarise the clustering results
        Args: 
            features: Nuclear properties
            data: results of clustering
    """
    #Get together features describing clustering results
    n_nuclei=len(features['label'])
    n_ellongated_nuc=len(data['label'])
    n_clusters = len(set(data['clusters'])-{-1}) # since -1 element denotes noice
    n_uncoupled_nuclei = list(data['clusters']).count(-1) # noise
    n_coupled_nuclei = len(data['clusters'])-n_uncoupled_nuclei
    
    if n_clusters>=2:
        #calculate clustering robustness without noise
        Silohhouette_score= sklm.silhouette_score(data.drop(['clusters','label'],axis=1)[data['clusters']> -1], data['clusters'][data['clusters']> -1])
    else:
        Silohhouette_score = 'NA'
    basic_clustering_features = [Clusterability_Features([n_nuclei,n_ellongated_nuc,n_clusters, n_uncoupled_nuclei,n_coupled_nuclei,Silohhouette_score])]
    basic_clustering_features = pd.DataFrame([o.__dict__ for o in basic_clustering_features])

    return basic_clustering_features

def tissue_angular_homogenity(features,ell_threshold = 0.9):
    """Function to characterise the dispersion of angles within clusters
    Args:
        features: Nuclear properies; orientation and eccentricity
    """
    ecc=features['eccentricity']
    angles=np.vectorize(degrees)(features['orientation'])
    angles=np.where(angles > 0, angles, abs(angles)+90)

    #Filter to get only elongated nuclei
    relevant_angles=(angles)[ecc > ell_threshold]
    
    #Measuring orientation dispersion of all nuclei in the tissue
    (_,_,_,
    std_orientation_tissue,CV_orientation_tissue,CD_orientation_tissue,
    IQR_orientation_tissue,Quartile_CD_orientation_tissue)= distribution_statistics(angles)
    #Measuring orientation dispersion of only ellongated nuclei in the tissue
    (_,_,_,
    std_orientation_elong,CV_orientation_elong,CD_orientation_elong,
    IQR_orientation_elong,Quartile_CD_orientation_elong)= distribution_statistics(relevant_angles)

    t_angles=[Tissue_Angular_Homogeneity([std_orientation_tissue,CV_orientation_tissue,
                                          CD_orientation_tissue,IQR_orientation_tissue,
                                          Quartile_CD_orientation_tissue,
                                          std_orientation_elong,CV_orientation_elong,
                                          CD_orientation_elong,IQR_orientation_elong,
                                          Quartile_CD_orientation_elong])]
    t_angles = pd.DataFrame([o.__dict__ for o in t_angles])

    return t_angles
              
def cluster_angular_homogenity(data):
    """Function to characterise the dispersion of angles within clusters
    Args:
        data: Results of clustering: centroids, angles and cluster membership
    """
    
    n_clusters = len(set(data['clusters'])-{-1}) # since -1 element denotes noice
    if n_clusters <1:
        #Setting cluster angluar features to default
        c_angles=[Cluster_Angular_Homogeneity()]
        c_angles = pd.DataFrame([o.__dict__ for o in c_angles])

    elif n_clusters >=1:
        #Summarizing dispersion statistics of the orientations of the clustered nuclei
        # For each cluster and at the tissue level (elongated and all) we measure the disperision statistics

        #Measuring clusterwise orientation dispersion 
        std_orientation=data.groupby('clusters')['angles'].std().array
        CV_orientation=data.groupby('clusters')['angles'].std().array/abs(data.groupby('clusters')['angles'].median().array)
        CD_orientation=(data.groupby('clusters')['angles'].var().array)/abs(data.groupby('clusters')['angles'].median().array)
        IQR_orientation=(data.groupby('clusters')['angles'].quantile(.75).array-data.groupby('clusters')['angles'].quantile(.25).array)
        Quartile_CD_orientation=((data.groupby('clusters')['angles'].quantile(.75).array-data.groupby('clusters')['angles'].quantile(.25).array))/(((data.groupby('clusters')['angles'].quantile(.75).array+data.groupby('clusters')['angles'].quantile(.25).array)))

        c_angles=[Cluster_Angular_Homogeneity([np.median(np.delete(std_orientation,0)),
                                               np.median(np.delete(CV_orientation,0)),
                                                np.median(np.delete(CD_orientation,0)),
                                                np.median(np.delete(IQR_orientation,0)),
                                                np.median(np.delete(Quartile_CD_orientation,0)),                           
                                                np.std(np.delete(std_orientation,0)),
                                                np.std(np.delete(CV_orientation,0)),
                                                np.std(np.delete(CD_orientation,0)),
                                                np.std(np.delete(IQR_orientation,0)),
                                                np.std(np.delete(Quartile_CD_orientation,0)),
                                                std_orientation[0],CV_orientation[0],
                                               CD_orientation[0],IQR_orientation[0],
                                               Quartile_CD_orientation[0]])]
        c_angles = pd.DataFrame([o.__dict__ for o in c_angles])

        
    return c_angles        

def cluster_spatial_positioning(data):
    """Function to characterise relative distances between identified clusters
    Args:
        data: Results of clustering: centroids, angles and cluster membership
    """
    
    n_clusters = len(set(data['clusters'])-{-1}) # since -1 element denotes noice
    if n_clusters <2:
        #Setting cluster angluar features to default
        cdist=[Cluster_Relative_Distances()]
        cdist = pd.DataFrame([o.__dict__ for o in cdist])

    elif n_clusters >=2:
       # Here we implement two approaches for measuring distances between clustes:
        # (1) border-boder distances and (2) centroid-centroid distances. 
        # We compute dispersion measures for the distances obtained. 
        
        d = dict(tuple(data.groupby('clusters')))
        d.pop(-1, None)

        min_dist_between_clusters=np.row_stack([[np.amin(ss.distance_matrix(np.column_stack([d[i]['X'].array,d[i]['Y'].array]), 
                                                 np.column_stack([d[j]['X'].array,d[j]['Y'].array]))) for j in d.keys()] for i in d.keys()])
        min_dist_between_clusters=np.delete(list(set(np.frombuffer(min_dist_between_clusters))) ,0)

        cen_dist_between_clusters=ss.distance_matrix(np.row_stack([(np.mean(d[i]['X'].array),np.mean(d[i]['Y'].array)) for i in d.keys()]),
                                              np.row_stack([(np.mean(d[i]['X'].array),np.mean(d[i]['Y'].array)) for i in d.keys()]))
        cen_dist_between_clusters=np.delete(list(set(np.frombuffer(cen_dist_between_clusters))) ,0)

        (avg_bor_bor_dist_cluster,min_bor_bor_dist_cluster,max_bor_bor_dist_cluster,
        std_bor_bor_dist_cluster,CV_bor_bor_dist_cluster,CD_bor_bor_dist_cluster,
        IQR_bor_bor_dist_cluster,Quartile_CD_bor_bor_dist_cluster)= distribution_statistics(min_dist_between_clusters)

        (avg_cen_cen_dist_cluster,min_cen_cen_dist_cluster,max_cen_cen_dist_cluster,
        std_cen_cen_dist_cluster,CV_cen_cen_dist_cluster,CD_cen_cen_dist_cluster,
        IQR_cen_cen_dist_cluster,Quartile_CD_cen_cen_dist_cluster)= distribution_statistics(cen_dist_between_clusters)

        cdist = [Cluster_Relative_Distances([avg_bor_bor_dist_cluster,min_bor_bor_dist_cluster,max_bor_bor_dist_cluster,
                                     std_bor_bor_dist_cluster,CV_bor_bor_dist_cluster,CD_bor_bor_dist_cluster,
                                     IQR_bor_bor_dist_cluster,Quartile_CD_bor_bor_dist_cluster,
                                     avg_cen_cen_dist_cluster,min_cen_cen_dist_cluster,max_cen_cen_dist_cluster,
                                     std_cen_cen_dist_cluster,CV_cen_cen_dist_cluster,CD_cen_cen_dist_cluster,
                                     IQR_cen_cen_dist_cluster,Quartile_CD_cen_cen_dist_cluster])]
        
        cdist = pd.DataFrame([o.__dict__ for o in cdist])

        
    return cdist

def cluster_membership_occupancy(data):
    """Function to characterise the number, area and density of nuclei in a cluster
    Args:
        data: Results of clustering: centroids, angles and cluster membership

    """
    
    
    
    n_clusters = len(set(data['clusters'])-{-1}) # since -1 element denotes noice

    if n_clusters == 0:
        membership=[Cluster_Membership_Features()]
        membership = pd.DataFrame([o.__dict__ for o in membership])
        areas=[Cluster_Area_Features()]
        areas = pd.DataFrame([o.__dict__ for o in areas])
        density=[Cluster_Density_Features()]
        density = pd.DataFrame([o.__dict__ for o in density])
        all_features = pd.concat([membership.reset_index(drop=True), areas.reset_index(drop=True),
                                 density], axis=1)
        
    elif n_clusters ==1:
        #obtain_total_cluster_areas_set_everything_else_to_default
        membership=[Cluster_Membership_Features()]
        membership = pd.DataFrame([o.__dict__ for o in membership])
        d = dict(tuple(data.groupby('clusters')))
        d.pop(-1, None)
        
        try:
            cluster_chull_areas=[ss.ConvexHull(np.column_stack([d[i]['X'].array,d[i]['Y'].array])).volume for i in d.keys()]
        except:
            cluster_chull_areas=[0,0,0]
        
        Total_cluster_area=np.sum(cluster_chull_areas)
        areas=[Cluster_Area_Features([Total_cluster_area,0,0,0,0,0,0,0,0])]
        areas = pd.DataFrame([o.__dict__ for o in areas])
        density=[Cluster_Density_Features()]
        density = pd.DataFrame([o.__dict__ for o in density])
        all_features = pd.concat([membership.reset_index(drop=True), areas.reset_index(drop=True),
                                 density], axis=1)
        
    elif n_clusters >1:
        #Summarizing the cluster membership distribution characteristics
        cluster_size_nums=np.delete(np.array(data.groupby(['clusters']).size()),0)
        (cluster_size_nums_avg,cluster_size_nums_min,cluster_size_nums_max,
        cluster_size_nums_std,cluster_size_nums_cv,cluster_size_nums_cd,
        cluster_size_nums_IQR,cluster_size_nums_Quartile_CD)= distribution_statistics(cluster_size_nums)

        #For each cluster calculate the area by calculating the area of the convex hull of cluster members
        # Note: concavehull implementation here might be a good addition as it will provide more imformative values. 

        d = dict(tuple(data.groupby('clusters')))
        d.pop(-1, None)
        try:
            cluster_chull_areas=[ss.ConvexHull(np.column_stack([d[i]['X'].array,d[i]['Y'].array])).volume for i in d.keys()]
        except:
            cluster_chull_areas=[0,0,0,0,0]
       

        (avg_cluster_area,min_cluster_area,max_cluster_area,
        std_cluster_area,CV_cluster_area,CD_cluster_area,
        IQR_cluster_area,Quartile_CD_cluster_area)= distribution_statistics(cluster_chull_areas)
        Total_cluster_area=np.sum(cluster_chull_areas)

        #Calculate cluster density: number of nuclei/ convex area of cluster
        cluster_density=np.divide(cluster_size_nums,cluster_chull_areas)
        (avg_cluster_density,min_cluster_density,max_cluster_density,
        std_cluster_density,CV_cluster_density,CD_cluster_density,
        IQR_cluster_density,Quartile_CD_cluster_density)= distribution_statistics(cluster_density)

        #return dataframe of features
        membership=[Cluster_Membership_Features([cluster_size_nums_avg,cluster_size_nums_min,cluster_size_nums_max,
                    cluster_size_nums_std,cluster_size_nums_cv,cluster_size_nums_cd,
                    cluster_size_nums_IQR,cluster_size_nums_Quartile_CD])]
        membership = pd.DataFrame([o.__dict__ for o in membership])
        areas=[Cluster_Area_Features([Total_cluster_area,
                    avg_cluster_area,min_cluster_area,max_cluster_area,
                         std_cluster_area,CV_cluster_area,CD_cluster_area,
                         IQR_cluster_area,Quartile_CD_cluster_area])]
        areas = pd.DataFrame([o.__dict__ for o in areas])
        density=[Cluster_Density_Features([avg_cluster_density,min_cluster_density,max_cluster_density,
                         std_cluster_density,CV_cluster_density,CD_cluster_density,
                         IQR_cluster_density,Quartile_CD_cluster_density])]
        density = pd.DataFrame([o.__dict__ for o in density])

        all_features = pd.concat([membership.reset_index(drop=True), areas.reset_index(drop=True),
                                 density], axis=1)
    return all_features

def clustering_mech_coupled_regions(features,dbscn_length=400,dbscn_min_size=15, ell_threshold=0.9):
    """ Function to spatially cluster ellongated nuclei.
    This function performs density based spatial clustering analysis using the orientation and centroid cordinates of the ellongated nuclei. 
    Args:
        features: dataframe of feature consisting of Centroid positions, eccentricity and orientation
        dbscn_length: radius to use in DBSCN algorithm
        dbscn_min_size: minimum size of the clusters to use in DBSCN algorithm
        ell_threshold: eccentricity threshold to use to filter elongated nuclei. 
    """
    
    ecc=features['eccentricity']
    angles=np.vectorize(degrees)(features['orientation'])
    angles=np.where(angles > 0, angles, abs(angles)+90)

    #Filter to get only elongated nuclei
    relevant_angles=(angles)[ecc > ell_threshold]
    cenx_rel=features['centroid-0'][ecc > ell_threshold]
    ceny_rel=features['centroid-1'][ecc > ell_threshold]
    labels_rel=features['label'][ecc > ell_threshold]
    cords=np.column_stack([cenx_rel,ceny_rel,relevant_angles])
    # Compute DBSCAN
    db = DBSCAN(eps=dbscn_length, min_samples=dbscn_min_size).fit(cords)
    clusters = db.labels_
    #save centroid angles and cluster identities
    clus_res=np.column_stack([cenx_rel,ceny_rel,relevant_angles,
                              clusters,labels_rel])
    df = pd.DataFrame(data=clus_res, columns=["X", "Y","angles","clusters","label"])
    
    return df

def vis_mechanically_coupled_regions(img_dir,output_dir,data,dbscn_length,dbscn_min_size,display_not_save=False):
    """ Function to plot the clusters
    Args:
        img_dir: path to image
        output_dir: path to output folder
        data: Results of clustering: centroids, angles and cluster membership
        dbscn_length: radius used for DBSCN
        dbscn_min_size: minimum cluster size od DBSCN
    """
    #Read in the image that is segmented/labelled for nuclei
    img=imread(img_dir)

    #save plots to show clusters
    fig = plt.figure(figsize=(6, 2))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #show segmented image labels
    ax0.imshow(img,aspect='auto') 
    ax0.axis('off')
    #nuclear centroid color-coded by their orientation
    img1=ax1.scatter(data["Y"], data["X"], c=data["angles"],s=1)
    ax1.set_xlim(0,img.shape[0])
    ax1.set_ylim(img.shape[1],0)
    plt.colorbar(img1)
    ax1.axis('off')

    # plot the cluster assignments
    img3=ax3.scatter(data[data["clusters"]> -1]["Y"], data[data["clusters"]> -1]["X"], 
                     c=data[data["clusters"]> -1]["clusters"],cmap="plasma",s=1)
    ax3.set_xlim(0,img.shape[0])
    ax3.set_ylim(img.shape[1],0)
    ax3.axis('off')

    #add titles
    ax0.title.set_text('Segmented Image')
    ax1.title.set_text('Filtered Orientation')
    ax3.title.set_text('Clusters')

    if display_not_save:
        plt.show()
    else: 
        plt.savefig((output_dir+"/"+img_dir.rsplit('/', 1)[-1][:-4]+"_"+str(dbscn_length)+"_"+ str(dbscn_min_size)+".png"),dpi=600, bbox_inches = 'tight',pad_inches = 0)
        fig.clf()
        plt.close(fig)
        plt.close('all')
        
    
    del fig,ax0,ax1,ax3,img1,img3
    