# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
from tifffile import imread
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm

def vis_score(T_id,data,img_path,output_path,min_display_val = 0.1, max_display_val = 0.7,pt_size=0.5, display_not_save=True):
    """Function to visualize the MGS of a given image
       
       Parameters:
           T_id: Tissue id/Name of the image
           data: grouped dataframe that contains the centroid and MGS of nuclei 
           img_path: path to the images
           output_path: path to store images
           min_display_val: miminum value to be plotted
           max_display_val: maximum value to be plotted
           display_not_save: display plot dont save
    """
    #img = imread(img_path+T_id+'.tif')
    img = imread(img_path)
    
    score=data.get_group(T_id)
    #save plots to show clusters
    fig = plt.figure(figsize=(9, 4))
    ax0 = fig.add_subplot(111,aspect='equal')
    
    
    #code the tumogenesis score
    img2=ax0.scatter(score['centroid-1'], score['centroid-0'], c=(score['MGS']),
                     s=pt_size,cmap='RdYlBu',
                    vmin=min_display_val, vmax=max_display_val)
    ax0.set_xlim(0,img.shape[0])
    ax0.set_ylim(img.shape[1],0)
    ax0.axis('off')
    plt.colorbar(img2)
    
    # plot if asked
    if display_not_save:
        plt.show()
    else: 
        plt.savefig((output_path+"/"+T_id+".png"),dpi=600, bbox_inches = 'tight',pad_inches = 0)
        fig.clf()
        plt.close(fig)
        plt.close('all')
        
def color_code_MGS_image(T_id,data,img_path,output_path):
    """Function to visualize the MGS of a given image by color coding labelled images
       
       Parameters:
           T_id: Tissue id/Name of the image
           data: grouped dataframe that contains the centroid and MGS of nuclei 
           img_path: path to the images
    """
    
    img = imread(img_path)
    
    score=data.get_group(T_id)
    score['label']=pd.to_numeric(score['nucid'].str.split('_').str[-1])

    t=np.multiply(0,np.array(img==1))
    for i in tqdm(range(len(score['label']))):
        t=t+np.multiply(np.array(score['MGS'])[i]+1,np.array(img==np.array(score['label'])[i]))

    im = Image.fromarray(t)
    im.save(output_path+"/"+T_id+ ".tif")
    
def plot_nuc_mgs_biomarker(path_to_DNA_image,path_to_MGS_image,path_to_biomarker_image):
    """Function for visualising score with DNA and biomarker labels
    """
    #Read in Images
    raw_image = imread(path_to_DNA_image)
    scored_image = plt.imread(path_to_MGS_image)
    biomarker = plt.imread(path_to_biomarker_image)
    
    # normalize images
    raw_image = ((raw_image-np.min(raw_image))/(np.max(raw_image)-np.min(raw_image)))*255
    raw_image = raw_image.astype(int)
    biomarker = ((biomarker-np.min(biomarker))/(np.max(biomarker)-np.min(biomarker)))*255
    biomarker = biomarker.astype(int)
    #Visulaise the data

    #save plots to show clusters
    fig = plt.figure(figsize=(12, 4))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    #show raw image 
    ax0.imshow(raw_image,aspect='auto',cmap='inferno') 
    ax0.axis('off')
    ax0.title.set_text('Image')
    #show segmented image
    ax1.imshow(scored_image,aspect='auto',cmap='Spectral') 
    ax1.axis('off')
    ax1.title.set_text('Nuclear health score')
    #show segmented image
    ax2.imshow(biomarker,aspect='auto',cmap='Greens_r') 
    ax2.axis('off')
    ax2.title.set_text('Biomarker')
    plt.show()
