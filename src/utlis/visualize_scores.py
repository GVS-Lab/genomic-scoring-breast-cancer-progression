# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
from tifffile import imread
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm

def vis_score(T_id,data,img_path,output_path,display_not_save=True):
    """Function to visualize the MGS of a given image
       
       Parameters:
           T_id: Tissue id/Name of the image
           data: grouped dataframe that contains the centroid and MGS of nuclei 
           img_path: path to the images
    """
    #img = imread(img_path+T_id+'.tif')
    img = imread(img_path)
    
    score=data.get_group(T_id)
    #save plots to show clusters
    fig = plt.figure(figsize=(9, 4))
    ax0 = fig.add_subplot(111,aspect='equal')
    
    
    #code the tumogenesis score
    img2=ax0.scatter(score['centroid-1'], score['centroid-0'], c=(score['MGS']),
                     s=0.5,cmap='RdYlBu',
                    vmin=0.1, vmax=0.7)
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
    
    