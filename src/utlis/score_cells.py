# -*- coding: utf-8 -*-
# import libraries
import sys
sys.path.append("..")

from pathlib import Path
from glob import glob
import pandas as pd
import os
from tqdm.notebook import tqdm
import subprocess

from src.utlis.segmentation_stardist_model import segment_objects_stardist2d
from src.utlis.Run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
from src.utlis.visualize_scores import vis_score,color_code_MGS_image,plot_nuc_mgs_biomarker


def score_vis_with_groundtruth_data(DNA_labeled_regions,biomarker_labeled_region,
                                    path_to_Rscript = False,
                                    path_to_pretrained_datamodels = False, 
                                    path_to_seg_model = False, 
                                    path_to_output_segmented_nuclei = False, 
                                    path_to_output_nuclear_features = False, 
                                    path_to_output_MGS_file = False, 
                                    path_to_output_MGS_img = False):
    """ Score and visualise nuclei along with ground truth(biomarker data
    """
    if(not path_to_Rscript):
        path_to_Rscript = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)) ,"scoring_model/")
        
    if(not path_to_pretrained_datamodels):
        path_to_pretrained_datamodels = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)) ,"scoring_model/model/")
        
    if(not path_to_seg_model):
        path_to_seg_model = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)),'segmentation_model/')
        
    if(not path_to_output_segmented_nuclei):
        path_to_output_segmented_nuclei = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)) , "Segmented_nucleus/")
        
    if(not path_to_output_nuclear_features):
        path_to_output_nuclear_features = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions))  , "NMCO_features/")
        
    if(not path_to_output_MGS_file):
        path_to_output_MGS_file = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)), "MGS/")
        
    if(not path_to_output_MGS_img):
        path_to_output_MGS_img = os.path.join(os.path.dirname(os.path.dirname(DNA_labeled_regions)) , "MGS_img/")
        
        
    #create output directories if they do not exist
    Path(path_to_output_segmented_nuclei).mkdir(parents=True, exist_ok=True)
    Path(path_to_output_nuclear_features).mkdir(parents=True, exist_ok=True)
    Path(path_to_output_MGS_file).mkdir(parents=True, exist_ok=True)
    Path(path_to_output_MGS_img).mkdir(parents=True, exist_ok=True)

    #Segment nuclei based on a pretrained model
    
    print("Segmenting images.....")
    segment_objects_stardist2d(image_dir = DNA_labeled_regions,
                               output_dir_labels = path_to_output_segmented_nuclei,
                               output_dir_ijroi = False,
                               use_pretrained = False,
                               model_name ='tissue_nuclear_segmentation',
                               model_dir = path_to_seg_model
                               )
    ### Extract single nuclear features for all images 
    print("Computing nuclear and chromatin features....")
    dna_image_path = sorted(glob(DNA_labeled_regions + "/*tif"))
    seg_image_path = sorted(glob(path_to_output_segmented_nuclei + "*.tif"))
    for i in tqdm(range(len(dna_image_path))):
        run_nuclear_chromatin_feat_ext(dna_image_path[i], seg_image_path[i], path_to_output_nuclear_features)
    


    # Score the nuclei based on a pretrained model (this is run in R)
    print("Computing cell heath score (MGS)....")
    scriptpath=os.path.join(path_to_Rscript,"score_nuclei.R")
    subprocess.call(['Rscript',scriptpath, path_to_output_nuclear_features,path_to_pretrained_datamodels, path_to_output_MGS_file])
    print(scriptpath)
    
    # Obtain plots of MGS for each image
    tumogenesis_score=pd.read_csv(path_to_output_MGS_file+'MGS_for_plotting.csv') # read in the file
    grouped = tumogenesis_score.groupby(tumogenesis_score.Image) #group
    for i in tqdm(range(len(seg_image_path))): 
        image = seg_image_path[i].rsplit('/', 1)[-1][:-4]
        vis_score(image,grouped,seg_image_path[i],path_to_output_MGS_img,0.35,0.6,5,False)
    
    #Visualise the scores with biomarker/groundtruth data for all images in a folder
    path_to_output_MGS_imgs = sorted(glob(path_to_output_MGS_img + "/*.png"))
    path_to_biomarker_imgs = sorted(glob(biomarker_labeled_region + "/*.tif"))

    for ind in range(len(dna_image_path)):
        plot_nuc_mgs_biomarker(dna_image_path[ind], path_to_output_MGS_imgs[ind], path_to_biomarker_imgs[ind])
