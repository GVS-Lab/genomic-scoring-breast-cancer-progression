# -*- coding: utf-8 -*-
"""
Contains function to segment object from images given a stardist model. 

"""
import sys
import imageio as imio
from glob import glob

from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import export_imagej_rois
from tifffile import imsave
import gc

def segment_objects_stardist2d(image_dir, output_dir_labels,output_dir_ijroi,use_pretrained=False,model_dir='models', model_name='DAPI_segmenation' ):
    # download / load a pretained model
    if use_pretrained:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    else:
        model = StarDist2D(None, name=model_name, basedir=model_dir)
    # read in the images,segment and save labels
    all_images = sorted(glob(image_dir +'*.tif'))
    for i in range(len(all_images)):
            X = imio.imread(all_images[i])
            X = normalize(X,1,99.8,axis=(0,1))
            labels, polygons = model.predict_instances(X,n_tiles=model._guess_n_tiles(X),prob_thresh=0.6)
            imsave((output_dir_labels+all_images[i].rsplit('/', 1)[-1]), labels)
            if output_dir_ijroi:
                export_imagej_rois((output_dir_ijroi+all_images[i].rsplit('/', 1)[-1][:-4]+".zip"), polygons['coord'])
            del X
            del labels
            del polygons
            gc.collect()