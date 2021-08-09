# -*- coding: utf-8 -*-
# edited from https://github.com/carpenterlab/unet4nuclei/blob/master/unet4nuclei/utils/evaluation.py and
# stardist's matching.py

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

def intersection_over_union(ground_truth, prediction):
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU

def metrics(IOU_matrix, threshold):
    
    n_true, n_pred = IOU_matrix.shape
    n_matched = min(n_true, n_pred)
    
    if IOU_matrix.shape[0] > 0:
        jaccard = np.max(IOU_matrix, axis=0).mean()
    else:
        jaccard = 0.0
    
    
    # compute optimal matching with scores as tie-breaker
    costs = -(IOU_matrix >= threshold).astype(float) - IOU_matrix / (2*n_matched)
    true_ind, pred_ind = linear_sum_assignment(-IOU_matrix)
    assert n_matched == len(true_ind) == len(pred_ind)
    matches = IOU_matrix[true_ind,pred_ind] >= threshold

    true_positives =  np.count_nonzero(matches)  # Correct objects
    false_positives = n_pred - true_positives  # Extra objects
    false_negatives = n_true - true_positives  # Missed objects
  
    #Precision
    precision = true_positives/(true_positives + false_positives + 1e-9) if true_positives > 0 else 0
    
    #Recall
    recall = true_positives/(true_positives + false_negatives + 1e-9) if true_positives > 0 else 0
    
    #Accuracy also known as "average precision" 
    Accuracy = true_positives/(true_positives + false_positives + false_negatives + 1e-9) if true_positives > 0 else 0
    
    #F1 also known as "dice coefficient"
    f1 = (2*true_positives)/(2*true_positives + false_positives + false_negatives + 1e-9) if true_positives > 0 else 0

    # obtain the sum of iou values for all matched objects
    sum_matched_score = np.sum(IOU_matrix[true_ind,pred_ind][matches])
    
    # the score average over all matched objects (tp)
    mean_matched_score = sum_matched_score / (true_positives + 1e-9)
    
    # the score average over all gt/true objects
    mean_true_score    = sum_matched_score / (n_true + 1e-9)
    
    #panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019
    panoptic_quality   = sum_matched_score / ( true_positives + false_positives/2 + false_negatives/2 + 1e-9)

    res = pd.DataFrame({"Threshold": threshold, 
                           "Jaccard": jaccard, 
                           "TP": true_positives, 
                           "FP": false_positives, 
                           "FN": false_negatives, 
                           "Precision": precision, 
                           "Recall": recall,
                           "Accuracy": Accuracy,
                           "F1": f1,
                           "sum_matched_score":sum_matched_score,
                           "mean_matched_score":mean_matched_score,
                           "mean_true_score":mean_true_score,
                           "panoptic_quality":panoptic_quality}, index=[0])
    del jaccard, true_positives, false_positives, false_negatives, precision, recall, Accuracy, f1, mean_matched_score, mean_true_score, panoptic_quality
    
    return res


def evaluate_segementation_per_image(ground_truth, prediction, thresholds_list, identifier):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    # Compute metrics accross all thresholds
    df = pd.DataFrame()
    for t in thresholds_list:
        df = pd.concat([df, metrics(IOU, t)], ignore_index=True, axis=0)
    
    df['Image_ID']=identifier
    return df

def evaluate_segementation_whole_dataset(ground_truth_list, prediction_list, thresholds_list):
     # Compute metrics accross all thresholds
    
    results = pd.DataFrame()
    for t in thresholds_list:
        df = pd.DataFrame()
        jaccard = 0
        for f in range(len(ground_truth_list)):
            # Compute IoU
            IOU = intersection_over_union(ground_truth_list[f], prediction_list[f])
            df = pd.concat([df, metrics(IOU, t)], ignore_index=True, axis=0)
            jaccard = max(jaccard, np.max(IOU, axis=0).mean())

        true_positives =  np.sum(df['TP'])  # Correct objects
        false_positives = np.sum(df['FP']) # Extra objects
        false_negatives = np.sum(df['FN'])  # Missed objects
        sum_matched_score = np.sum(df['sum_matched_score']) # sum matched score
        #Precision
        precision = true_positives/(true_positives + false_positives + 1e-9) if true_positives > 0 else 0

        #Recall
        recall = true_positives/(true_positives + false_negatives + 1e-9) if true_positives > 0 else 0

        #Accuracy also known as "average precision" 
        Accuracy = true_positives/(true_positives + false_positives + false_negatives + 1e-9) if true_positives > 0 else 0

        #F1 also known as "dice coefficient"
        f1 = (2*true_positives)/(2*true_positives + false_positives + false_negatives + 1e-9) if true_positives > 0 else 0

        # the score average over all matched objects (tp)
        mean_matched_score = sum_matched_score / (true_positives + 1e-9)

        # the score average over all gt/true objects
        mean_true_score    = sum_matched_score / (true_positives + false_negatives+ 1e-9)

        #panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019
        panoptic_quality   = sum_matched_score / ( true_positives + false_positives/2 + false_negatives/2 + 1e-9)

        res = pd.DataFrame({"Threshold": t, 
                           "Jaccard": jaccard, 
                           "TP": true_positives, 
                           "FP": false_positives, 
                           "FN": false_negatives, 
                           "Precision": precision, 
                           "Recall": recall,
                           "Accuracy": Accuracy,
                           "F1": f1,
                           "mean_matched_score":mean_matched_score,
                           "mean_true_score":mean_true_score,
                           "panoptic_quality":panoptic_quality}, index=[0])
        results = pd.concat([results, res], ignore_index=True, axis=0)

    return results