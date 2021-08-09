# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 

def summarise_feature_table(data):
    """ Function that summarises distribution characteristics for all columns in a feature table. 
    Measures computed are median, min, max, standard deviation (SD) Coefficient of Variation (CV) and Coefficient of Dispersion (CD), Inter_Quartile_Range(IQR) and Quartile Coeeffient of Dispersrion (QCD).  
        
        Args: 
            data: feature table with the columns of interest
    """
    np.seterr(all='ignore')
    median_features = pd.DataFrame(np.array(np.median(data,axis=0))).T
    median_features.columns = ['median_' + str(col) for col in data]

    min_features = pd.DataFrame(np.array(np.min(data,axis=0))).T
    min_features.columns = ['min_' + str(col) for col in data]
    max_features = pd.DataFrame(np.array(np.max(data,axis=0))).T
    max_features.columns = ['max_' + str(col) for col in data]

    SD_features = pd.DataFrame(np.array(np.std(data,axis=0))).T
    SD_features.columns = ['std_' + str(col) for col in data]
    CV_features = pd.DataFrame(np.array(np.std(data,axis=0))/np.array(np.nanmedian(data,axis=0))).T
    CV_features.columns = ['CV_' + str(col) for col in data]
    CD_features = pd.DataFrame(np.array(np.var(data,axis=0))/np.array(np.nanmedian(data,axis=0))).T
    CD_features.columns = ['CD_' + str(col) for col in data]
    IQR_features = pd.DataFrame(np.array(np.subtract(*np.nanpercentile(data, [75, 25],axis=0)))).T
    IQR_features.columns = ['IQR_' + str(col) for col in data]
    QCD_features = pd.DataFrame(np.array(np.subtract(*np.nanpercentile(data, [75,25],axis=0)))/np.array(np.add(*np.nanpercentile(data, [75, 25],axis=0)))).T
    QCD_features.columns = ['QCD_' + str(col) for col in data]


    all_features = pd.concat([median_features.reset_index(drop=True),
                              min_features.reset_index(drop=True),
                              max_features.reset_index(drop=True),
                              SD_features.reset_index(drop=True),
                              CV_features.reset_index(drop=True),
                              CD_features.reset_index(drop=True),
                              IQR_features.reset_index(drop=True),
                              QCD_features.reset_index(drop=True)], axis=1)
    return all_features
