import os
import pandas as pd
from utils.static import type_dataset

import params.filePath as paramF

def getPath(datasetType):
    '''
    Given the datasetType, return the proper train/test file path.
    '''
    path_train = os.path.join(paramF.path_dataset, f'{datasetType}_TrainDataset.json')
    path_test = os.path.join(paramF.path_dataset, f'{datasetType}_TestDataset.json')
    return path_train, path_test

def selectCol(df_train, df_test, datasetType):
    '''
    Given the dataset type, select the corresponded columns.
    params:
        df_train/df_test - dataframe
        datasetType - select type from utils.static.type_dataset
    
    return:
        df_train/df_test - DataFrame
    
    type_dataset:
        ALL = 'all'
        IDRONLY = 'idrOnly'
        SMOOTH_ALL = 'smooth_all'
        SMOOTH_IDRONLY = 'smooth_idrOnly'
    '''
    
    list_col = []
    if datasetType==type_dataset.ALL or datasetType==type_dataset.IDRONLY or type_dataset.ALL_FD:
        list_col = ['id', 'sequence', 'reference']
    elif datasetType==type_dataset.SMOOTH_ALL or datasetType==type_dataset.SMOOTH_IDRONLY:
        list_col = ['id', 'sequence', 'reference_smoothed']
        
    df_train = df_train[list_col]
    df_test = df_test[list_col]
    
    return df_train, df_test