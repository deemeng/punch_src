import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import params.PDBparser as paramsPDB
from utils.file import read_json2list, dump_list2json, csv2df
from utils.static import bcolors, type_dataset



def get_entityList(datasetType: str, df_featureEntity:pd.DataFrame) -> list:
    
    list_entityID = []
    
    if datasetType == type_dataset.ALL or datasetType == type_dataset.SMOOTH_ALL:
        list_entityID = list(df_featureEntity['id'].unique())
    elif datasetType == type_dataset.IDRONLY:
        list_entityID = list(df_featureEntity[df_featureEntity['contain_idr']==1]['id'].unique())
    elif datasetType == type_dataset.SMOOTH_IDRONLY:
        list_entityID = list(df_featureEntity[df_featureEntity['contain_idr_smoothed']==1]['id'].unique())
        
    return list_entityID

def split_TrainTest_chain(df_featureEntity, list_entityDataset, path_save, datasetType=type_dataset.ALL, test_percentage=0.33, random_state=42):
    '''
    Given the list of entity IDs, split them into Train & Test sets. Save train & test to JSON files. 
    
    params:
        df_featureEntity - feature table of all entities.
        list_entityDataset - list of dict. [{'id':, 'sequence':, 'reference':, 'reference_smoothed', 'contain_idr':, 'contain_idr_smoothed'}, {}]
        path_save - dataset folder path
        datasetType - str, choose from class utils.static.type_dataset.
                        all - no smoothing, all entities
                        idrOnly - no smoothing, only entities contain IDRs
                        smooth_all - smoothing, all entities
                        smooth_idrOnly - smoothing, only entities contain IDRs
    '''
    
    try:
        print(f'dataset type: {type_dataset.get_info(datasetType)}')
    except:
        print(f'Please choose make sure you choose type_dataset from utils.static.type_dataset. {bcolors.WARNING}\n')
    
    list_entityID = get_entityList(datasetType, df_featureEntity)
        
    # get test & train IDs
    list_trainID, list_testID = train_test_split(list_entityID, test_size=test_percentage, random_state=random_state)
    print(f'Size of [dataset type]: {len(list_entityID)}')
    print(f'Size of training: {len(list_trainID)}\nSize of test: {len(list_testID)}')
    
    # get list of dicts for train & test
    list_train = [dict_entity for dict_entity in list_entityDataset if dict_entity['id'] in list_trainID]
    list_test = [dict_entity for dict_entity in list_entityDataset if dict_entity['id'] in list_testID]
    
    # path
    path_trainDataset = os.path.join(path_save, f'{datasetType}_TrainDataset.json')
    path_testDataset = os.path.join(path_save, f'{datasetType}_TestDataset.json')
    
    # save
    print('Saving .. ..')
    dump_list2json(list_train, path_trainDataset)
    dump_list2json(list_test, path_testDataset)
    
    print('Done!!!')
    
if __name__ == '__main__':
    # split train & test datasets
    
    df_featureEntity30 = csv2df(paramsPDB.path_pdb_featureEntity30)
    list_entityDataset = read_json2list(paramsPDB.path_IDRdataset_smoothed)
    path_save = paramsPDB.path_dataset
    split_TrainTest_chain(df_featureEntity30, list_entityDataset, path_save, type_dataset.ALL)
    split_TrainTest_chain(df_featureEntity30, list_entityDataset, path_save, type_dataset.SMOOTH_ALL)
    split_TrainTest_chain(df_featureEntity30, list_entityDataset, path_save, type_dataset.IDRONLY)
    split_TrainTest_chain(df_featureEntity30, list_entityDataset, path_save, type_dataset.SMOOTH_IDRONLY)