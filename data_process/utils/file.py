import csv
import sys
import wget
import json

import pandas as pd
import numpy as np

csv.field_size_limit(sys.maxsize)

###
# fileIO
###
def save_np(arr: list, file_path: str):
    '''
    Save a list of numpy array to a file.
    params:
        arr - a list of numpy array
        file_path - file path with .npy

    '''
    np.save(file_path, np.array(arr, dtype=np.float32), allow_pickle=True)

def load_np(file_path: str):
    '''
    Save a list of numpy array to a file.
    params:
        file_path - file path with .npy

    return:
        arr - a list of numpy array
    '''
    arr = np.load(file_path, allow_pickle=True)
    return arr

def txt2list(path_file, delimiter=','):
    '''
    params:
        path_life - txt or csv file
        delimiter - default ','
        
    return:
        list of items
    '''
    list_content = []
    
    with open(path_file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            list_content += row
    return list_content

def generate_fasta_header(list_info, delimiter=','):
    '''
    params:
        list_info - a list of features the header want to include, the minimum info is entryID_entityID
        delimiter - for separating each infomation
        
    return:
        header, e.g. >102M_1
    '''
    return '>'+delimiter.join(item for item in list_info)

def df2csv(df_data, file_path):
    df_data.to_csv(file_path, index=False, sep=',')

def csv2df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=',')
    return df
###
# PDBe
###
def download_pdbe_updatedCif(list_pdbid: list, path_pdbe_cif):
    '''
    Download updated mmcif files from PDBe, which includes SIFTS infomation (uniprot ID)
    
    params:
        list_pdbid - list of pdbIDs need to download. 
        path_pabe_cif - folder to save those mmcif files
    '''
    for pdbid in list_pdbid:
        URL = f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdbid.lower()}_updated.cif"
        response = wget.download(URL, path_pdbe_cif)
        
def dump_list2json(listData: list, path_json: str):
    '''
    Save a dictionary to a JSON file.
    Note that JSON does not recognize Array or Numpy, change it to int or list!!
    '''
    with open(path_json, 'w') as fout:
        json.dump(listData, fout)
        
def read_json2dict(path_json: str) -> dict:
    '''
    Read JSON to a dict
    '''
    with open(path_json, 'r') as f:
        dictData = json.load(f)
        
    return dictData

def read_json2list(path_json: str) -> list:
    '''
    Read JSON to a list of dicts
    '''
    with open(path_json, 'r') as f:
        listData = json.load(f)
        
    return listData
