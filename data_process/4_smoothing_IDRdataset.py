import params.PDBparser as paramsPDB
from utils.file import read_json2list, dump_list2json, df2csv, csv2df
from utils.pdbParser import ref_smoothing

import pandas as pd

'''
smooth IDR annotation:
1. smoothing
2. with or without missing-residues/IDRs

25 Jul, 2023
Di.
'''
list_idrDataset = read_json2list(paramsPDB.path_IDRdataset)
df_featureEntity100 = csv2df(paramsPDB.path_pdb_featureEntity100)

list_id = []
list_contrainIDR = []
list_contrainIDR_smoothed = []


print('Smoothing ... ...')
for i in range(len(list_idrDataset)):
    dict_idr = list_idrDataset[i]
    
    smoothed_reference, contain_idr_smoothed, contain_idr = ref_smoothing(dict_idr['reference'])
    list_idrDataset[i]['reference_smoothed'] = smoothed_reference
    list_idrDataset[i]['contain_idr'] = contain_idr
    list_idrDataset[i]['contain_idr_smoothed'] = contain_idr_smoothed
    
    list_id.append(dict_idr['id'])
    list_contrainIDR.append(contain_idr)
    list_contrainIDR_smoothed.append(contain_idr_smoothed)
    
df_entity30 = pd.DataFrame({'id': list_id, 'contain_idr': list_contrainIDR, 'contain_idr_smoothed': list_contrainIDR_smoothed})
df_featureEntity30 = df_entity30.merge(df_featureEntity100, on='id', how='left')

print('Save smoothed dataset ... ...')
dump_list2json(list_idrDataset, paramsPDB.path_IDRdataset_smoothed)
print('Save pdb_featureEntity30 ... ...')
df2csv(df_featureEntity30, paramsPDB.path_pdb_featureEntity30)
print('Done!!!')