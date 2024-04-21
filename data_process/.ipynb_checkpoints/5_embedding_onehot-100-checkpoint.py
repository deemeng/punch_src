import pandas as pd
import numpy as np

from embedding.onehot import save_embedded
import params.PDBparser as paramsPDB
from utils.file import csv2df, read_json2list

'''
Aim to embed the IDR datasets (30% identity). And save them to files.

Di
23 Aug, 2023
'''

# 1. model
# embed and save
list_idrDataset = read_json2list(paramsPDB.path_IDRdataset_100)

list_seq = []
list_id = []

df_pdbMissing= pd.DataFrame(read_json2list(paramsPDB.path_IDRdataset_100))
df_pdbMissing['length'] = [len(seq) for seq in df_pdbMissing['sequence']]

df_pdbMissing_30= pd.DataFrame(read_json2list(paramsPDB.path_IDRdataset))
diff_ids = set(df_pdbMissing['id']) - set(df_pdbMissing_30['id'])

for dict_idr in list_idrDataset:
    if dict_idr['id'] in diff_ids:
        # seq = ' '.join(dict_idr['sequence'])
        list_seq.append(dict_idr['sequence'])
        list_id.append(dict_idr['id'])

save_embedded(list_seq, list_id, paramsPDB.path_embedded_onehot)
print('Done!!!')