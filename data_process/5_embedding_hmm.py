import pandas as pd
import numpy as np

from embedding.hmm import save_embedded
import params.PDBparser as paramsPDB
from utils.file import csv2df, read_json2list

'''
Aim to embed the IDR datasets (30% identity). And save them to files.

Di
23 Aug, 2023
'''

# 1. model
# embed and save
list_idrDataset = read_json2list(paramsPDB.path_IDRdataset)

list_seq = []
list_id = []
        
for dict_idr in list_idrDataset:
    # seq = ' '.join(dict_idr['sequence'])
    list_seq.append(seq)
    list_id.append(dict_idr['id'])

save_embedded(list_seq, list_id, paramsPDB.path_embedded_onehot)
print('Done!!!')