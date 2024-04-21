import pandas as pd
import numpy as np

import torch
# only the encoder part of the mdoel
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

from embedding.plm import save_embedded
import params.PDBparser as paramsPDB
from utils.file import csv2df, read_json2list

'''
Aim to embed the IDR datasets (30% identity). And save them to files.

Di
20 Feb, 2024
'''

# 1. model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
model = model.to(device)
model = model.eval()

# 2. embed and save
list_idrDataset = read_json2list(paramsPDB.path_IDRdataset_100)

list_seq = []
list_id = []

df_pdbMissing= pd.DataFrame(read_json2list(paramsPDB.path_IDRdataset_100))
df_pdbMissing['length'] = [len(seq) for seq in df_pdbMissing['sequence']]

df_pdbMissing_30= pd.DataFrame(read_json2list(paramsPDB.path_IDRdataset))
diff_ids = set(df_pdbMissing['id']) - set(df_pdbMissing_30['id'])
        
for dict_idr in list_idrDataset:
    if dict_idr['id'] in diff_ids:
        seq = ' '.join(dict_idr['sequence'])
        list_seq.append(seq)
        list_id.append(dict_idr['id'])

save_embedded(list_seq, list_id, paramsPDB.path_embedded_protTrans, model, tokenizer)
print('Done!!!')