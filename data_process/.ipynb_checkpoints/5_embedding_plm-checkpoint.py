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
25 Jul, 2023
'''

# 1. model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
model = model.to(device)
model = model.eval()

# 2. embed and save
list_idrDataset = read_json2list(paramsPDB.path_IDRdataset)

list_seq = []
list_id = []
        
for dict_idr in list_idrDataset:
    seq = ' '.join(dict_idr['sequence'])
    list_seq.append(seq)
    list_id.append(dict_idr['id'])

save_embedded(list_seq, list_id, paramsPDB.path_embedded_protTrans, model, tokenizer)
print('Done!!!')