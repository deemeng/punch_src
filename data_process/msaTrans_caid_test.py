from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from json import loads, dumps
import itertools
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Bio import SeqIO

from tqdm import tqdm

# https://github.com/facebookresearch/esm
# pip install fair-esm
import esm

from utils.file import save_np, csv2df, read_json2list, dump_list2json
from utils.alignmentParser import read_msa, greedy_select
from params.PDBparser import *

torch.set_grad_enabled(False)

'''
Aim to embed the CAID pdb_disorder using MSA-Transformer. And save them to files.

Di
16 Feb, 2024
'''
# 1. mdoel & tokenizer
msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

# 2. predict & save embedded sequences
path_hmm = '/home/dimeng/project/idr/data/caid/a3m'

df_caid= pd.DataFrame(read_json2list(path_caid_dataset_json)) # entities' features
df_caid = df_caid[(df_caid['len_1022']==0)&(df_caid['length']<=1022)]

print(df_caid.shape)

for idx, row in df_caid.iterrows():
    name = row['id']
    length = row['length']
    print(name)
    # This is where the data is actually read in
    inputs = read_msa(os.path.join(path_hmm, f'{name}.a3m'))
    if len(inputs[0][1])>1022:
        print(f'Sequence length ({len(inputs[0][1])}) is greater than 1022. ')
        inputs = [(i[0], i[1][:length]) for i in inputs]
        
    inputs = greedy_select(inputs, num_seqs=128) # can change this to pass more/fewer sequences
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]
    seq_representation = token_representations[:, :, 1: ].mean(1)
    # save embedd sequences
    save_np(seq_representation, os.path.join(path_caid_features_msaTrans, f'{name}.npy'))

result = df_caid.to_json(orient="records")
list_caid = loads(result)
dump_list2json(list_caid, path_caid_dataset_json)

print('Done!!!')