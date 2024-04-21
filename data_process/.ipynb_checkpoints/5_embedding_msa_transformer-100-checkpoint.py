from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Bio import SeqIO

from tqdm import tqdm

import esm

from utils.file import save_np, csv2df, read_json2list
from utils.alignmentParser import read_msa, greedy_select
from params.PDBparser import *

torch.set_grad_enabled(False)

'''
Aim to embed the IDR datasets (30% identity) using MSA-Transformer. And save them to files.

Di
06 Sep, 2023
'''

# 1. mdoel & tokenizer
msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

# 2. predict & save embedded sequences
path_hmm = '/home/dimeng/project/idr/data/a3m'

df_entityFeature100 = pd.DataFrame(read_json2list(path_IDR_fullyDisordered_dataset_100))
df_entityFeature100['length'] = [len(seq) for seq in df_entityFeature100['sequence']]
df_entityFeature100 = df_entityFeature100[df_entityFeature100['length']<=1022]

PDB_IDS = list(set(df_entityFeature100['id'].tolist()))
'''
START
Only for if some proteins are missing here.
'''
embedd_PDB_IDS = [fname[:-4] for fname in os.listdir(path_embedded_msaTrans)]
unembedd_PDB_IDS = list(set(PDB_IDS) - set(embedd_PDB_IDS))
PDB_IDS = unembedd_PDB_IDS
print(f'The number of unembedded: {len(PDB_IDS)}')
'''
END
'''

for name in PDB_IDS:
    print(name)
    # This is where the data is actually read in
    inputs = read_msa(os.path.join(path_hmm, f'{name}.a3m'))
    
    inputs = greedy_select(inputs, num_seqs=128) # can change this to pass more/fewer sequences
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]
    seq_representation = token_representations[:, :, 1: ].mean(1)
    # save embedd sequences
    save_np(seq_representation, os.path.join(path_embedded_msaTrans, f'{name}.npy'))

print('Done!!!')