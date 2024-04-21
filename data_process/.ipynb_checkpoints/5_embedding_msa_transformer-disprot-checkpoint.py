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
Aim to embed the Fully Disordered sequences in Disprot using MSA-Transformer. And save them to files.

Di
16 Feb, 2024
'''
# 1. mdoel & tokenizer
msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

# 2. predict & save embedded sequences
path_hmm = '/home/dimeng/project/idr/data/disprot/a3m'

df_disprot_fd= pd.DataFrame(read_json2list(path_disprot_dataset_fullDisordered)) # entities' features

df_disprot_fd = df_disprot_fd[df_disprot_fd['length']<=1022]
# entyID_entityID
seq_IDS = list(set(df_disprot_fd['id'].tolist()))

'''
START
Only for if some proteins missed embedded here.
'''
'''
embedd_seq_IDS = [fname[:-4] for fname in os.listdir(path_embedded_msaTrans)]
unembedd_seq_IDS = list(set(seq_IDS) - set(embedd_seq_IDS))
seq_IDS = unembedd_seq_IDS
print(f'The number of unembedded: {len(seq_IDS)}')
'''
'''
END
'''
for name in seq_IDS:
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