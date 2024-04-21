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

from utils.file import save_np, csv2df, dump_dict2json
from utils.alignmentParser import read_msa
from params.PDBparser import *
from utils.sequence import get_tupleIdx

torch.set_grad_enabled(False)

'''
Aim to embed the IDR datasets (30% identity). And save them to files.
Two types of MSA embeddings.

1. Probability MSA
2. set the value of org AA to 1

Di
7 Sep, 2023
'''


# path_hmm = '/home/dimeng/project/data/pdb_entity30/a3m'
path_hmm = '/home/dimeng/project/idr/data/hmm/a3m'

df_entityFeature30 = csv2df(path_pdb_featureEntity30)

# entyID_entityID
PDB_IDS = list(set(df_entityFeature30['id'].tolist()))

# get the maximum number_of_templates, use to normalize the number_of_template
dict_idr_hmmInfo = {}
max_numTemplate = 0
for name in PDB_IDS:
    # This is where the data is actually read in
    inputs = read_msa(os.path.join(path_hmm, f'{name}.a3m'))

    num_template = len(inputs) # including query sequence
    query = inputs[0]
    seq = query[1]
    len_seq = len(seq)

    dict_idr_hmmInfo[name] = {'id': name, 'sequence': seq, 'len_seq': len_seq, 'num_template': num_template}
    max_numTemplate = max(max_numTemplate, num_template)
    
# save the alignment information to file
dump_dict2json(dict_idr_hmmInfo, path_IDR_hmmInfo)

for name in PDB_IDS:
    print(name)
    # This is where the data is actually read in
    inputs = read_msa(os.path.join(path_hmm, f'{name}.a3m'))
    ref_seq = inputs[0][1] # take the reference
    # len_seq * 22 features
    # 20 common AAs
    # 1 abnormal AAs
    # 1 gap
    # 1 count the number of templates/alignments
    seq_aaCount = np.zeros((len(ref_seq), 23), dtype='int') # count the frequence
    for _, seq in inputs:
        tuple_idx = get_tupleIdx(seq)
        seq_aaCount[tuple_idx] += 1
    seq_prop = seq_aaCount/len(inputs) # compute the probability
    seq_prop[:, -1] = len(inputs)/max_numTemplate
    # set the value of AA in orignial sequence (reference) to 1
    ref_tupleIdx = get_tupleIdx(ref_seq)
    seq_prop_ref = seq_prop.copy()
    seq_prop_ref[ref_tupleIdx] = 1

    # save embedding result
    save_np(seq_prop, os.path.join(path_embedded_hmm_prob, f'{name}.npy'))
    save_np(seq_prop_ref, os.path.join(path_embedded_hmm_ref, f'{name}.npy'))
    
print('Done!!!')