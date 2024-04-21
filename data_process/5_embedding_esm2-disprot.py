import itertools
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO

from tqdm import tqdm
import pandas as pd

import esm

from utils.file import save_np, read_json2list
from params.PDBparser import *

torch.set_grad_enabled(False)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

list_entity = read_json2list(path_disprot_dataset_fullDisordered)

for entity in list_entity:
    print(entity['id'])
    data = [(entity['id'], entity['sequence'])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    embedded_seq = token_representations[0, 1 : - 1].numpy()
    # save the embedding
    save_np(embedded_seq, os.path.join(path_embedded_esm2, f"{entity['id']}.npy"))

print('Done!!!!')
