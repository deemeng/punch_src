from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from Bio import SeqIO

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

# This is an efficient way to delete lowercase characters and insertion characters from a string
'''
https://github.com/soedinglab/hh-suite/wiki#the-same-alignment-in-a3m

a2m vs a3m format

1. a2m
lowercase letters: insertions
-: gap
.: gap aligned insertion

2. a3m
lowercase letters: insertions
-: gap
'''
deletekeys = dict.fromkeys(string.ascii_lowercase) # given keys and values are None.
deletekeys["."] = None # gap aligned insertion (>1st time of searching...)
deletekeys["*"] = None # not sure this one
translation = str.maketrans(deletekeys) # create a mapping table

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ 
    Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. 
    Make sure all the template sequences have the same length of the reference.
    """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter 
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]