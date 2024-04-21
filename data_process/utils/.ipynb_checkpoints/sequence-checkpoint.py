import re
import numpy as np

# The mapping here uses hhblits convention, so that B is mapped to D, J and O
# are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
# remaining 20 amino acids are kept in alphabetical order.
# There are 2 non-amino acid codes, X (representing any amino acid) and
# "-" representing a missing amino acid in an alignment.  The id for these
# codes is put at the end (20 and 21) so that they can easily be ignored if
# desired.
HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

def sequence_mapping_list(list_seq: list) -> list: # !!!ProTrans mapping, used to be sequence_mapping
    '''
    Given a list of sequences, map rarely Amino Acids [U Z O B] to [X].
    
    params:
        list_seq - list of sequences, e.g. ['A E T C Z A O', 'S K T Z P']
        
    return:
        the list of sequences with rarely AAs mapped to X.
    '''
    return [re.sub(f'[UZOB]', 'X', sequence) for sequence in list_seq]

def sequence_mapping_single(seq: str) -> str: # !!!ProTrans mapping
    '''
    Given a list of sequences, map rarely Amino Acids [U Z O B] to [X].
    
    params:
        list_seq - list of sequences, e.g. ['A E T C Z A O', 'S K T Z P']
        
    return:
        the list of sequences with rarely AAs mapped to X.
    '''
    return re.sub(f'[UZOB]', 'X', seq)

def sequence_mapping_hhblits(seq: str) -> str: # HHblits mapping
    '''
    Given a list of sequences, map rarely Amino Acids [B J O U Z] to:
    B -> D
    J and O -> X
    U -> C
    Z -> E
    are mapped to X, U is mapped to C, and Z is mapped to E. 
    
    params:
        seq - a single sequence, e.g. 'AETCZAO'
        
    return:
        the seq with rarely AAs mapped to X.
    '''
    seq = re.sub(f'[B]', 'D', seq)
    seq = re.sub(f'[JO]', 'X', seq)
    seq = re.sub(f'[U]', 'C', seq)
    seq = re.sub(f'[Z]', 'E', seq)
    return seq

def get_tupleIdx(seq):
    '''
    Given protein sequences, generate the tuple indices for the sequece.
    params:
    seq - str, protein sequence
    
    return:
    tuple_idx - tuple indices for each AA in the sequence. ([position_idx], [AA_idx])
    e.g.:
    sequence_length = 126
    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
         39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
         52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
         65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
         91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        117, 118, 119, 120, 121, 122, 123, 124, 125]),
     array([13, 12, 17,  9,  6, 13, 12, 12,  0, 10, 15, 15,  0,  9,  5, 16, 16,
         7, 14,  9, 16,  1, 16,  9, 14, 11,  2,  6,  2,  7,  5, 17, 19, 15,
        17, 19, 18, 19, 13, 13, 14, 12,  5,  6, 12, 12, 14,  4,  9,  9, 14,
        19,  4, 15, 13, 15,  2,  8, 15, 13,  5, 12, 13, 17, 12, 12, 14,  4,
        15,  5, 15,  8,  2, 17,  0, 14, 11, 14,  5, 19,  9, 15,  7, 15,  3,
         9, 13, 12,  3,  2,  3,  0, 10, 19, 19,  1,  0, 10,  5,  0, 14, 15,
        15,  3,  8,  3,  3, 14,  3, 14,  3, 18,  3,  3,  3, 10,  3, 12, 16,
         0,  0, 14, 16, 14, 17, 12])
    '''
    seq_indices = [HHBLITS_AA_TO_ID[aa] for aa in seq]
    tuple_idx = [(idx, aa_idx) for idx, aa_idx in enumerate(seq_indices)]
    tuple_idx = tuple(np.transpose(tuple_idx))
    return tuple_idx