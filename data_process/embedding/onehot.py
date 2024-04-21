import torch
import os
import numpy as np
from utils.sequence import sequence_mapping_list
from utils.file import save_np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def onehot_encoding(seq: str) -> np.array:
    '''
    param:
        seq - str, input protein sequence
    return:
        encoding matrix - np.array, size: len_seq*21
    '''
    
    # 20 Amino Acids and 1 others
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    # Define the mapping of amino acids to indices, mapping abnormal amino acids to #
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    # Initialize an empty matrix for one-hot encoding
    num_amino_acids = len(amino_acids)
    encoding_matrix = np.zeros((len(seq), num_amino_acids))
    
    # Fill the matrix with one-hot encoded values
    for i, aa in enumerate(seq):
        if aa in aa_to_index: # 20 usual AAs
            index = aa_to_index[aa]
            encoding_matrix[i][index] = 1
        # else: # abnormal AAs
            # encoding_matrix[i][num_amino_acids-1] = 1    
    return encoding_matrix
    
def get_onehot_embedding(list_seq: list) -> list[np.array]:
    '''
    Given a list of sequences, generate a list of onehot embedding of these sequences. 
    
    params:
        list_seq - list of Uniprot sequences.
        
    return:
        embedded_seq - list of embedded sequences (numpy arraies). Unequal lengths.
    '''
    # 1. map rarely Amino Acids to X
    list_seq = sequence_mapping_list(list_seq)
    
    # 2. get all onehot embedding for the list of sequences
    embedded_seq = [onehot_encoding(seq) for seq in list_seq]
    
    return embedded_seq

def save_embedded(list_seq_unp: list, list_seqID: list, file_folder: str):
    '''
    Embed a list of sequences and save them to files.
    
    params:
        list_seq_unp - list of uniprot sequences.
        list_seqID - file names
        file_folder - the folder to save the sequence
        mdoel - PLM
        tokenizer
    '''
    
    for idx in range(len(list_seq_unp)):
        print(list_seqID[idx])
        seq_embedded = get_onehot_embedding([list_seq_unp[idx]])
        file_path = os.path.join(file_folder, f'{list_seqID[idx]}.npy')
        save_np(seq_embedded, file_path)
        
