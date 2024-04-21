import numpy as np
import pandas as pd
import torch

import re

from utils.common import load_np

class PadRightTo(object):
    """Pad the tensor to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        padding = self.output_size - sample.size()[-1]
        return torch.nn.functional.pad(sample, (0, padding), 'constant', 0)

def read_plm(plm_path, start_token=False, end_token=False):
    plm = load_np(plm_path)
    # plm = plm.dropna().astype(np.float32)
    plm = torch.tensor(plm, dtype=torch.float32)
    if start_token:
        plm = plm[1:]
    if end_token:
        plm = plm[:-1]
    return plm

def parse_target(target):
    '''
    params:
        target - str, e.g. '1000000110000'
    '''
    new_target = torch.tensor([int(t) for t in target], dtype=torch.uint8)
    return new_target

def sequence_mapping(list_seq: list) -> list:
    '''
    Given a list of sequences, map rarely Amino Acids [U Z O B] to [X].
    
    params:
        list_seq - list of sequences, e.g. ['A E T C Z A O', 'S K T Z P']
        
    return:
        the list of sequences with rarely AAs mapped to X.
    '''
    return [re.sub(f'[UZOB]', 'X', sequence) for sequence in list_seq]
