import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.utils import parse_target, read_plm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class Sequence:
    def __init__(self, seq_id, sequence, target, feature_path=None, data_transform=None,
                 target_transform=None, feature_type='protTrans'):
        self.seq_id = seq_id
        self.sequence = sequence
        self._target = parse_target(target)
            
        self.data_transform = data_transform
        self.target_transform = target_transform
        if feature_type=='protTrans':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'protTrans/{}.npy'.format(self.seq_id)))
        elif feature_type=='onehot':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
        elif feature_type=='hmm_prob':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'hmm/prob/{}.npy'.format(self.seq_id)))
        elif feature_type=='hmm_aa':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'hmm/aa/{}.npy'.format(self.seq_id)))
        elif feature_type=='hmm_prob_numTemp':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'hmm/prob_NumTemplate/{}.npy'.format(self.seq_id)))
        elif feature_type=='hmm_aa_numTemp':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'hmm/aa_NumTemplate/{}.npy'.format(self.seq_id)))
        elif feature_type=='esm2':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'esm2/{}.npy'.format(self.seq_id)), start_token=True, end_token=True)
        elif feature_type=='msa_transformer':
            self.seq_encoding = read_plm(os.path.join(feature_path, 'msaTrans/{}.npy'.format(self.seq_id)))
        elif feature_type=='hmm_prob@onehot':
            hmm_prob_encoding = read_plm(os.path.join(feature_path, 'hmm/prob/{}.npy'.format(self.seq_id)))
            onehot_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((hmm_prob_encoding.squeeze(0), onehot_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='hmm_prob_numTemp@onehot':
            hmm_encoding = read_plm(os.path.join(feature_path, 'hmm/prob_NumTemplate/{}.npy'.format(self.seq_id)))
            onehot_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((hmm_encoding.squeeze(0), onehot_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='protTrans@onehot':
            plm_encoding = read_plm(os.path.join(feature_path, 'protTrans/{}.npy'.format(self.seq_id)))
            onehot_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm_encoding.squeeze(0), onehot_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='esm2@onehot':
            plm_encoding = read_plm(os.path.join(feature_path, 'esm2/{}.npy'.format(self.seq_id)), start_token=True, end_token=True)
            onehot_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm_encoding.squeeze(0), onehot_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='msa_transformer@onehot':
            plm_encoding = read_plm(os.path.join(feature_path, 'msaTrans/{}.npy'.format(self.seq_id)))
            onehot_encoding = read_plm(os.path.join(feature_path, 'onehot/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm_encoding.squeeze(0), onehot_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='msa_transformer@hmm_prob_numTemp':
            plm_encoding = read_plm(os.path.join(feature_path, 'msaTrans/{}.npy'.format(self.seq_id)))
            hmm_encoding = read_plm(os.path.join(feature_path, 'hmm/prob_NumTemplate/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm_encoding.squeeze(0), hmm_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='protTrans@hmm_prob_numTemp':
            plm_encoding = read_plm(os.path.join(feature_path, 'protTrans/{}.npy'.format(self.seq_id)))
            hmm_encoding = read_plm(os.path.join(feature_path, 'hmm/prob_NumTemplate/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm_encoding.squeeze(0), hmm_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)
        elif feature_type=='protTrans@msa_transformer':
            plm1_encoding = read_plm(os.path.join(feature_path, 'msaTrans/{}.npy'.format(self.seq_id)))
            plm2_encoding = read_plm(os.path.join(feature_path, 'protTrans/{}.npy'.format(self.seq_id)))
            self.seq_encoding = torch.cat((plm1_encoding.squeeze(0), plm2_encoding.squeeze(0)), dim=1)
            self.seq_encoding = self.seq_encoding.unsqueeze(0)

    @property
    def data(self):
        data = self.seq_encoding.mT.squeeze(0)
        if self.data_transform is not None:
            data = self.data_transform(data)
        return data.float()

    @property
    def target(self):
        target = self._target
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target.float()

    @property
    def clean_target(self):
        return self._target.numpy()

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return 'Sequence({}, {})'.format(self.seq_id, self.sequence)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, i):
        return self.data, self.target

    def as_dict(self):
        return {"seq_id": self.seq_id, "sequence": self.sequence, "target": self.target, "data": self.data}


# Base class for the two datasets, with common functionality
class IDRDataset(Dataset):
    def __init__(self, data, feature_root, transform=None, target_transform=None, feature_type='plm'):
        self.transform = transform
        self.target_transform = target_transform
        self.raw_data = data
        self.feature_root = feature_root
        self.feature_type = feature_type

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        seq_id, sequence, target, _ = self.raw_data.iloc[idx]
        item = Sequence(seq_id, sequence, target, feature_path=self.feature_root, feature_type=self.feature_type,
                                      data_transform=self.transform, target_transform=self.target_transform)
        return item

def pad_packed_collate(batch: List[Sequence]):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(sequence, target)].
             sequence is a FloatTensor
             target has the same variable length with sequence
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        seqs, labels = [batch[0].data], [batch[0].target]
        lengths = [seqs[0].size(0)]
        
    if len(batch) > 1:
        # get data and sorted by the length of sequence
        seqs, labels, lengths = zip(*[(item.data.T, item.target, item.data.size(1)) for item in sorted(batch, key=lambda x: x.data.size(1), reverse=True)])
    seqs = pad_sequence(seqs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    packed_seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
    packed_labels = pack_padded_sequence(labels, lengths, batch_first=True)
    
    return batch, packed_seqs, packed_labels
    
def collate_fn(batch: List[Sequence]):
    data = torch.stack([item.data for item in batch])
    target = torch.stack([item.target for item in batch])
    return batch, data, target

