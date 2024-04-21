import os
from utils.static import type_dataset
from params.utils import get_numFeature

# ALL: all pdb sequences
# ALL_FD: all pdb sequences + Disprot fully disordered
datasetType = type_dataset.ALL_FD_78

# exclude similar sequences with CIAD from IDR, 30% identity
exclude_caid = True

# esm&msa_transformer: https://github.com/facebookresearch/esm
dict_featureType = {1: 'onehot', 2: 'hmm_prob_numTemp', 3: 'hmm_aa_numTemp', 4: 'protTrans', 5: 'esm2', 
                    6: 'msa_transformer', 7: 'hmm_prob', 8: 'hmm_aa', 9: 'hmm_prob@onehot', 10: 'hmm_prob_numTemp@onehot',
                    11: 'protTrans@onehot', 12: 'esm2@onehot', 13: 'msa_transformer@onehot', 14:'msa_transformer@hmm_prob_numTemp', 
                    15:'protTrans@hmm_prob_numTemp', 16:'protTrans@msa_transformer' 
                   }
featureType = dict_featureType[4]
# netName = 'model.cnn2_L11'
netName = 'model.cnn2_L3_100_50'
# netName = 'model.rnn_baseline_bi_test'
model_name = f'{netName[6:]}_{datasetType}'

# lstm belongs to RNN category. 
NNtype_list = ['cnn', 'rnn', 'cbrcnn']
net_type = NNtype_list[0]

# Rnn model, True; otherwise, False.
# transpose = False

# mdoel paramethers
padding = True

lr = 0.0001
train_epochs = 10
batch_size = 50
dropout = 0.0

# filtering sequence length
n_features, MAX_seq_length = get_numFeature(featureType)

# load model
# if True, overwriting model_pth after training.
load_model = False

final = False

k_folds = 5
duplicate_disprot_FD = 3
