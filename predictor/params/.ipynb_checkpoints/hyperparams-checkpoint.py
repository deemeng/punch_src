import os
from utils.static import type_dataset

datasetType = type_dataset.ALL
featureType = 'onehot'
# netName = 'model.cnn_baseline'
netName = 'model.cnn_L2C5K1'
model_name = f'{netName[6:]}_{datasetType}'

NNtype_list = ['cnn', 'rnn', 'cbrcnn']
NNtype = NNtype_list[0]

# Rnn model, True; otherwise, False.
transpose = False

# mdoel paramethers
padding = False

lr = 0.00005
train_epochs = 30
batch_size = 1
dropout = 0.0

if featureType=='plm':
    n_features = 1024
elif featureType=='onehot':
    n_features = 21
elif featureType=='hmm':
    n_features = 20

# filtering sequence length
MAX_seq_length = 400000

# load model
# if True, overwriting model_pth after training.
load_model = False