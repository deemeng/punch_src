import os
from utils.static import type_dataset

datasetType = type_dataset.ALL

# netName = 'model.cnn_baseline'
netName = 'model.cnn2_L4'
model_name = f'{netName[6:]}_{datasetType}'

NNtype_list = ['cnn', 'rnn', 'cbrcnn']
NNtype = NNtype_list[0]

# Rnn model, True; otherwise, False.
transpose = False

# mdoel paramethers
padding = False

lr = 0.00005
train_epochs = 50
batch_size = 1
dropout = 0.0

n_features = 1024
# filtering sequence length
MAX_seq_length = 400000

# load model
# if True, overwriting model_pth after training.
load_model = False