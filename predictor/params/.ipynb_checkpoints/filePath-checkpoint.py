import os
from params.hyperparams import *

# idr/
# ROOT = os.path.realpath('..')
ROOT = os.path.realpath('..')

###
# Data
###
path_data = os.path.join(ROOT, 'data')

# 1. dataset
path_dataset = os.path.join(path_data, 'dataset')

# 2. features
path_features = os.path.join(path_data, 'features')
# path_embedded_protTrans = os.path.join(path_features, 'protTrans')
# path_embedded_onehot = os.path.join(path_features, 'onehot')
# path_embedded_hmm = os.path.join(path_features, 'hmm')

# 3. model
path_predictor = os.path.join(ROOT, 'predictor')
path_output = os.path.join(path_predictor, 'output')

plots_dir = os.path.join(path_output, f'plots/{featureType}/{model_name}/{datasetType}')
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)
    
model_pth = os.path.join(path_output, f'trained_model/{featureType}/{model_name}.pth')
auc_loss_pth = os.path.join(path_output, f'auc_loss/{featureType}/{model_name}.csv')

# log
path_log = os.path.join(path_output, f'log/{featureType}')

# 4. uniprot
path_uniprot = os.path.join(path_data, 'uniprot')