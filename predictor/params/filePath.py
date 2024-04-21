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
path_dataset_test = os.path.join(path_dataset, f'{datasetType}_TestDataset.json')
path_dataset_all = os.path.join(path_dataset, 'IDRdataset.json')
# merge Disprot&IDR
path_IDR_fullyDisordered_dataset = os.path.join(path_dataset, 'IDR_fullyDisordered_dataset.json')
path_IDR_fullyDisordered_dataset_100 = os.path.join(path_dataset, 'IDR_fullyDisordered_dataset_100.json')
# 2. features
path_features = os.path.join(path_data, 'features')
path_embedded_eval = os.path.join(path_features, 'evaluation')
path_embedded_protTrans = os.path.join(path_features, 'protTrans')
# path_embedded_onehot = os.path.join(path_features, 'onehot')
# path_embedded_hmm = os.path.join(path_features, 'hmm')

# 3. model
path_predictor = os.path.join(ROOT, 'predictor_v3')
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

# 5. predictions
path_pred = os.path.join(path_output, 'pred')  
path_pred_plot = os.path.join(path_pred, 'plots')
path_pred_files = os.path.join(path_pred, 'files')

# 6. final model
if final:
    path_final = os.path.join(path_predictor, 'final_caid')
    path_plots_f = os.path.join(path_final, 'plots')
    # path_plots_model_f = os.path.join(path_plots_f, model_name)
    
    plots_dir_f = os.path.join(path_plots_f, f'{featureType}/{model_name}/{datasetType}')
    
    if not os.path.isdir(plots_dir_f):
        os.mkdir(plots_dir_f)
        
    model_pth_f = os.path.join(path_final, f'trained_model/{featureType}/{model_name}.pth')
    auc_loss_pth_f = os.path.join(path_final, f'auc_loss/{featureType}/{model_name}.csv')
    
    # log
    path_log_f = os.path.join(path_final, 'log')

    # predictions
    path_pred_f = os.path.join(path_final, 'pred')
    path_pred_plot_f = os.path.join(path_pred_f, 'plots')
    path_pred_files_f = os.path.join(path_pred_f, 'files')

# 7.CAID
path_caid = os.path.join(path_data, 'caid')

path_caid_dataset = os.path.join(path_caid, 'dataset')
path_caid_dataset_json = os.path.join(path_caid_dataset, 'caid_dataset.json')

path_caid_features = os.path.join(path_caid, 'features')
path_caid_features_protTrans = os.path.join(path_caid_features, 'protTrans')

# 8. Redundancy sequences
#  exclude similar sequences from IDR/Disprot
path_CAIDvsIDR = os.path.join(path_data, 'CAIDvsIDR')
path_CAIDvsDisprot = os.path.join(path_data, 'CAIDvsDisprot')
