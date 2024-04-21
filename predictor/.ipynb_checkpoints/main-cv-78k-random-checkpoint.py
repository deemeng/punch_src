import os
from typing import List
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from dataset.idr_dataset import IDRDataset, Sequence, collate_fn, pad_packed_collate
from utils.main_support import selectCol, getPath
from dataset.utils import PadRightTo
from utils.static import type_dataset

import params.filePath as paramF
import params.hyperparams as paramH

from model.utils import plot_auc_and_loss, plot_roc_curve, save_checkpoint, load_checkpoint, count_modelParams
from model.train import train, test, predict_one_sequence

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - here is more specific
writer = SummaryWriter(paramF.plots_dir, filename_suffix='board')

# import model
import importlib
model = importlib.import_module(paramH.netName)

import logging
logging.root.setLevel(logging.INFO)
# logging.basicConfig(level=logging.NOTSET)
logging.basicConfig(filename=os.path.join(paramF.path_log, f'{paramH.model_name}.log'), 
                    filemode='a',
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %P',
                    level=logging.INFO)

if __name__ == '__main__':
    # Performance tuning
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    ######

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    '''
    1. data
    '''
    # get file path
    if paramH.datasetType==type_dataset.ALL_FD_78:
        path_dataset = paramF.path_IDR_fullyDisordered_dataset_100

    else:
        sys.exit(f"wrong data type: {paramH.datasetType}\nshould be: type_dataset.ALL_FD_78") 
        
    # Load the data
    df_dataset = pd.read_json(path_dataset, orient='records', dtype=False)

    if paramH.exclude_caid:
        df_CAIDvsIDR = pd.read_csv(os.path.join(paramF.path_CAIDvsIDR, 'alnRes30.m8'), sep='\t', header=None)
        df_CAIDvsDisprot = pd.read_csv(os.path.join(paramF.path_CAIDvsDisprot, 'alnRes30.m8'), sep='\t', header=None)
        list_caid =  list(df_CAIDvsIDR[1]) + list(df_CAIDvsDisprot[1])
        df_dataset = df_dataset.loc[~df_dataset['clstr_id'].isin(list_caid), :]
    
    # copy the Disprot_FD by duplicate_disprot_FD times.    
    if paramH.duplicate_disprot_FD > 1:
        df_IDR_fullyDisordered = df_dataset.loc[df_dataset['clstr_id']=='disprot', :]
        df_IDR_FD_multi = pd.DataFrame(np.repeat(df_IDR_fullyDisordered.values, paramH.duplicate_disprot_FD-1, axis=0), columns=df_IDR_fullyDisordered.columns)
        df_dataset = pd.concat([df_dataset, df_IDR_FD_multi], ignore_index=True)
    
    # select columns
    df_dataset = df_dataset.loc[:, ['id', 'sequence', 'reference']]
    
    # Filter protein length less than 
    df_dataset['p_len'] = df_dataset['sequence'].map(lambda x: len(x))
    df_dataset = df_dataset[df_dataset['p_len']<=paramH.MAX_seq_length]

    MAX_length = max(df_dataset['p_len'])

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=paramH.k_folds, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(df_dataset)):
        # start from 1
        fold = fold + 1
        
        # train & test
        df_train = df_dataset.iloc[train_ids, :]
        df_test = df_dataset.iloc[test_ids, :]
        
        # IDRDataset
        train_dataset = IDRDataset(data=df_train, feature_root=paramF.path_features, feature_type=paramH.featureType,
                               transform=PadRightTo(MAX_length), target_transform=PadRightTo(MAX_length))
        test_dataset = IDRDataset(data=df_test, feature_root=paramF.path_features, feature_type=paramH.featureType,
                               transform=PadRightTo(MAX_length), target_transform=PadRightTo(MAX_length))
        
        # Defining the dataloader for the training set and the test set
        train_loader = DataLoader(train_dataset, batch_size=paramH.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=paramH.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
        
        # Best epoch
        max_auc = -1
        max_auc_loss = -1
        best_epoch = -1

        '''
        2. model
        '''
        # Instantiate the model
        if paramH.net_type=='cnn':
            net = model.Net(in_features=paramH.n_features, dropout=paramH.dropout).to(device)
        elif paramH.net_type=='rnn': # lstm is a kind of rnn.
            net = model.Net(paramH.n_features, hidden_size=paramH.rnn_feature.hidden_size, num_layers=paramH.rnn_feature.num_layers, bidirectional=paramH.rnn_feature.bidirectional).to(device)
        num_params = count_modelParams(net)
        print(f'Model name: {paramH.model_name}\nNumber of parameters: {num_params}')
    
        # save all losses and aucs to arraries.
        all_train_loss, all_test_loss, all_test_aucs = np.array([]), np.array([]), np.array([])
        
        # Define the loss function and the optimizer
        criterion = nn.MSELoss(reduction='mean')
    
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=paramH.lr)
    
        start_epoch = 0
    
        if paramH.load_model:
            net, _, start_epoch, _ = load_checkpoint(net, optimizer, f'{paramF.model_pth}_f{fold}')
            
            df_auc_loss = pd.read_csv(f'{paramF.auc_loss_pth}_f{fold}')
            all_train_loss = np.array(df_auc_loss['train_loss'])
            all_test_loss = np.array(df_auc_loss['test_loss'])
            all_test_aucs = np.array(df_auc_loss['test_auc'])
        '''
        # add model graph to tensorboard.
        for batch_idx, (sequences, data, target) in enumerate(test_loader):
            data, target = data[0].to(device), target[0].to(device)
            writer.add_graph(net, input_to_model=data)
            break
        '''
        '''
        3. training
        '''
        for epoch in range(start_epoch, paramH.train_epochs):
            print(f'Fold_{fold}:')
            epoch_loss, losses = train(net, train_loader, optimizer, criterion, device, epoch)
            
            all_train_loss = np.concatenate((all_train_loss, [losses.mean()]))
    
            test_loss, test_auc = test(net, test_loader, criterion, device)
            all_test_loss = np.append(all_test_loss, [test_loss])
            all_test_aucs = np.append(all_test_aucs, [test_auc])
    
            '''
            if (epoch+1) % 10 == 0:
                plot_auc_and_loss(all_train_loss, all_test_loss, all_test_aucs, epoch)
            '''
            ##
            # for tensorboard plots
            ##
            writer.add_scalars("TRAIN & VAL Loss", {f'F{fold}_TRAIN': epoch_loss, 
                                                    f'F{fold}_VAL': test_loss}, epoch)
            
            writer.add_scalars("VAL AUC", {f'F{fold}_VAL': test_auc}, epoch)
            
            # update best epoch
            if test_auc > max_auc:
                max_auc = test_auc
                max_auc_loss = test_loss
                best_epoch = epoch + 1
        
        '''
        4. Test one sequence.
        '''
        ''' 
        train_one_id = ['4E9M_1']
        test_one_data = train_data[train_data['id'].isin(train_one_id)]
        test_one_disorder = IDRDataset(data=test_one_data, feature_root=paramF.path_features, feature_type=paramH.featureType)
        sequence: Sequence = test_one_disorder[0]
        prediction = predict_one_sequence(net, sequence, device)
        for idx, (aa, pred) in enumerate(zip(sequence.sequence, prediction)):
            # print(f'{idx}\t{aa}\t{pred}')
            print(f'{idx}\n{pred}')
        '''
        '''
        5. saving
        '''

        
        pd.DataFrame({'epoch': list(range(1, (paramH.train_epochs+1))), 'train_loss':all_train_loss,
                      'test_loss':all_test_loss, 'test_auc': all_test_aucs}).to_csv(f'{paramF.auc_loss_pth}_f{fold}', index=False)
        
        logging.warning(f"model_name: {paramH.model_name}_fold{fold} \n \
                    num_params: {num_params} \n \
                    best_epoch: {best_epoch} \n \
                    test_loss: {max_auc_loss} \n \
                    max_auc: {max_auc} \n \
                    netName: {paramH.netName} \n \
                    batch_size: {paramH.batch_size} \n \
                     train_epochs: {paramH.train_epochs} \n \
                    learning_rate: {paramH.lr} \n")

        # Save Model
        save_checkpoint(net, optimizer, all_test_loss[-1], paramH.train_epochs, f'{paramF.model_pth}_f{fold}')