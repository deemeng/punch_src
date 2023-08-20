import os
from typing import List
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset.idr_dataset import IDRDataset, Sequence, collate_fn
from utils.main_support import selectCol, getPath

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
    path_train, path_test = getPath(paramH.datasetType)
    # Load the data
    train_data = pd.read_json(path_train, orient='records', dtype=False)
    test_data = pd.read_json(path_test, orient='records', dtype=False)
    
    # select columns
    train_data, test_data = selectCol(train_data, test_data, paramH.datasetType)

    # Best epoch
    max_auc = -1
    max_auc_loss = -1
    best_epoch = -1

    # Filter protein length less than 
    train_data['p_len'] = train_data['sequence'].map(lambda x: len(x))
    train_data = train_data[train_data['p_len']<=paramH.MAX_seq_length]

    test_data['p_len'] = test_data['sequence'].map(lambda x: len(x))
    test_data = test_data[test_data['p_len']<=paramH.MAX_seq_length]
    
    # IDRDataset
    train_disorder = IDRDataset(data=train_data, feature_root=paramF.path_features)
    test_disorder = IDRDataset(data=test_data, feature_root=paramF.path_features)

    # Defining the dataloader for the training set and the test set
    train_loader = DataLoader(train_disorder, batch_size=paramH.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader = DataLoader(test_disorder, batch_size=paramH.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,
                             pin_memory=True)

    '''
    2. model
    '''
    # Instantiate the model
    net = model.Net(in_features=paramH.n_features, dropout=paramH.dropout).to(device)
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
        net, _, start_epoch, _ = load_checkpoint(net, optimizer, paramF.model_pth)
        
        df_auc_loss = pd.read_csv(paramF.auc_loss_pth)
        all_train_loss = np.array(df_auc_loss['train_loss'])
        all_test_loss = np.array(df_auc_loss['test_loss'])
        all_test_aucs = np.array(df_auc_loss['test_auc'])
        
    # add model graph to tensorboard.
    for batch_idx, (sequences, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        if paramH.transpose:
            writer.add_graph(net, input_to_model=data.transpose(1, 2))
        else:
            writer.add_graph(net, input_to_model=data)
        break
    '''
    3. training
    '''
    for epoch in range(start_epoch, paramH.train_epochs):
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
        writer.add_scalars("TRAIN & VAL Loss", {'TRAIN': epoch_loss, 
                                                'VAL': test_loss}, epoch)

        writer.add_scalars("VAL: AUC", {'AUC': test_auc}, epoch)
        '''
        writer.add_scalars("VAL: Loss vs auc", {'Loss': test_loss, 
                                                'AUC': test_auc}, epoch)
        '''
        # update best epoch
        if test_auc > max_auc:
            max_auc = test_auc
            max_auc_loss = test_loss
            best_epoch = epoch + 1
            
    plot_auc_and_loss(all_train_loss, all_test_loss, all_test_aucs, paramH.train_epochs-1)
    fig_test = plot_roc_curve(net, test_loader, device)
    fig_train = plot_roc_curve(net, train_loader, device, setType='Train')
    
    # writer.add_figure(f'VAL: Loss vs auc', fig_auc_loss)

    writer.add_figure(f'Train: roc curve', fig_train)
    writer.add_figure(f'VAL: roc curve', fig_test)
    
    '''
    4. Test one sequence.
    '''
    
    train_one_id = ['4E9M_1']
    test_one_data = train_data[train_data['id'].isin(train_one_id)]
    test_one_disorder = IDRDataset(data=test_one_data, feature_root=paramF.path_features)
    sequence: Sequence = test_one_disorder[0]
    prediction = predict_one_sequence(net, sequence, device)
    for idx, (aa, pred) in enumerate(zip(sequence.sequence, prediction)):
        # print(f'{idx}\t{aa}\t{pred}')
        print(f'{idx}\n{pred}')
    
    '''
    5. saving
    '''
    # save losses and aucs
    pd.DataFrame({'epoch': list(range(1, (paramH.train_epochs+1))), 'train_loss':all_train_loss,
                  'test_loss':all_test_loss, 'test_auc': all_test_aucs}).to_csv(paramF.auc_loss_pth, index=False)

    logging.warning(f"model_name: {paramH.model_name} \n \
                num_params: {num_params} \n \
                best_epoch: {best_epoch} \n \
                test_loss: {max_auc_loss} \n \
                max_auc: {max_auc} \n \
                netName: {paramH.netName} \n \
                batch_size: {paramH.batch_size} \n \
                 train_epochs: {paramH.train_epochs} \n \
                learning_rate: {paramH.lr} \n")
    
    # save Model
    save_checkpoint(net, optimizer, all_test_loss[-1], paramH.train_epochs, paramF.model_pth)
    # torch.save(net.state_dict(),model_pth)
