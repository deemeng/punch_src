import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from torch.utils.data import DataLoader
from dataset.idr_dataset import Sequence
import params.filePath as paramF
import params.hyperparams as paramH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trim_padding_and_flat(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_trimmed_pred = np.array([])
    for i, seq in enumerate(sequences):
        tmp_pred = pred[i][:len(seq)].cpu().detach().numpy()
        # tmp_pred = pred[i].cpu().detach().numpy()
        all_target = np.concatenate([all_target, seq.clean_target])
        all_trimmed_pred = np.concatenate([all_trimmed_pred, tmp_pred])
    return all_target, all_trimmed_pred

def concat_target_and_output(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_pred = np.array([])
    for i, seq in enumerate(sequences):
        all_target = np.concatenate([all_target, seq.clean_target])
    all_pred = pred.cpu().detach().numpy()
    return all_target, all_pred

def get_targetPred(sequences: List[Sequence], pred):    
    if paramH.padding:
        target, pred = trim_padding_and_flat(sequences, pred)
    else:
        target, pred = concat_target_and_output(sequences, pred)
    return target, pred
    
def batch_auc(target, pred):
    '''
    Given target&pred, calculate AUC score.
    params:
        target - np.array, ground truth
        pred - np.array, predicted valued by a predictor.

    return:
        auc - float, auc score.
    '''
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def get_batch_PreTargetList(pred, target, lens):
    '''
    if batch_size>1, ignore the padding regions and get the actural pred and target lists.

    params:
        pred - list, list of padded prediction values.
        target - list, list of padded target values.
        lens - list, true lengths of the sequences.
    return:
        pre_list - list, all predictions for multiple sequences
        target_list - list, all true values for multiple sequences
    '''
    pre_list = []
    target_list = []
    
    for p, t, l in zip(pred, target, lens):
        pre_list += p[:l].tolist()
        target_list += t[:l].tolist()
        
    return pre_list, target_list

def batch_auc_rnn(pre_list, target_list):
    fpr, tpr, thresholds = metrics.roc_curve(target_list, pre_list, pos_label=1)   
    auc = metrics.auc(fpr, tpr)
    return auc

def plot_auc_and_loss(train_losses, test_losses, test_aucs, epoch, title="AUC and Loss"):
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(8.5, 7.5))

    x_test = np.arange(1, epoch + 2)
    x_train = np.linspace(0, epoch + 1, len(train_losses))

    ax1.plot(x_train, train_losses, color='slategrey', linewidth=1, label='Train Loss')
    ax1.plot(x_test, test_losses, color='dodgerblue', marker='o', linewidth=2, label='Test Loss')
    max_ticks = 22
    ax1.set_xticks(np.linspace(0, epoch + 2, max_ticks, dtype=int))
    ax1.tick_params(axis='y', color='slategrey', labelcolor='slategrey')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(x_test, test_aucs, color='orange', marker='o', linewidth=2, label='Test AUC')
    ax2.tick_params(axis='y', color='orange', labelcolor='orange')
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylabel('AUC')
    # Set the minimum y-axis value to 0.0 and maximum y-axis value to 1.0 (AUC is between 0.0 and 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, which='major', axis='y', linestyle='dotted')

    plt.title(f'{paramH.model_name}: '+title)
    fig.legend(ncol=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(f'{paramF.plots_dir}{paramH.model_name}_{title}_{epoch}.png')
    # plt.show()
    return fig

## not using
def plot_roc_curve_rnn(model, data_loader, device, setType='Test'):
    model.eval()
    
    epoch_pre = []
    epoch_target = []
    with torch.no_grad():
        for sequences, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target, lens = pad_packed_sequence(target, batch_first=True)
            
            output = model(data)
            
            batch_pre, batch_target = get_batch_PreTargetList(output.cpu(), target.cpu(), lens)
            epoch_pre = epoch_pre + batch_pre
            epoch_target = epoch_target + batch_target

    fpr, tpr, thresholds = metrics.roc_curve(epoch_target, epoch_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    r = np.linspace(0, 1, 1000)
    fs = np.mean(np.array(np.meshgrid(r, r)).T.reshape(-1, 2), axis=1).reshape(1000, 1000)
    cs = ax.contour(r[::-1], r, fs, levels=np.linspace(0.1, 1, 10), colors='silver', alpha=0.7, linewidths=1,
                    linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f', fontsize=10, manual=[(l, 1 - l) for l in cs.levels[:-1]])
    ax.plot(fpr, tpr, color='orange', linewidth=1, label=f'{setType} AUC = %0.3f' % auc)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.legend(loc='lower right')
    plt.title(f'{paramH.model_name}: ROC Curve for {setType} Set')
    plt.savefig(f'{paramF.plots_dir}{paramH.model_name}_{setType}_ROC_curve.png')
    # plt.show()
    
    return fig

# Function that get the results from the model on the test set and plot the ROC curve
def plot_roc_curve(model, data_loader, device, setType='Test'):
    model.eval()
    all_output, all_target = np.array([]), np.array([])
    with torch.no_grad():
        for sequences, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data).cpu()

            if paramH.padding:
                target, output = trim_padding_and_flat(sequences, output)
            else:
                target, output = concat_target_and_output(sequences, output)
            all_target = np.concatenate([all_target, target])
            all_output = np.concatenate([all_output, output.reshape(-1)])

    fpr, tpr, thresholds = metrics.roc_curve(all_target, all_output, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # plot
    plot_title = f'{paramH.model_name}: ROC Curve for {setType} Set'
    file_path = f'{paramF.plots_dir}{paramH.model_name}_{setType}_ROC_curve.png'
    fig = plot_auc(fpr, tpr, auc, plot_title, file_path, setType)
    return fig

def plot_auc(fpr, tpr, auc, plot_title, file_path, setType='Test'):
    fig, ax = plt.subplots(figsize=(8, 8))
    r = np.linspace(0, 1, 1000)
    fs = np.mean(np.array(np.meshgrid(r, r)).T.reshape(-1, 2), axis=1).reshape(1000, 1000)
    cs = ax.contour(r[::-1], r, fs, levels=np.linspace(0.1, 1, 10), colors='silver', alpha=0.7, linewidths=1,
                    linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f', fontsize=20, manual=[(l, 1 - l) for l in cs.levels[:-1]])
    ax.plot(fpr, tpr, color='green', linewidth=3, label=f'{setType} AUC = %0.3f' % auc)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlabel("FPR", fontsize=20)
    ax.set_ylabel("TPR", fontsize=20)
    plt.legend(loc='lower right', fontsize=20)
    plt.title(plot_title, fontsize=20)
    plt.rcParams['font.size'] = 20
    plt.savefig(file_path)
    # plt.show()
    return fig

# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        seq_loss = criterion(output[i][:len(seq)], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        # print(output)
        # seq_loss = criterion(output[i], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)

# for rnn unpadding model
def get_loss_rnn(output, target, lens, criterion) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for o, t, l in zip(output, target, lens):
        seq_loss = criterion(o[:l], t[:l])
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(lens)

# save and load model
def save_checkpoint(net, optimizer, Loss, EPOCH, PATH):
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
                }, PATH)

def save_model(net, optimizer, EPOCH, PATH):
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)
    
def load_checkpoint(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(losslogger, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch, losslogger

def load_model(net, optimizer, PATH):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("=> (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return net, optimizer, start_epoch

def count_modelParams(net):
    '''
    Given a mdoel, count the parameters inside this model.
    params:
        net - nn. Module

    return:
        int, number of params.
    '''
    return sum(p.numel() for p in net.parameters())
