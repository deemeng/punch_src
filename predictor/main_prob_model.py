import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import metrics

from utils.main_support import selectCol, getPath
from utils.static import protein_seq
from utils.sequence import sequence_mapping_str

import params.filePath as paramF
import params.hyperparams as paramH

def count_num(num_AAisIDR, seq, ref):
    '''
    count the number of each amino acid which labeled as disordered.
    params:
        num_AAisIDR - dict, {aa_name: number of times labeled as IDR}
        seq - str, one protein sequence
        ref - list, list of labels, 0 or 1.

    return:
        num_AAisIDR - updated num_AAisIDR
        num_IDR - int, number of AAs which been annotated as 1 (disordered)
        num_aa - int, sequence length.
    '''
    num_IDR = 0 
    num_aa = len(seq)
    for i in range(num_aa):
        if int(ref[i])==1:
            num_AAisIDR[seq[i]] = num_AAisIDR[seq[i]] + 1
            num_IDR = num_IDR + 1
    return num_AAisIDR, num_IDR, num_aa

def calculate_prob(df_data):
    '''
    calculate the probability to be disordered for each amino acid
    params:
        df_data
    '''
    prob_AAisIDR = {aa:0.0 for aa in protein_seq.amino_acids}
    num_AAisIDR = {aa:0 for aa in protein_seq.amino_acids}
    
    num_aa = 0
    num_IDR = 0
    for idx, row in train_data.iterrows():
        seq = sequence_mapping_str(row['sequence']) # replace rare AAs with X
        ref = row['reference']
        num_AAisIDR, num_IDR_1, num_aa_1 = count_num(num_AAisIDR, seq, ref)
        num_aa = num_aa + num_aa_1
        num_IDR = num_IDR + num_IDR_1
        
    prob_AAisIDR = {k:num_AAisIDR[k]/num_aa for k in num_AAisIDR.keys()}
    return prob_AAisIDR, num_AAisIDR, num_aa, num_IDR

def predict_one_sequence(prob_AAisIDR, seq):
    '''
    predict one protein sequence.
    params:
        prob_AAisIDR - {'aa1': prob, ...}
        seq - str, AMCDLASP...
    return:
        pre_idr - list, [0.1, 0.32, 0.05, ....]
    '''
    pre_idr = [prob_AAisIDR[aa] for aa in seq]
    return pre_idr

def predict_test(prob_AAisIDR, df_test):
    '''
    Assign the probability of each AA being disordered to all the sequences, and generate the prediction.
    params:
        prob_AAisIDR - {'aa1': prob, ...}
        df_test - dataframe, columns: id (pdbID_entityID), sequence, reference.
    return:
        list_pred - list, pretions
        list_label - list, ground truth.
    '''
    list_pred = []
    list_label = []
    for _, row in df_test.iterrows():
        seq = sequence_mapping_str(row['sequence'])
        ref = row['reference']
        list_label = list_label + [int(r) for r in ref]
        list_pred = list_pred + predict_one_sequence(prob_AAisIDR, seq)

    return list_pred, list_label

# Function that get the results from the model on the test set and plot the ROC curve
def plot_roc_curve(pred: list, label: list):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 8))
    r = np.linspace(0, 1, 1000)
    fs = np.mean(np.array(np.meshgrid(r, r)).T.reshape(-1, 2), axis=1).reshape(1000, 1000)
    cs = ax.contour(r[::-1], r, fs, levels=np.linspace(0.1, 1, 10), colors='silver', alpha=0.7, linewidths=1,
                    linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f', fontsize=20, manual=[(l, 1 - l) for l in cs.levels[:-1]])
    ax.plot(fpr, tpr, color='green', linewidth=3, label=f'AUC = %0.3f' % auc)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlabel("FPR", fontsize=20)
    ax.set_ylabel("TPR", fontsize=20)
    plt.legend(loc='lower right', fontsize=20)
    plt.title(f'prob_model: ROC Curve', fontsize=20)
    plt.rcParams['font.size'] = 20
    plt.savefig(f'prob_model_ROC_curve.png')
    # plt.show()
    
    return fig

# Defining main function
def main():
	# get file path
    path_train, path_test = getPath(paramH.datasetType)
    # Load the data
    train_data = pd.read_json(path_train, orient='records', dtype=False)
    test_data = pd.read_json(path_test, orient='records', dtype=False)
    # select columns
    train_data, test_data = selectCol(train_data, test_data, paramH.datasetType)
    
    prob_AAisIDR, num_AAisIDR, num_aa, num_IDR = calculate_prob(train_data)
    print(f'The percentage of Disorder: {num_IDR/num_aa}')
    list_pred, list_label = predict_test(prob_AAisIDR, test_data)

# Using the special variable
# __name__
if __name__=="__main__":
	main()