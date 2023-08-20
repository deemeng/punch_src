import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

'''
functions related to parse updated_mmcif file (from PDBe)

Dee.
21 jul, 2023
'''

def get_missing_tab(path_mmcif_entry):
    '''
    get missing related features table.
    params:
        path_mmcif_entry - string, path to updated_mmcif file, (*_updated.cif)
    
    return:
        df_missing - Dataframe
    '''
    missing_keys = ['_pdbx_unobs_or_zero_occ_residues.id', '_pdbx_unobs_or_zero_occ_residues.PDB_model_num', '_pdbx_unobs_or_zero_occ_residues.polymer_flag',
            '_pdbx_unobs_or_zero_occ_residues.occupancy_flag', '_pdbx_unobs_or_zero_occ_residues.auth_asym_id', 
            '_pdbx_unobs_or_zero_occ_residues.auth_comp_id', '_pdbx_unobs_or_zero_occ_residues.auth_seq_id', 
            '_pdbx_unobs_or_zero_occ_residues.PDB_ins_code', '_pdbx_unobs_or_zero_occ_residues.label_asym_id', 
            '_pdbx_unobs_or_zero_occ_residues.label_comp_id', '_pdbx_unobs_or_zero_occ_residues.label_seq_id', ]
    mmcif_entry = MMCIF2Dict(path_mmcif_entry)
    dict_missing = {}
    try:
        for k in missing_keys:
            dict_missing[k.split('.')[1]] = mmcif_entry[k]
        df_missing = pd.DataFrame(dict_missing)
    except:
        df_missing = pd.DataFrame({})
    return df_missing

def get_missing_idx(df_missing, chains):
    '''
    Given missing-features table and the chains related to an entity, return the list of missing label_seq_id.
    params:
        df_missing - Dataframe, including the features. 
        chains - list, all chains related to the entity. e.g. ['A', 'B']
    return:
        missing_idx - list of missing residue label_seq_id, the label_seq_id (PDB index for an entity sequence) of a sequence always start from 1.
    '''
    
    if df_missing.shape[0]==0: # no missing residue
        return []
    
    missing_idx = []
    for c in chains:
        df_missing_c = df_missing[df_missing['label_asym_id']==c]
        c_missing_idx = pd.to_numeric(df_missing_c['label_seq_id']).tolist()
        if len(missing_idx)==0:
            missing_idx = c_missing_idx
        else:
            missing_idx = list(set(missing_idx)&set(c_missing_idx))
    return missing_idx

def ref_smoothing(reference, threshold_len=4):
    '''
    params:
        reference - str, annotations for a PBD entity
        threshold_len - int, the minimum length of an IDR region, less than this threshold will be relabeled as 0. default is 4.
        
    return:
        new_reference - string, smoothed reference.
        contain_idr/contain_idr_smoothed - int, 1-contain idr
                                                0-does not contain idr
    '''
    new_reference = reference
    
    contain_idr = 0
    contain_idr_smoothed = 0
    # number of continious 
    count = 0
    for i in range(len(reference)):
        if int(reference[i])==1:
            count += 1
            contain_idr = 1
            # last item
            if i==len(reference)-1:
                if count < threshold_len:
                    new_reference = new_reference[:(i-count+1)] + '0'*count
                else:
                    contain_idr_smoothed = 1
        else:
            if count < threshold_len:
                new_reference = new_reference[:(i-count)] + '0'*count + new_reference[i:]
            else:
                contain_idr_smoothed = 1
            count = 0
    return new_reference, contain_idr_smoothed, contain_idr