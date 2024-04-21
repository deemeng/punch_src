import os
import Bio
from Bio import SeqIO

import params.PDBparser as paramsPDB
from utils.file import dump_list2json
from utils.pdbParser import get_missing_tab, get_missing_idx

import pandas as pd

df_cls30 = pd.read_csv(paramsPDB.path_cls30_tab, header=None, sep='\t')
# 23,581 entities with 30% identity, from 22,142 entries
list_entity = df_cls30.iloc[:, 0].unique()
list_entry = list(set([entity.split('_')[0] for entity in list_entity]))

df_pdbEntityInfo = pd.read_csv(paramsPDB.path_pdb_featureEntity100)
# df_pdbEntityInfo['entry_entity'] = [f'{entry}_{entity}' for entry, entity in zip(df_pdbEntityInfo['rcsb_id'], df_pdbEntityInfo['entity_id'])]
# df_pdbEntityInfo30 = df_pdbEntityInfo[df_pdbEntityInfo['entry_entity'].isin(list_entity)]
df_pdbEntityInfo30 = df_pdbEntityInfo[df_pdbEntityInfo['id'].isin(list_entity)]
dict_entity_chain = {k: list(v) for k, v in df_pdbEntityInfo30.groupby(['rcsb_id', 'entity_id'])['chain_id']}

# list of dicts {'id':'', 'sequence':'', 'reference': ''}
list_entityDataset = []

with open(paramsPDB.path_cls30Rep) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        id_entryEtity = record.id
        seq_entity = record.seq
        print(id_entryEtity)
        
        entryID = id_entryEtity.split('_')[0]
        entityID = int(id_entryEtity.split('_')[1])
        
        key = (entryID, entityID)
        chains = dict_entity_chain[key]
        path_entry = os.path.join(paramsPDB.path_mmcif, f'{entryID.lower()}_updated.cif')
        
        df_missing = get_missing_tab(path_entry)
        missing_idx = get_missing_idx(df_missing, chains)
        
        list_reference = [1 if i+1 in missing_idx else 0 for i in range(len(seq_entity))]
        
        # list of dictionaries
        entityData = {}
        entityData['id'] = id_entryEtity
        entityData['sequence'] = str(seq_entity)
        entityData['reference'] = ''.join(str(x) for x in list_reference)
        list_entityDataset.append(entityData)
print('Saving the dataset... ...')
dump_list2json(list_entityDataset, paramsPDB.path_IDRdataset)
print('Done!!!')