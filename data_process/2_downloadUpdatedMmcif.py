import params.PDBparser as paramsPDB
from utils.file import download_pdbe_updatedCif

import pandas as pd

'''
Aim to download all updated mmcif files from pdbe. 
Check more info about the differences between .pdb, .cif & *_updated.cif from 
    https://docs.google.com/document/d/1ESUldxIdOaUvJpU9tlXBc5xvwzX2UibggU69eTUAqHY/edit#heading=h.36rri1uau13d
    
Dee
21 Jul, 2023
'''
df_cls30 = pd.read_csv(paramsPDB.path_cls30_tab, header=None, sep='\t')
# 23,581 entities with 30% identity, from 22,142 entries
list_entity = df_cls30.iloc[:, 0].unique()
list_entry = list(set([entity.split('_')[0] for entity in list_entity]))
print(f'Number of entities: {len(list_entity)}\nNumber od entries: {len(list_entry)}')

print('Downloading')
download_pdbe_updatedCif(list_pdbid=list_entry, path_pdbe_cif=paramsPDB.path_mmcif)
