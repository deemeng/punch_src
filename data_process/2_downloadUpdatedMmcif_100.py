import params.PDBparser as paramsPDB
from utils.file import download_pdbe_updatedCif

import pandas as pd

'''
Aim to download all updated mmcif files from pdbe. 
Check more info about the differences between .pdb, .cif & *_updated.cif from 
    https://docs.google.com/document/d/1ESUldxIdOaUvJpU9tlXBc5xvwzX2UibggU69eTUAqHY/edit#heading=h.36rri1uau13d
    
Dee
12 Jan, 2024
'''
df_cls30 = pd.read_csv(paramsPDB.path_cls30_tab, header=None, sep='\t')
# 100% identity, download all sequences
# 231,624 entities from 168,082 entries.
list_entity = df_cls30.iloc[:, 1].unique()
list_entry = list(set([entity.split('_')[0] for entity in list_entity]))
print(f'Number of entities: {len(list_entity)}\nNumber od entries: {len(list_entry)}')

print('Downloading...')
download_pdbe_updatedCif(list_pdbid=list_entry, path_pdbe_cif=paramsPDB.path_mmcif)
