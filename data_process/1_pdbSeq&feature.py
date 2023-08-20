import pandas as pd
import numpy as np
import json

import params.PDBparser as paramsPDB
from utils.file import txt2list, generate_fasta_header, df2csv
from utils.common import get_pdbSeq_GraphQL

# 1. get all pdb entryIDs needed to download
path_file = paramsPDB.path_pdb_entryIDlist
delimiter = ','
list_all_entryID = txt2list(path_file, delimiter)
num_all_entryID = len(list_all_entryID)

print(f'Number of enrties: {num_all_entryID}')
step = 100
# generate batch downloading start&end point indexs
list_idx = np.arange(0, num_all_entryID, step).tolist() # step=100
if list_idx[-1]<num_all_entryID:
    list_idx.append(num_all_entryID)

# [{rcsb_id:, entity_id:, uniprot_accession:, pdbx_seq_one_letter_code_can:, rcsb_sample_sequence_length, asym_ids:, auth_asym_ids:, }, {}]
list_dict_pdbInfo = []

# 'w+', write & create file if it is not exist
with open(paramsPDB.path_pdb_seqEntity100, 'w+') as file:
    print('Downloading ... ...')
    for i in range(len(list_idx)-1):
        start = list_idx[i]
        end = list_idx[i+1]
        print(f'start: {start}, end: {end}')
        list_entryID = list_all_entryID[start:end]
        result = get_pdbSeq_GraphQL(list_entryID)
        dict_result = json.loads(result)
        list_entries = dict_result['data']['entries']
        
        for entry in list_entries:
            for entity in entry['polymer_entities']:
                '''
                1. save features
                '''
                if entity['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers'] is None:
                    continue
                if entity['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers'][0]['database_name']!='UniProt':
                    continue
                
                list_chains = entity['rcsb_polymer_entity_container_identifiers']['asym_ids']
                for i in range(len(list_chains)):
                    dict_entity = {}
                    dict_entity['rcsb_id'] = entry['rcsb_id']
                    dict_entity['entity_id'] = entity['rcsb_polymer_entity_container_identifiers']['entity_id']
                    # one entity could map to more than one uniprot sequence.
                    dict_entity['uniprot_accession'] = ','.join(uniprot['database_accession'] for uniprot in entity['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers'])

                    # dict_entity['sequence'] = entity['entity_poly']['pdbx_seq_one_letter_code_can']
                    dict_entity['sequence_length'] = entity['entity_poly']['rcsb_sample_sequence_length']

                    dict_entity['chain_id'] = list_chains[i]
                    dict_entity['auth_chain_id'] = entity['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'][i]

                    list_dict_pdbInfo.append(dict_entity)

                '''
                2. write sequence to a file
                '''
                # [entryID_entityID]
                header = generate_fasta_header([entry['rcsb_id']+'_'+entity['rcsb_polymer_entity_container_identifiers']['entity_id']])
                sequence = entity['entity_poly']['pdbx_seq_one_letter_code_can']

                file.write(header)
                file.write('\n')
                file.write(sequence)
                file.write('\n')
print('Saving features ... ...')
# save features
df_pdbInfo = pd.DataFrame(list_dict_pdbInfo)
df_pdbInfo['id'] = [rcsb_id+'_'+str(entity_id) for rcsb_id, entity_id in zip(df_pdbInfo['rcsb_id'], df_pdbInfo['entity_id'])]
df2csv(df_pdbInfo, paramsPDB.path_pdb_featureEntity100)
print('Done!!!')