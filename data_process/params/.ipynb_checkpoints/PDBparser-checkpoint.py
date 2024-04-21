import os

# idr/
# ROOT = os.path.realpath('..')
ROOT = os.path.realpath('..')

###
# 1. Data
###
path_data = os.path.join(ROOT, 'data')

# 1.1. pdb data
path_pdb = os.path.join(path_data, 'pdb')

path_pdb_entryIDlist = os.path.join(path_pdb, 'list_file.txt') # list of pdb entries (x-ray + protein) to download
path_pdb_allSeqChain = os.path.join(path_pdb, 'pab_seqres.txt') # all sequences (entriyID_chainID) from PDB

path_pdb_seqEntity100 = os.path.join(path_pdb, 'pdb_entity100.fasta')
path_pdb_featureEntity100 = os.path.join(path_pdb, 'pdb_entityFeature100.csv')
path_pdb_featureEntity30 = os.path.join(path_pdb, 'pdb_entityFeature30.csv')

# 30% identity
path_pdb30 = os.path.join(path_pdb, 'pdb_entity30')
path_cls30Rep = os.path.join(path_pdb30, 'clusterRes_rep_seq.fasta')
path_cls30_tab = os.path.join(path_pdb30, 'clusterRes_cluster.tsv')

# 100% identity
path_pdb100 = os.path.join(path_pdb, 'pdb_entity100')
path_cls100Rep = os.path.join(path_pdb100, 'clusterRes_rep_seq.fasta')
path_cls100_tab = os.path.join(path_pdb100, 'clusterRes_cluster.tsv')
path_pdb100_seq = os.path.join(path_pdb100, 'seq')

path_mmcif = os.path.join(path_pdb, 'updated_mmcif') # updated_mmcif from PDBe


# 1.2. dataset
path_dataset = os.path.join(path_data, 'dataset')

path_IDRdataset = os.path.join(path_dataset, 'IDRdataset.json')
path_IDRdataset_100 = os.path.join(path_dataset, 'IDRdataset_100.json')
path_IDRdataset_smoothed = os.path.join(path_dataset, 'IDRdataset_smoothed.json')
path_IDR_hmmInfo = os.path.join(path_dataset, 'IDR_hmmInfo.json')

path_train = os.path.join(path_dataset, f'all_TrainDataset.json')
path_test = os.path.join(path_dataset, f'all_TestDataset.json')

path_train_withFullyDisorder = os.path.join(path_dataset, f'withFullyDisorder_TrainDataset.json')
path_test_withFullyDisorder = os.path.join(path_dataset, f'withFullyDisorder_TestDataset.json')

# 1.3. embedding
path_features = os.path.join(path_data, 'features')

path_embedded_protTrans = os.path.join(path_features, 'protTrans')
path_embedded_esm2 = os.path.join(path_features, 'esm2')
path_embedded_msaTrans = os.path.join(path_features, 'msaTrans')
path_embedded_onehot = os.path.join(path_features, 'onehot')

path_embedded_hmm = os.path.join(path_features, 'hmm')
path_embedded_hmm_prob = os.path.join(path_embedded_hmm, 'prob')
path_embedded_hmm_aa = os.path.join(path_embedded_hmm, 'aa')
path_embedded_hmm_prob_NumTemplate = os.path.join(path_embedded_hmm, 'prob_NumTemplate')
path_embedded_hmm_aa_NumTemplate = os.path.join(path_embedded_hmm, 'aa_NumTemplate')

# 2. CAID
path_caid = os.path.join(path_data, 'caid')
path_idr_pdb = os.path.join(path_caid, 'disorder_pdb.fasta')
path_idr_pdb_folder = os.path.join(path_caid, 'seq')

path_caid_dataset = os.path.join(path_caid, 'dataset')
path_caid_dataset_json = os.path.join(path_caid_dataset, 'caid_dataset.json')

path_caid_features = os.path.join(path_caid, 'features')
path_caid_features_protTrans = os.path.join(path_caid_features, 'protTrans')
path_caid_features_onehot = os.path.join(path_caid_features, 'onehot')
path_caid_features_esm2 = os.path.join(path_caid_features, 'esm2')
path_caid_features_msaTrans = os.path.join(path_caid_features, 'msaTrans')

# 3. Disprot
path_disprot = os.path.join(path_data, 'disprot')

path_disprot_dataset = os.path.join(path_disprot, 'dataset')
path_disprot_all = os.path.join(path_disprot_dataset, 'all_DisProt release_2023_06 with_ambiguous_evidences.json')
path_disprot_IDPO_tab = os.path.join(path_disprot_dataset, 'IDPO_v0.3.0.csv')

path_disprot_dataset_disordered = os.path.join(path_disprot_dataset, 'disprot_dataset_disordered.json') # all Disprot sequences
path_disprot_dataset_fullDisordered = os.path.join(path_disprot_dataset, 'disprot_dataset_fullDisordered.json') # Disprot: fully disordered sequences only

path_disprot_dataset_fullDisordered_seq_folder = os.path.join(path_disprot, 'seq')
path_disprot_dataset_fullDisordered_seq = os.path.join(path_disprot_dataset, 'disprot_dataset_fullDisordered.fasta')

path_disprot_features = os.path.join(path_disprot, 'features')
path_disprot_features_protTrans = os.path.join(path_disprot_features, 'protTrans/')
path_disprot_features_onehot = os.path.join(path_disprot_features, 'onehot/')

path_disprot_alphafoldDB = os.path.join(path_disprot, 'alphafoldDB')

# 4. merge Disprot&IDR
path_IDR_fullyDisordered_dataset = os.path.join(path_dataset, 'IDR_fullyDisordered_dataset.json')
path_IDR_fullyDisordered_dataset_100 = os.path.join(path_dataset, 'IDR_fullyDisordered_dataset_100.json')

# 5. CAID vs IDR/Disprot
path_CAIDvsIDR = os.path.join(path_data, 'CAIDvsIDR')
path_CAIDvsDisprot = os.path.join(path_data, 'CAIDvsDisprot')