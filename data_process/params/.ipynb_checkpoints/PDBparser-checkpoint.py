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

path_mmcif = os.path.join(path_pdb, 'updated_mmcif') # updated_mmcif from PDBe


# 1.2. dataset
path_dataset = os.path.join(path_data, 'dataset')

path_IDRdataset = os.path.join(path_dataset, 'IDRdataset.json')
path_IDRdataset_smoothed = os.path.join(path_dataset, 'IDRdataset_smoothed.json')

# 1.3. embedding
path_features = os.path.join(path_data, 'features')

path_embedded_protTrans = os.path.join(path_features, 'protTrans')
path_embedded_onehot = os.path.join(path_features, 'onehot')
path_embedded_hmm = os.path.join(path_features, 'hmm')