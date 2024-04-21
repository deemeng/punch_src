def get_numFeature(featureType):
    '''
    return number of features based on the feature type.
    '''
    MAX_seq_length = 400000
    if featureType=='protTrans':
        n_features = 1024
    elif featureType=='esm2':
        n_features = 1280
    elif featureType=='msa_transformer':
        n_features = 768
        MAX_seq_length = 1022
    elif featureType=='onehot':
        n_features = 21
    elif featureType=='hmm_prob' or featureType=='hmm_aa':
        n_features = 22
    elif featureType=='hmm_prob_numTemp' or featureType=='hmm_aa_numTemp':
        n_features = 23
    elif featureType=='hmm_prob@onehot':
        n_features = 43
    elif featureType=='hmm_prob_numTemp@onehot':
        n_features=44
    elif featureType=='protTrans@onehot':
        n_features=1045
    elif featureType=='esm2@onehot':
        n_features=1301
    elif featureType=='msa_transformer@onehot':
        MAX_seq_length = 1022
        n_features=789
    elif featureType=='msa_transformer@hmm_prob_numTemp':
        MAX_seq_length = 1022
        n_features=791
    elif featureType=='protTrans@hmm_prob_numTemp':
        n_features=1047
    elif featureType=='protTrans@msa_transformer':
        MAX_seq_length = 1022
        n_features=1024+768
    return n_features, MAX_seq_length