'''
Functions related to parse the specific txt file

Input file format:
--------------
>disprot_ID
sequence
reference (3 classes: 0, 1, -), '-' means annotation not available.
--------------

Output file format
--------------
id, disprot_id
sequence
reference (3 classes: 0, 1, -1), '-1' means annotation not available.
length, sequence_length
contain_idr, 1/0
--------------
Di.
18 Oct, 2023
'''

def get_labelInfo(seq_label: list):
    '''
    give a list of labels of a sequence, return 
    - the number of positive regions, 
    - and the start/end idx of each positive region.

    params:
        seq_label - list of 0/1 s. could be the label of linker or disorder, depending on which project you are working on.
    
    
    return:
        num_positiveRegion - int
        dict_regions - {1: {'start': , 'end': , 'len_linker':}, 2: {...}, ...}
    '''
    count_len = 0
    count_region = 0
    # if this in the middle of a positive region

    dict_regions = {}
    dict_singleRegion = {}
    for i, label in enumerate(seq_label):
        if label==1:
            count_len = count_len + 1
            if count_len==1: # the start of a region
                count_region = count_region + 1
                dict_singleRegion['start'] = i
        if ((label==0 or label==-1) or i==len(seq_label)-1) and (count_len>0): # the end of a region
            dict_singleRegion['end'] = i - 1
            if i==len(seq_label)-1:
                dict_singleRegion['end'] = i
            dict_singleRegion['len_linker'] = dict_singleRegion['end'] - dict_singleRegion['start'] + 1
            dict_regions[count_region] = dict_singleRegion # save region
            count_len = 0 # reset the count length
            dict_singleRegion = {} # reset singleResion dictionary
    return dict_regions, count_region

def read_fasta2list(file_path):
    list_dataset = [] # save all dict_seq into a list
    dict_seq = {} # save info to a dict, {'id':str, 'length':int, 'reference':[]}
    line_counter = 0
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read and process each line one by one
        for line in file:
            if line_counter%3==0:
                dict_seq['id'] = line.strip()[1:] # >ID
            elif line_counter%3==1:
                dict_seq['sequence'] = line.strip()
                dict_seq['length'] = len(line.strip())
            elif line_counter%3==2:
                dict_seq['reference'] = [-1 if r=='-' else int(r) for r in line.strip()]
                dict_seq['info_idr'], dict_seq['num_idr'] = get_labelInfo(dict_seq['reference'])
                list_dataset.append(dict_seq)
                dict_seq = {}
            # next line
            line_counter += 1
    return list_dataset