# different text color for different situations.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    
class type_dataset:
    
    ALL = 'all'
    IDRONLY = 'idrOnly'
    SMOOTH_ALL = 'smooth_all'
    SMOOTH_IDRONLY = 'smooth_idrOnly'
    
    type_dataset_info = {
        'all': 'no smoothing, all entities', 
        'idrOnly': 'no smoothing, only entities contain IDRs', 
        'smooth_all': 'smoothing, all entities', 
        'smooth_idrOnly': 'smoothing, only entities contain IDRs'
    }
    
    @classmethod
    def get_info(cls, x):
        return cls.type_dataset_info[x]