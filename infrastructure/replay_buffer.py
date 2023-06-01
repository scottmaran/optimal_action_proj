"""
A simple, generic replay buffer

"""
#from infrastructure.utils import *
import pandas as pd
import numpy as np

class ReplayBuffer():
    """
    Defines a replay buffer to store past trajectories
    """
    def __init__(self, max_size=1000000, filepath=None, test=0):
        self.max_size = max_size
        # store each rollout
        self.paths = []
        
        # dictionary to store components
        self.components = dict()
        self.train = None
        self.val = None
        self.test = None
        self.num_entries = 0

        self.filepath = filepath
        if self.filepath != None:
            self.add_initial(filepath)
            
        # if test set
        if test != 0 :
            self.train = dict()
            self.val = dict()
            self.test = dict()

    def __len__(self):
        if len(self.components) == 0:
            return 0
        else:
            # returns first value's shape in dictionary
            return self.components[next(iter(self.components))].shape[0]
        
    ''' 
    Split components into train/val/test dict 
    '''
    def split_data(self, train_split=0.7):
        
        train_num = int(self.num_entries*train_split)
        val_num = (self.num_entries - int(self.num_entries*train_split))//2

        indices = list(range(self.num_entries))
        train_indices = indices[0:train_num]
        val_indices = indices[train_num:train_num+val_num]
        test_indices = indices[train_num+val_num:]

        #check to make sure slices correct
        assert self.num_entries == len(train_indices) + len(val_indices) + len(test_indices)

        for key, value in self.components.items():
            self.train[key] = value[train_indices]
        for key, value in self.components.items():
            self.val[key] = value[val_indices]
        for key, value in self.components.items():
            self.test[key] = value[test_indices]
        
    def add_initial(self, filepath):
        # read in pickle file
        df = pd.read_pickle(filepath)
        # store each column as numpy array
        for col_name in df.columns:
            
            stacked = np.stack(df[col_name].values).squeeze()  # convert pandas column of arrays into one array
            if len(stacked.shape) > 2:
                self.components[col_name] = stacked.reshape(stacked.shape[0], -1)
            else:
                self.num_entries = stacked.shape[0]
                self.components[col_name] = stacked
                
            
    def sample_random_data(self, batch_size, data_dict_to_use=None):
        """
        Samples a batch of random transitions
        """
        batch_dict = dict()
        
        if data_dict_to_use == 'train':
            size = self.train[next(iter(self.train))].shape[0]
            indices = np.random.choice(size, batch_size, replace=False) # replace=False so no repeated
            for key, value in self.train.items():
                batch_dict[key] = value[indices]
        elif data_dict_to_use == 'val':
            size = self.val[next(iter(self.val))].shape[0]
            indices = np.random.choice(size, batch_size, replace=False) # replace=False so no repeated
            for key, value in self.val.items():
                batch_dict[key] = value[indices]
        elif data_dict_to_use == 'test':
            size = self.test[next(iter(self.test))].shape[0]
            indices = np.random.choice(size, batch_size, replace=False) # replace=False so no repeated
            for key, value in self.test.items():
                batch_dict[key] = value[indices]
        else:
            size = self.components[next(iter(self.components))].shape[0]
            indices = np.random.choice(size, batch_size, replace=False) # replace=False so no repeated
            for key, value in self.components.items():
                batch_dict[key] = value[indices]
        
        return batch_dict