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
        # store (concatenated) component arrays from each rollout
        # self.season = None
        # self.gameid = None
        # self.obs = None
        # self.acs = None
        # self.rews = None
        # self.next_obs = None
        # self.terminals = None
        
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
                
            
    def sample_random_data(self, batch_size):
        """
        Samples a batch of random transitions
        """
        # assert (
        #         self.obs.shape[0]
        #         == self.acs.shape[0]
        #         == self.rews.shape[0]
        #         == self.next_obs.shape[0]
        #         == self.terminals.shape[0]
        # )

        ## TODO return batch_size number of random entries\
        indices = np.random.choice(self.num_entries, batch_size, replace=False) # replace=False so no repeated
        
        batch_dict = dict()
        for key, value in self.components.items():
            batch_dict[key] = value[indices].copy()
        
        return batch_dict