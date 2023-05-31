"""
Defines a trainer which updates a behavior cloning agent
"""

import torch
import numpy as np
import pandas as pd

from infrastructure import pytorch_util as ptu

class BCTrainer:
    """
    A class which defines the training algorithm for the agent. Handles
    sampling data, updating the agent, and logging the results.

    ...

    Attributes
    ----------
    agent : BCAgent
        The agent we want to train

    Methods
    -------
    run_training_loop:
        Main training loop for the agent
    collect_training_trajectories:
        Collect data to be used for training
    train_agent
        Samples a batch and updates the agent
    """
    
    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        #self.logger = Logger(self.params['logdir'])
    
        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )
        
        #############
        ## AGENT
        #############
        ac_dim = 8
        ob_dim = 368
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])
        
    
    def run_training_loop(self, epochs):
        '''
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        '''

        #self.start_time = time.time()
        #path_to_data = f"datasets/bc_dataset_full"

        for itr in range(epochs):
            print("\n\n********** Epoch %i ************"%itr)
            
            # # decide if metrics should be logged
            # if itr % self.params['scalar_log_freq'] == 0:
            #     self.log_metrics = True
            # else:
            #     self.log_metrics = False
            
            # use_batchsize = self.params['batch_size']
            # paths = self.collect_training_trajectories(path_to_data)
            # self.agent.add_to_replay_buffer(paths)
            all_logs = self.train_agent()
    
    def train_agent(self):
        """
        Samples a batch of trajectories and updates the agent with the batch
        """
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_batches']):
            # sample some data from the data buffer
            batch_dict = self.agent.replay_buffer.sample_random_data(self.params['train_batch_size'])
            state_batch = batch_dict['state']
            action_batch = batch_dict['action']
            train_log = self.agent.train(ptu.from_numpy(state_batch), ptu.from_numpy(action_batch))
            all_logs.append(train_log)
            
            print(f"Batch {train_step}: Training Loss = {train_log['Training Loss']}")
        return all_logs
        
    