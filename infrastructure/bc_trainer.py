"""
Defines a trainer which updates a behavior cloning agent
"""

import torch
import numpy as np
import pandas as pd
import pickle

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
        ob_dim = 391
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim
        self.params['agent_params']['train_split'] = self.params['train_split']
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
            
            # eval
            if self.params['train_split'] != 1:
                train_logs = self.train_agent(mode='train')
                eval_logs = self.eval_agent(mode='val')
            else:
                train_logs = self.train_agent(mode=None)
                eval_logs = None
            
        if self.params['save']:
            model_path = self.params["logdir"] + "/model"
            print(f'Saving model at path {model_path}...')
            self.agent.actor.save(model_path)
            
            with open(self.params["logdir"] + "/train_logs.pkl", "wb") as fp:   #Pickling
                pickle.dump(train_logs, fp)
            if eval_logs != None:
                with open(self.params["logdir"] + "/eval_logs.pkl", "wb") as fp:   #Pickling
                    pickle.dump(eval_logs, fp)
    
    def train_agent(self, mode='train'):
        """
        Samples a batch of trajectories and updates the agent with the batch
        mode: {train, val, test}
        """
        print(f'\n Mode={mode} - training agent using sampled data from replay buffer...')
        all_logs = []
        running_loss = 0
        print_every = 1000
        for train_step in range(self.params['num_batches']):
            # sample some data from the data buffer
            batch_dict = self.agent.replay_buffer.sample_random_data(self.params['train_batch_size'], data_dict_to_use=mode)
            state_batch = batch_dict['state']
            action_batch = batch_dict['action']
            train_log = self.agent.train(ptu.from_numpy(state_batch), ptu.from_numpy(action_batch))
            running_loss += train_log['Training Loss']
            all_logs.append(train_log)
            
            if train_step % print_every == (print_every-1):    # print every print_every mini-batches
                print(f"{mode} running loss = {running_loss / print_every:.3f}")
                running_loss = 0.0
        return all_logs
    
    def eval_agent(self, mode='val'):
        """
        Samples a batch of trajectories and updates the agent with the batch
        mode: {train, val, test}
        """
        print(f'\n Mode={mode} - training agent using sampled data from replay buffer...')
        val_loss = self.agent.eval(mode)
        print(f"{mode} loss = {val_loss:.3f}")
        return val_loss
        
        
    