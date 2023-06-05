"""
Defines a trainer which updates a behavior cloning agent
"""

import torch
import numpy as np
import pandas as pd
import pickle

from infrastructure import pytorch_util as ptu

class IQLTrainer:
    """
    A class which defines the training algorithm for the agent. Handles
    sampling data, updating the agent, and logging the results.

    ...

    Attributes
    ----------
    agent : IQL Agent
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
        self.params['agent_params']['gamma'] = self.params['gamma']
        self.params['agent_params']['iql_expectile'] = self.params['iql_expectile']
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
        train_logs = []
        avg_logs = []
        eval_logs = []
        for itr in range(epochs):
            print("\n\n********** Epoch %i ************"%itr)
            
            if self.params['train_split'] != 1:
                all_log, avg_log = self.train_agent(mode='train')
                train_logs.append(all_log)
                avg_logs.append(avg_log)
                
                eval_log = self.eval_agent(mode='val')
                eval_logs.append(eval_log)
            else:
                all_log, avg_log = self.train_agent(mode=None)
                train_logs.append(all_log)
            
        if self.params['save']:
            model_path = self.params["logdir"] + "/model"
            print(f'Saving model at path {model_path}...')
            self.agent.awac_actor.save(model_path)
            
            with open(self.params["logdir"] + "/train_logs.pkl", "wb") as fp:   #Pickling
                pickle.dump(train_logs, fp)
            with open(self.params["logdir"] + "/train_avg_logs.pkl", "wb") as fp:   #Pickling
                pickle.dump(avg_logs, fp)
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
        avg_logs = []
        print_every = 1000
        running_actor_loss = 0
        running_q_loss = 0
        running_v_loss = 0
        for train_step in range(self.params['num_batches']):
            # sample some data from the data buffer
            batch_dict = self.agent.replay_buffer.sample_random_data(self.params['train_batch_size'], data_dict_to_use=mode)
            train_log = self.agent.train(batch_dict)
            
            running_actor_loss += train_log['actor_loss']
            running_q_loss += train_log['critic_q_loss']
            running_v_loss += train_log['critic_v_loss']
            
            all_logs.append(train_log)
            if train_step % print_every == (print_every-1):    # print every print_every mini-batches
                print(f"Outputing log for batch {train_step} out of {self.params['num_batches']}:")
                print(f"critic v loss: {running_v_loss/print_every}")
                print(f"critic q loss: {running_q_loss/print_every}")
                print(f"actor loss: {running_actor_loss/print_every}")
                avg_log = {"critic v loss": running_v_loss/print_every, "critic q loss":running_q_loss/print_every , "actor loss":running_actor_loss/print_every }
                avg_logs.append(avg_log)
                running_actor_loss = 0
                running_q_loss = 0
                running_v_loss = 0
        return all_logs, avg_log
        
    def eval_agent(self, mode='val'):
        """
        Samples a batch of trajectories and updates the agent with the batch
        mode: {train, val, test}
        """
        print(f'\n Mode={mode} - training agent using sampled data from replay buffer...')
        val_loss = self.agent.eval(mode)
        print(f"{mode} loss = {val_loss:.3f}")
        return [{"actor loss": val_loss}]
        
        
    