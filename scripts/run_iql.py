"""
Runs implicit q-learning
Hyperparameters for the experiment are defined in main()

export PYTHONPATH=~/Desktop/cs224r/optimal_action_proj/
"""

import os
import time
import argparse
import numpy as np
import torch

from infrastructure.iql_trainer import IQLTrainer
from infrastructure import pytorch_util as ptu
from agents.iql_agent import IQLAgent

def run_iql(params):
    """
    Runs behavior cloning with the specified parameters

    Args:
        params: experiment parameters
    """
        
    # AGENT PARAMS
    agent_params = {
        'n_layers': params['n_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
        'filepath': params['filepath']
    }
        
    params['agent_class'] = IQLAgent
    params['agent_params'] = agent_params
    
    trainer = IQLTrainer(params)
    trainer.run_training_loop(
        epochs=params['epochs']
    )
    

def main():
    
    parser = argparse.ArgumentParser()
    
    # algo params
    parser.add_argument('--iql_expectile', type=float, default=0.8)
    
    # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    # training data collected (in the env) during each iteration
    parser.add_argument('--train_batch_size', type=int, default=128)
    # train split
    parser.add_argument('--train_split', type=float, default=1)
    # eval data collected (in the env) for logging metrics
    parser.add_argument('--num_batches', type=int, default=5000)

    parser.add_argument('--gamma', type=float, default=0.9) # what is usual default
    # depth, of policy to be learned
    parser.add_argument('--n_layers', type=int, default=2)
    # width of each layer, of policy to be learned
    parser.add_argument('--size', type=int, default=64)
    # LR for supervised learning
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    
    parser.add_argument('--filepath', '-f', type=str, default="./datasets/bc_dataset_full.pkl")
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=2430)
    args = parser.parse_args()
    # convert args to dictionary
    params = vars(args)
    
    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../models')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = 'iql_' + time.strftime("%d-%m-%Y_%H-%M")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    run_iql(params)


if __name__ == "__main__":
    main()
