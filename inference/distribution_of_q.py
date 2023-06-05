''' 
File to generate histogram & plot of true action's q-value, 
compared to simulated actions
'''
from infrastructure.replay_buffer import ReplayBuffer
import os
import time
import argparse

import torch
from q_histogram import QDist

def load_q(path_to_q):
    q_model.load_state_dict(torch.load())
    q_model.eval()
    return q_model

def main():
    parser = argparse.ArgumentParser()
    # Nunmber of simulations
    parser.add_argument('--sim', type=int, default=20)
    parser.add_argument('--qmodel', type=str, default=None)
    # use the training split used for the qmodel, so can use the test data in the replay buffer
    parser.add_argument('q_train_perc', type=float, default=0.7)
    args = parser.parse_args()
    params = vars(args)
    
    # Load q_model
    q_model = load_q(params['qmodel'])
    # Load replay buffer
    replay_buffer = ReplayBuffer(filepath="datasets/bc_dataset_full.pkl", train_percentage=params['q_train_perc'])
    qdist = QDist(q_model, replay_buffer)
    qdist.generate_actions()

if __name__ == "__main__":
    main()