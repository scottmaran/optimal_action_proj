from infrastructure.replay_buffer import ReplayBuffer
from policies.MLP_policy import MLPPolicyAWAC
from critics.iql_critic import IQLCritic
from infrastructure import pytorch_util as ptu
from torch.nn import functional as F

import numpy as np
import torch
from collections import namedtuple

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)

class IQLAgent():
    """
    Attributes
    ----------
    actor : MLPPolicySL
        An MLP that outputs an agent's actions given its observations
    replay_buffer: ReplayBuffer
        A replay buffer which stores collected trajectories

    Methods
    -------
    train:
        Calls the actor update function
    add_to_replay_buffer:
        Updates a the replay buffer with new paths
    sample
        Samples a batch of trajectories from the replay buffer
    """
    def __init__(self, agent_params):
        self.agent_params = agent_params
        # actor/policy
        self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['learning_rate']
        )
        
        self.num_param_updates = 0
        self.target_update_freq = 1000
        
        self.agent_params['grad_norm_clipping'] = 10
        #self.optimizer_spec = 
        self.exploitation_critic = IQLCritic(agent_params)

        # replay buffer
        print(f"Creating replaying buffer from {self.agent_params['filepath']}")
        self.replay_buffer = ReplayBuffer(max_size=self.agent_params['max_replay_buffer_size'], filepath=self.agent_params['filepath'], train_percentage=self.agent_params['train_split'])
    
    def estimate_advantage(self, ob_no, ac_na):
        # TODO: Estimate the advantage function
        # Advantage := Q_hat(s,a) - V_hat(s)
        q = self.exploitation_critic.q_net_target(torch.concatenate((ob_no, ac_na),axis=1))
        v = self.exploitation_critic.v_net(ob_no)
        return q-v
    
    def train(self, batch_dict):
        """
        :param ob_no: batch_size x obs_dim batch of observations
        :param ac_na: batch_size x ac_dim batch of actions
        """
        log = {}
        
        ob_no = ptu.from_numpy(batch_dict['state'])
        ac_na = ptu.from_numpy(batch_dict['action'])
        next_ob_no = ptu.from_numpy(batch_dict['next_state'])
        re_n = ptu.from_numpy(batch_dict['reward'])
        terminal_n = torch.zeros(re_n.shape)
        #terminal_n = ptu.from_numpy(batch_dict['terminal'])
        
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        exploitation_critic_loss = self.exploitation_critic.update_v(ob_no, ac_na)
        # add to dictionary
        exploitation_critic_loss.update(self.exploitation_critic.update_q(ob_no, ac_na, next_ob_no, re_n, terminal_n))
        
        # update actor
        advantage = self.estimate_advantage(ob_no, ac_na).detach()
        actor_loss = self.awac_actor.update(observations=ob_no, actions=ac_na, adv_n=advantage)
        
        log['critic_v_loss'] = exploitation_critic_loss['Training Q Loss']
        log['critic_q_loss'] = exploitation_critic_loss['Training V Loss']
        log['actor_loss'] = actor_loss
        
        if self.num_param_updates % self.target_update_freq == 0:
            self.exploitation_critic.update_target_network()
        
        self.num_param_updates += 1
        
        return log
    
    def eval(self, mode):
        if mode == 'val':
            state = ptu.from_numpy(self.replay_buffer.val['state'])
            action = ptu.from_numpy(self.replay_buffer.val['action'])
        else:
            state = ptu.from_numpy(self.replay_buffer.test['state'])
            action = ptu.from_numpy(self.replay_buffer.test['action'])
        
        advantage = self.estimate_advantage(state, action).detach()
        eval_actor_loss = self.awac_actor.eval(state, action, advantage)
        return eval_actor_loss

    def sample(self, batch_size):
        """
        :param batch_size: size of batch to sample from replay buffer
        """
        return self.replay_buffer.sample_random_data(batch_size)