import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np

from infrastructure import pytorch_util as ptu

def create_q_network(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )

class IQLCritic():

    def __init__(self, hparams, **kwargs):
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.learning_rate = hparams['learning_rate']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']
        
        # given observation and action, output value
        self.q_net = create_q_network(self.ob_dim + self.ac_dim, 1)
        self.q_net_target = create_q_network(self.ob_dim + self.ac_dim, 1)

        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # TODO define value function
        # HINT: see Q_net definition above and optimizer below
        # HINT: Define using same hparams as Q_net, but adjust output dimensions
        self.v_net = create_q_network(self.ob_dim, 1)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        
        self.iql_expectile = hparams['iql_expectile']

    def expectile_loss(self, diff):
        # TODO: Implement the expectile loss given the difference between q and v
        # HINT: self.iql_expectile provides the \tau value as described 
        # in the problem statement.
        return torch.pow(diff, 2)*torch.abs(torch.where(diff <= 0, self.iql_expectile - 1, self.iql_expectile))

    def update_v(self, ob_no, ac_na):
        """
        Update value function using expectile loss
        """
        #ob_no = ptu.from_numpy(ob_no)
        #ac_na = ptu.from_numpy(ac_na).to(torch.long)

        ### YOUR CODE HERE ###
        q_input = torch.concatenate((ob_no, ac_na),axis=1)     # shape(batch_size, obs_dim+ac_dim)
        target_q_value = self.q_net_target(q_input)    # shape (batch_size, 1)
        v_func = self.v_net(ob_no)
        value_loss = self.expectile_loss(target_q_value - v_func).mean()
        ### YOUR CODE HERE ###

        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}



    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Use target v network to train Q
        """
        # ob_no = ptu.from_numpy(ob_no)
        # ac_na = ptu.from_numpy(ac_na).to(torch.long)
        # next_ob_no = ptu.from_numpy(next_ob_no)
        # reward_n = ptu.from_numpy(reward_n)
        # terminal_n = ptu.from_numpy(terminal_n)
        
        # TODO: Compute loss for updating Q_net parameters
        ### YOUR CODE HERE ###
        q_input = torch.concat((ob_no, ac_na),axis=1)     # shape(batch_size, obs_dim+ac_dim)
        q_value = self.q_net_target(q_input).squeeze(1)                # shape (batch_size, 1)
        v_func = self.v_net(next_ob_no).squeeze(1).detach()
        # Want to keep v_func values if not terminal
        target = reward_n + self.gamma*v_func*(1-terminal_n)
        loss = nn.functional.mse_loss(q_value, target)
        ### YOUR CODE HERE ###
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        
        #self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)
