from infrastructure.replay_buffer import ReplayBuffer
from policies.MLP_policy import MLPPolicySL
from policies.MLP_policy import MixturePolicy
from policies.MLP_policy import MLPPolicyCNN
from torch.nn import functional as F
from infrastructure import pytorch_util as ptu

class BCAgent():
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
        #self.actor = MLPPolicySL(
        #self.actor = MixturePolicy(
        self.actor = MLPPolicyCNN(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # replay buffer
        print(f"Creating replaying buffer from {self.agent_params['filepath']}")
        self.replay_buffer = ReplayBuffer(max_size=self.agent_params['max_replay_buffer_size'], filepath=self.agent_params['filepath'], train_percentage=self.agent_params['train_split'])
    
    def train(self, ob_no, ac_na):
        """
        :param ob_no: batch_size x obs_dim batch of observations
        :param ac_na: batch_size x ac_dim batch of actions
        """
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        loss = self.actor.update(ob_no, ac_na)
        return loss
    
    def eval(self, mode):
        if mode == 'val':
            state = self.replay_buffer.val['state']
            action = self.replay_buffer.val['action']
        else:
            state = self.replay_buffer.test['state']
            action = self.replay_buffer.test['action']
            
        return F.mse_loss(ptu.from_numpy(self.actor.get_action(state)), ptu.from_numpy(action))

    def sample(self, batch_size):
        """
        :param batch_size: size of batch to sample from replay buffer
        """
        return self.replay_buffer.sample_random_data(batch_size)