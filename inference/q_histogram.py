'''
Histogram of Q-values class
'''

class QDist():
    """
    Defines a class to encode the distribution of q-values
    """
    def __init__(self, qmodel, replay_buffer):
        self.q_model = qmodel
        self.replay_buff = replay_buffer
    
    ''' 
    Simulate actions from replay buffer by generating close values 
    for each dimension 
    
    action: 8-dim np array of (x,y,sin_orientation, cosine_orientation, 
                            sin_direction, cosine_direction, speed, acc)
    '''
    def simulate_actions(self, action):
        return None
    
    
    '''
    Creates histogram of percentile of true action's q-value,
    with respect to simulated actions 
    '''
    def generate_histogram(self):
        return None
    