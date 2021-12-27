import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

'''
makes the collision avoidance and goal navigation two seperate modules
'''
class ActorNetwork(keras.Model):
    '''
    nav_dims_ratio defines what proportions of fc1_dims and fc2_dims will be used to train
        the navigation module
    '''
    def __init__(self, n_goals, n_actions, fc1_dims=256, fc2_dims=256, nav_dims_ratio=0.125):
        super(ActorNetwork, self).__init__()
        nav_fc1_dims = round(fc1_dims*nav_dims_ratio)
        nav_fc2_dims = round(fc2_dims*nav_dims_ratio)
        assert 0 < nav_fc1_dims < fc1_dims
        assert 0 < nav_fc2_dims < fc2_dims
        grid_fc1_dims = fc1_dims - nav_fc1_dims
        grid_fc2_dims = fc2_dims - nav_fc2_dims

        self.n_goals = n_goals
        self.grid_fc1 = Dense(grid_fc1_dims, activation='relu')
        self.grid_fc2 = Dense(grid_fc2_dims, activation='relu')
        self.nav_fc1 = Dense(nav_fc1_dims, activation='relu')
        self.nav_fc2 = Dense(nav_fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        grid_fc1_value = self.grid_fc1(state[:,:-self.n_goals*2])
        grid_fc2_value = self.grid_fc2(grid_fc1_value)

        nav_fc1_value = self.nav_fc1(state[:,-self.n_goals*2:])
        nav_fc2_value = self.nav_fc2(nav_fc1_value)

        concat_value = tf.concat([grid_fc2_value, nav_fc2_value], axis=1)
        pi = self.pi(concat_value)

        return pi

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=1024):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(n_actions, activation=None)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        q = self.q(value)
        return q
