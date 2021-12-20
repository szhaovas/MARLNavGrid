import pickle
import copy
import tensorflow as tf
from MARL_networks import CriticNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from MARL_env import MARLNavEnv
from MARL_utils import plotLearning
from replay_buffer import ReplayBuffer
from itertools import product

n_bots = 2
mem_size=1000000
grid_size = (100,)

'''
implements Deep Q-learning algorithm

train a global DQN that takes in the entire grid and returns actions for each bot
'''
class DQNAgent:
    '''
    alpha:        optimizer learning rate for network
    gamma:        reward discount
    epsilon:      exploration rate
    '''
    def __init__(self, alpha=0.003, gamma=0.99, epsilon=1.0, n_bots=2, n_actions=5, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = 1e-3
        self.eps_min = 0.01
        self.n_bots = n_bots
        self.n_actions = n_actions
        self.action_space = self._get_action_space()
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, grid_size)
        self.q_eval = CriticNetwork(discrete=True)
        self.q_eval.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')

    '''
    global DQN observes the entire grid
    '''
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            flat_observation = observation.ravel()
            state = np.array([flat_observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    '''
    output of DQN is of dimension n_actions ** n_bots
        each output node represents a possible assignment of actions to all bots

        The mapping of node to action assignments goes as follows:
        - node 0: action 0 for bot 0, action 0 for bot 1
        - node 1: action 0 for bot 0, action 1 for bot 1
        - node 2: action 0 for bot 0, action 2 for bot 1
        - node 3: action 0 for bot 0, action 3 for bot 1
        - node 4: action 0 for bot 0, action 4 for bot 1
        - node 5: action 1 for bot 0, action 0 for bot 1
        - node 6: action 1 for bot 0, action 1 for bot 1
        - node 7: action 1 for bot 0, action 2 for bot 1
        - node 8: action 1 for bot 0, action 3 for bot 1
        - node 9: action 1 for bot 0, action 4 for bot 1
        - node 10: action 2 for bot 0, action 0 for bot 1
        - etc...
        - i.e. it's just the cartesian product of [0...(n_actions-1)] with itself repeated n_bots times
    '''
    def _get_action_space(self):
        return list(product([i for i in range(self.n_actions)], repeat=self.n_bots))

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

if __name__ == '__main__':
    agent = DQNAgent(n_bots=n_bots)

    env = MARLNavEnv()
    score_history = []

    num_episodes = 1000

    for i in range(num_episodes):
        try:
            done = False
            score = 0
            env.reset()
            observation = copy.deepcopy(env.grid)
            while not done:
                action = agent.choose_action(observation)
                decoded_action = agent.action_space[action]
                _, reward, done, info = env.step(decoded_action)
                observation_ = copy.deepcopy(env.grid)
                score += reward
                flat_observation = observation.ravel()
                flat_observation_ = observation_.ravel()
                agent.memory.store_transition(flat_observation, action, reward, flat_observation_, done)
                observation = observation_
                agent.learn()
            score_history.append(score)
        except KeyboardInterrupt:
            filename = 'marl_nav_dqn{}.png'.format(i)
            plotLearning(scores=score_history, filename=filename, window=100)
            pickle.dump(agent, open('DQNAgent{}.p'.format(i), 'wb'))
            quit()

        avg_score = np.mean(score_history[-100:])
        print('episode: ', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score)

    filename = 'marl_nav_dqn.png'
    plotLearning(scores=score_history, filename=filename, window=100)
    pickle.dump(agent, open('DQNAgent.p', 'wb'))
