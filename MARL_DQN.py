import pickle
import copy
import tensorflow as tf
from MARL_networks import CriticNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from MARL_env import MARLNavEnv
from MARL_utils import plotLearning
# from replay_buffer import ReplayBuffer
from priority_buffer import Memory
from itertools import product


'''
implements Deep Q-learning algorithm

train a global DQN that takes in the entire grid and estimates utility values
    for all possible full action assignments
    - prioritized experience replay
'''
class DQNAgent:
    '''
    alpha:        optimizer learning rate for network
    gamma:        reward discount
    epsilon:      exploration rate
    '''
    def __init__(self, n_bots, n_actions, grid_size, alpha=0.003, gamma=0.99, epsilon=1.0,
                    mem_size=1000000, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = 1e-3
        self.eps_min = 0.01
        self.n_bots = n_bots
        self.n_actions = n_actions
        self.action_space = self._get_action_space()
        self.batch_size = batch_size
        # self.memory = ReplayBuffer(mem_size, grid_size)
        self.memory = Memory(mem_size, grid_size)
        self.q_eval = CriticNetwork(n_actions=n_actions**n_bots)
        self.q_eval.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')

    @classmethod
    def from_pickle(cls, version=''):
        try:
            return pickle.load(open('DQNAgent{}.p'.format(version), 'rb'))
        except FileNotFoundError:
            print('Cannot find DQNAgent version, exiting...')
            quit()
        except _ as err:
            print(err)
            quit()

    '''
    chooses a random action assignment if epsilon check passes
        otherwise chooses the action assignment that has the highest estimated utility value
        also returns action space index to be stored in replay buffer
    '''
    def choose_actions(self, flat_grid):
        if np.random.random() < self.epsilon:
            index = np.random.choice(self.n_actions**self.n_bots)
        else:
            # tensor_grid = tf.convert_to_tensor([flat_grid], dtype=tf.float32)
            evals = self.q_eval(flat_grid.reshape(1,-1))
            index = np.argmax(evals)

        actions = self.action_space[index]

        return actions, index

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
        return np.array(list(product([i for i in range(self.n_actions)], repeat=self.n_bots)))

    def learn(self):
        # if self.memory.mem_cntr < self.batch_size:
        #     return
        if self.memory.tree.data_counter < self.batch_size:
            return

        states, actions, rewards, states_, dones, tree_idx, ISWeights = \
                self.memory.sample_buffer(self.batch_size)

        evals = self.q_eval(states)
        evals_ = self.q_eval(states_)

        target_evals = np.copy(evals)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        target_evals[batch_index, actions] = rewards + self.gamma * np.max(evals_, axis=1)*dones

        self.q_eval.train_on_batch(states, target_evals, ISWeights)

        # only 1 non-zero TD error for each sample
        abs_errors = np.max(abs(target_evals - evals), axis=1)
        self.memory.batch_update(tree_idx, abs_errors)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

if __name__ == '__main__':
    env = MARLNavEnv(map_filename='minimap.txt', max_steps=10)
    resume_previous = input('Resume from a previously trained agent?({blank}/{version_num}/n):')
    if resume_previous == '':
        agent = DQNAgent.from_pickle()
    elif resume_previous == 'n':
        agent = DQNAgent(n_bots=env.nbots, n_actions=5, grid_size=env.nrows*env.ncols)
    else:
        try:
            version_num = int(resume_previous)
            agent = DQNAgent.from_pickle(version=version_num)
        except ValueError:
            print('Unknown input, exiting...')
            quit()

    score_history = []
    num_episodes = 1000
    for i in range(num_episodes):
        try:
            done = False
            score = 0
            env.reset()
            flat_grid = env.filled_grid().astype(np.float32).ravel()
            while not done:
                actions, index = agent.choose_actions(flat_grid)
                _, reward, done, _ = env.step(actions)
                flat_grid_ = env.filled_grid().astype(np.float32).ravel()
                agent.memory.store_transition(flat_grid, index, reward, flat_grid_, done)
                flat_grid = flat_grid_
                score += reward
                agent.learn()
            score_history.append(score)
        except KeyboardInterrupt:
            filename = 'marl_nav_dqn{}.png'.format(i)
            plotLearning(scores=score_history, filename=filename, window=100)
            pickle.dump(agent, open('DQNAgent{}.p'.format(i), 'wb'))
            resume = input('DQNAgent saved! Continue training?(y/n):')
            if resume == 'y':
                continue
            elif resume == 'n':
                quit()
            else:
                print('Unknown input, resuming...')
                continue

        avg_score = np.mean(score_history[-100:])
        print('episode: ', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score)

    filename = 'marl_nav_dqn.png'
    plotLearning(scores=score_history, filename=filename, window=100)
    pickle.dump(agent, open('DQNAgent.p', 'wb'))
