import pickle
import copy
import tensorflow as tf
import tensorflow.keras as keras
from MARL_networks import CriticNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from MARL_env import MARLNavEnv
from MARL_utils import plotLearning
from replay_buffer import ReplayBuffer
from itertools import product


'''
implements Deep Q-learning algorithm

train a global DQN that takes in the entire grid PLUS a full action assignment for ALL bots
    and returns estimated utility value
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
        self.memory = ReplayBuffer(mem_size, grid_size, n_bots)
        self.q_eval = CriticNetwork()
        self.q_eval.compile(optimizer=Adam(learning_rate=alpha))

    '''
    iterates over each possible full action assignment in action_space, estimates its utility
        and chooses argmax

    if training is True, disable random action
    '''
    def choose_actions(self, flat_grid, training=False):
        if not training and np.random.random() < self.epsilon:
            index = np.random.choice(self.n_actions**self.n_bots)
            actions = self.action_space[index]
            max_eval = None
        else:
            max_eval = float('-inf')
            max_index = 0
            for index, assignment in enumerate(self.action_space):
                eval = self.q_eval([flat_grid], [assignment.astype(np.float32)])
                if eval > max_eval:
                    max_eval = eval
                    max_index = index

            actions = self.action_space[index]

        return actions, max_eval

    '''
    action space:
        - 0: action 0 for bot 0, action 0 for bot 1
        - 1: action 0 for bot 0, action 1 for bot 1
        - 2: action 0 for bot 0, action 2 for bot 1
        - 3: action 0 for bot 0, action 3 for bot 1
        - 4: action 0 for bot 0, action 4 for bot 1
        - 5: action 1 for bot 0, action 0 for bot 1
        - 6: action 1 for bot 0, action 1 for bot 1
        - 7: action 1 for bot 0, action 2 for bot 1
        - 8: action 1 for bot 0, action 3 for bot 1
        - 9: action 1 for bot 0, action 4 for bot 1
        - 10: action 2 for bot 0, action 0 for bot 1
        - etc...
        - i.e. it's just the cartesian product of [0...(n_actions-1)] with itself repeated n_bots times
    '''
    def _get_action_space(self):
        return np.array(list(product([i for i in range(self.n_actions)], repeat=self.n_bots)))

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        with tf.GradientTape() as tape:
            q_nexts = tf.squeeze(self.q_eval(states, actions.astype(np.float32)))
            q_targets = np.zeros(self.batch_size, dtype=np.float32)
            for batch_index, s_ in enumerate(states_):
                _, max_eval = self.choose_actions(s_, training=True)
                q_ = rewards[batch_index] + self.gamma * max_eval * dones[batch_index]
                q_targets[batch_index] = q_

            q_targets = tf.convert_to_tensor(q_targets, dtype=tf.float32)
            loss = keras.losses.MSE(q_nexts, q_targets)

            gradient = tape.gradient(loss, self.q_eval.trainable_variables)
            self.q_eval.optimizer.apply_gradients(zip(
                gradient, self.q_eval.trainable_variables))

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

if __name__ == '__main__':
    env = MARLNavEnv(map_filename='minimap.txt', max_steps=10)
    agent = DQNAgent(n_bots=env.nbots, n_actions=5, grid_size=env.nrows*env.ncols, batch_size=1)

    score_history = []
    num_episodes = 1000
    for i in range(num_episodes):
        try:
            done = False
            score = 0
            env.reset()
            flat_grid = env.filled_grid().astype(np.float32).ravel()
            while not done:
                actions, _ = agent.choose_actions(flat_grid)
                _, reward, done, _ = env.step(actions)
                flat_grid_ = env.filled_grid().astype(np.float32).ravel()
                agent.memory.store_transition(flat_grid, actions, reward, flat_grid_, done)
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
