import pickle
import tensorflow as tf
from MARL_networks import ActorNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from MARL_env import MARLNavEnv
from MARL_utils import plotLearning

n_bots = 2

'''
implements REINFORCE algorithm

train a policy network for each bot using shared reward
'''
class PGAgent:
    '''
    alpha:        optimizer learning rate for network
    gamma:        reward discount
    '''
    def __init__(self, alpha=0.003, gamma=0.99, n_bots=2, n_actions=5):
        self.gamma = gamma
        self.n_bots = n_bots
        self.n_actions = n_actions
        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policies = [ActorNetwork(n_actions=n_actions) for i in range(n_bots)]
        for a in self.policies:
            a.compile(optimizer=Adam(learning_rate=alpha))

    '''
    each bot can only observe its observation grid
        consisted of its surroundings upto obs_range and goals
    '''
    def choose_action(self, obs_grids):
        assert len(obs_grids) == self.n_bots

        actions = []
        for pol_index, o in enumerate(obs_grids):
            # Keras Dense layer expects 2D input
            state = tf.convert_to_tensor([np.ravel(o)], dtype=tf.float32)
            probs = self.policies[pol_index](state)
            action_probs = tfp.distributions.Categorical(probs=probs)
            action = action_probs.sample()[0]
            actions.append(action)

        return np.array(actions)

    '''
    each policy network only gets observation grid as input, so memory stores
        obs_grids instead of entire grids
    '''
    def store_transition(self, obs_grids, actions, reward):
        self.obs_memory.append(obs_grids)
        self.action_memory.append(actions)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        for pol_index, p in enumerate(self.policies):
            # zip returns an iterator that gets erased after iterating (new in Python3)
            zipped_rewards_obs = zip(G, self.obs_memory)
            with tf.GradientTape() as tape:
                loss = 0
                for step_index, (g, obs_grids) in enumerate(zipped_rewards_obs):
                    state = tf.convert_to_tensor([np.ravel(obs_grids[pol_index])], dtype=tf.float32)
                    probs = p(state)
                    action_probs = tfp.distributions.Categorical(probs=probs)
                    log_prob = action_probs.log_prob(actions[step_index][pol_index])
                    # import pdb; pdb.set_trace()
                    loss += -g * tf.squeeze(log_prob)
                gradient = tape.gradient(loss, p.trainable_variables)
                p.optimizer.apply_gradients(zip(gradient, p.trainable_variables))

        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []

if __name__ == '__main__':
    agent = PGAgent(n_bots = n_bots)

    env = MARLNavEnv()
    score_history = []

    num_episodes = 1000

    for i in range(num_episodes):
        try:
            done = False
            score = 0
            obs_grids = env.reset()
            while not done:
                actions = agent.choose_action(obs_grids)
                new_obs_grids, reward, done, info = env.step(actions)
                agent.store_transition(obs_grids, actions, reward)
                obs_grids = new_obs_grids
                score += reward
            score_history.append(score)

            agent.learn()
            avg_score = np.mean(score_history[-100:])
            print('episode: ', i,'score: %.1f' % score,
                'average score %.1f' % avg_score)
        except KeyboardInterrupt:
            filename = 'marl_nav{}.png'.format(i)
            plotLearning(score_history, filename=filename, window=100)
            pickle.dump(agent, open('PGAgent{}.p'.format(i), 'wb'))
            quit()

    filename = 'marl_nav.png'
    plotLearning(score_history, filename=filename, window=100)
    pickle.dump(agent, open('PGAgent.p', 'wb'))
