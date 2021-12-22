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
    beta:         weight of entropy loss when calculating loss
    gamma:        reward discount
    '''
    def __init__(self, alpha=0.001, beta=0.1, gamma=0.99, n_bots=2, n_actions=5):
        self.beta = beta
        self.gamma = gamma
        self.n_bots = n_bots
        self.n_actions = n_actions
        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        # since all robots are homogenous we can train a single network for all of them
        self.policy = ActorNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=alpha))

    '''
    each bot can only observe its observation grid
        consisted of its surroundings upto obs_range and goals
    '''
    def choose_action(self, obs_states):
        assert len(obs_states) == self.n_bots

        actions = []
        for o in obs_states:
            # Keras Dense layer expects 2D input
            state = tf.convert_to_tensor([o], dtype=tf.float32)
            probs = self.policy(state)
            action_probs = tfp.distributions.Categorical(probs=probs)
            action = action_probs.sample()[0]
            if action == 5:
                # debug when self.policy.trainable_variables become all nan
                import pdb; pdb.set_trace()
            actions.append(action)

        return np.array(actions)

    def store_transition(self, obs_states, actions, reward):
        self.obs_memory.append(obs_states)
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

        for bot_index in range(self.n_bots):
            # zip returns an iterator that gets erased after iterating (new in Python3)
            zipped_rewards_obs = zip(G, self.obs_memory)
            with tf.GradientTape() as tape:
                loss = 0
                for step_index, (g, obs_states) in enumerate(zipped_rewards_obs):
                    state = tf.convert_to_tensor([obs_states[bot_index]], dtype=tf.float32)
                    probs = self.policy(state)
                    entropy_loss = -tf.reduce_sum(tf.math.multiply(probs, tf.math.log(probs)))
                    action_probs = tfp.distributions.Categorical(probs=probs)
                    log_prob = action_probs.log_prob(actions[step_index][bot_index])
                    loss += -(1-self.beta)*g*tf.squeeze(log_prob) - self.beta*entropy_loss
                gradient = tape.gradient(loss, self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []

if __name__ == '__main__':
    agent = PGAgent(n_bots = n_bots)

    env = MARLNavEnv(obs_range=1)
    score_history = []

    num_episodes = 1000

    for i in range(num_episodes):
        try:
            done = False
            score = 0
            obs_states = env.reset(randomize=True)
            while not done:
                actions = agent.choose_action(obs_states)
                new_obs_states, reward, done, info = env.step(actions)
                agent.store_transition(obs_states, actions, reward)
                obs_states = new_obs_states
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
