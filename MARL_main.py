

class Agent:
    '''
    alpha:        optimizer learning rate for actor network
    beta:         optimizer learning rate for critic network
    gamma:        reward discount
    tau:          weighting factor for transfering current actor/critic to
                  target actor/critic networks
    batch_size:   batch size for sampling replay buffer
                  (i.e. state/action/reward memories)
    '''
    def __init__(self, alpha=0.003, beta=0.002, gamma=0.99,
                tau=0.005, n_actions=5, n_bots=3, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.n_bots = n_bots

        # replay buffer parameters
        self.batch_size = batch_size
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.terminal_memory = []
        self.memory_size = 0

        # actors and critic
        self.actors = [ActorNetwork(n_actions=n_actions) for i in range(n_bots)]
        for a in self.actors:
            a.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=beta))

        # stablizing target actors and target critic
        self.target_actors = [ActorNetwork(n_actions=n_actions) for i in range(n_bots)]
        for a in self.target_actors:
            a.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic = CriticNetwork()
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # initialize target networks to be the same as current networks
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for a, ta in zip(self.actors, self.target_actors):
            new_target_weights = []
            for aw, taw in zip(a.weights, ta.weights):
                new_target_weights.append(aw * tau + taw[i]*(1-tau))
            ta.set_weights(new_target_weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def choose_action(self, obs_grids):
        assert len(obs_grids) == self.n_bots

        actions = []
        for idx, o in enumerate(obs_grids):
            # Keras Dense layer expects 2D input
            state = tf.convert_to_tensor([np.ravel(o)], dtype=tf.float32)
            probs = self.actors[idx](state)
            action_probs = tfp.distributions.Categorical(probs=probs)
            action = action_probs.sample()
            actions.append(action)

        return np.array(actions)

    def store_transition(self, state, new_state, action, reward, done):
        self.state_memory.append(state)
        self.new_state_memory.append(new_state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)
        self.memory_size += 1

    def sample_buffer(self, batch_size):
        idx = np.random.choice(self.memory_size, batch_size, replace=False)

        states = np.array([self.state_memory[i] for i in idx])
        new_states = np.array([self.new_state_memory[i] for i in idx])
        actions = np.array([self.action_memory[i] for i in idx])
        rewards = np.array([self.reward_memory[i] for i in idx])
        dones = np.array([self.terminal_memory[i] for i in idx])

        return states, new_states, actions, rewards, dones

    def learn(self):
        if self.memory_size < self.batch_size:
            return

        states, new_states, actions, rewards, dones = self.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

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

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
