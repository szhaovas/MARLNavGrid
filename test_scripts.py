import pickle
from MARL_env import MARLNavEnv, individual_action_texts
from MARL_PolicyGradient import PGAgent

def test_env():
    env = MARLNavEnv()
    obs_grids = env.reset()
    env.render()
    MARLNavEnv.render_obs_grids(obs_grids)
    print('\n\n------------------------------------------')
    print('RANDOM ACTIONS:')
    for i in range(10):
        print('-------------------STEP_{}---------------------'.format(i))
        actions = env.sample_actions()
        new_obs_grids, reward, done, info = env.step(actions)
        print('Action executed at step {}:{}'.format(i, list(map(lambda i: individual_action_texts[i], actions))))
        print('reward:{}; done:{}'.format(reward, done))
        print('in_collision:{}'.format(info['in_collision']))
        print('goals_reached:{}'.format(info['goals_reached']))
        env.render()
        MARLNavEnv.render_obs_grids(new_obs_grids)
        print('\n\n')

    obs_grids = env.reset()
    env.render()
    MARLNavEnv.render_obs_grids(obs_grids)
    goal_sequence = [[3, 2] for i in range(9)]
    goal_sequence.append([0, 1])
    print('\n\n------------------------------------------')
    print('GOAL SEQUENCE:')
    for idx, actions in enumerate(goal_sequence):
        print('-------------------STEP_{}---------------------'.format(idx))
        new_obs_grids, reward, done, info = env.step(actions)
        print('Action executed at step {}:{}'.format(idx, list(map(lambda i: individual_action_texts[i], actions))))
        print('reward:{}; done:{}'.format(reward, done))
        print('in_collision:{}'.format(info['in_collision']))
        print('goals_reached:{}'.format(info['goals_reached']))
        env.render()
        MARLNavEnv.render_obs_grids(new_obs_grids)
        print('\n\n')

    obs_grids = env.reset()
    env.render()
    MARLNavEnv.render_obs_grids(obs_grids)
    collision_sequence = [[3, 4] for i in range(9)] + [[0, 1] for i in range(6)]
    print('\n\n------------------------------------------')
    print('COLLISION SEQUENCE:')
    for idx, actions in enumerate(collision_sequence):
        print('-------------------STEP_{}---------------------'.format(idx))
        new_obs_grids, reward, done, info = env.step(actions)
        print('Action executed at step {}:{}'.format(idx, list(map(lambda i: individual_action_texts[i], actions))))
        print('reward:{}; done:{}'.format(reward, done))
        print('in_collision:{}'.format(info['in_collision']))
        print('goals_reached:{}'.format(info['goals_reached']))
        env.render()
        MARLNavEnv.render_obs_grids(new_obs_grids)
        print('\n\n')

def demo_PGAgent(version=''):
    agent = pickle.load(open('PGAgent{}.p'.format(version), 'rb'))
    env = MARLNavEnv()
    env.render()
    step_counter = 0
    done = False
    obs_grids = env.reset()
    while not done:
        print('-------------------STEP_{}---------------------'.format(step_counter))
        actions = agent.choose_action(obs_grids)
        obs_grids, _, done, _ = env.step(actions)
        env.render()
        step_counter += 1
        print('\n\n')

if __name__ == '__main__':
    # demo_PGAgent(version=963)
    test_env()
