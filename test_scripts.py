import pickle
from MARL_env import MARLNavEnv, individual_action_texts
from MARL_PolicyGradient import PGAgent

def test_env():
    env = MARLNavEnv()
    obs_grids = env.reset()
    env.render(render_obs_grids=True)
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
        env.render(render_obs_grids=True)
        print('\n\n')

    obs_grids = env.reset()
    env.render(render_obs_grids=True)
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
        env.render(render_obs_grids=True)
        print('\n\n')

    obs_grids = env.reset()
    env.render(render_obs_grids=True)
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
        env.render(render_obs_grids=True)
        print('\n\n')

def demo_PGAgent(version='', num_resets=1):
    agent = pickle.load(open('PGAgent{}.p'.format(version), 'rb'))
    env = MARLNavEnv()
    for i in range(num_resets):
        print('---------------------------------------------------')
        print('-------------------MAP_RESET_{}---------------------'.format(i))
        print('---------------------------------------------------')
        step_counter = 0
        done = False
        score = 0
        collision_counter = 0
        if i == 0:
            obs_grids = env.reset()
        else:
            obs_grids = env.reset(randomize=True)
        while not done:
            print('-------------------STEP_{}---------------------'.format(step_counter))
            actions = agent.choose_action(obs_grids)
            obs_grids, reward, done, info = env.step(actions)
            score += reward
            if any(info['in_collision']):
                collision_counter += 1
            env.render()
            step_counter += 1
            print('\n\n')
        print('---------------------------------------------------')
        print('score:{}'.format(score))
        print('num_collisions:{}'.format(collision_counter))
        print('goals_reached:{}'.format([True if goal_loc[0] is None else False for goal_loc in env.goal_locations]))
        print('---------------------------------------------------')

if __name__ == '__main__':
    demo_PGAgent(num_resets=5)
    # test_env()
