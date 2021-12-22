import numpy as np
import copy
from operator import add
from random import randint
import matplotlib.pyplot as plt


individual_action_options = [
    # right
    (0, 1),
    # left
    (0, -1),
    # up
    (-1, 0),
    # down
    (1, 0),
    # stay
    (0, 0)
]

individual_action_texts = [
    'right',
    'left',
    'up',
    'down',
    'stay'
]


class MARLNavEnv:
    def __init__(self, map_filename="map1.txt", obs_range=1, max_steps=100):
        with open(map_filename, 'r') as f:
            self.obs_range = obs_range
            self.max_steps = max_steps
            self.step_counter = 0

            self.nrows, self.ncols = map(int, f.readline()[:-1].split())

            self.nbots = int(f.readline()[:-1])
            self._start_bot_locations = [tuple(map(int, f.readline()[:-1].split())) for i in range(self.nbots)]
            self.bot_locations = copy.deepcopy(self._start_bot_locations)

            self.ngoals = int(f.readline()[:-1])
            self._start_goal_locations = [tuple(map(int, f.readline()[:-1].split())) for i in range(self.ngoals)]
            self.goal_locations = copy.deepcopy(self._start_goal_locations)

            '''
            grid only shows obstacles when training
                only includes robots and goals when rendering
            '''
            map_list = [None] * self.nrows
            for i in range(self.nrows):
                map_list[i] = list(map(int, f.readline().rstrip('\n').split()))
            self.grid = np.array(map_list)


    def sample_actions(self):
        actions = [np.random.choice(len(individual_action_options)) for i in range(self.nbots)]
        return np.array(actions)


    '''
    observation grid:
        - one for each bot
        - dimension is (self.obs_range*2+1, self.obs_range*2+1)
        - from the world grid, slice the region surrounding the current bot up to distance self.obs_range,
            use this slice to fill the observation grid
        - current bot always at the center
        - if robot is on edge and thus the slice is of a smaller size than (self.obs_range*2+1, self.obs_range*2+1),
            fill the rest with obstacles(2) to represent edges
    '''
    def create_observation_grids(self):
        obs_grids = [None] * self.nbots
        for i in range(self.nbots):
            curr_bot_r, curr_bot_c = self.bot_locations[i]
            row_range_low = max((curr_bot_r-self.obs_range),0)
            row_range_high = min((curr_bot_r+self.obs_range+1),self.nrows)
            col_range_low = max((curr_bot_c-self.obs_range),0)
            col_range_high = min((curr_bot_c+self.obs_range+1),self.ncols)

            # prefill with obstacles
            grid = np.ones((self.obs_range*2+1, self.obs_range*2+1))*2

            # overwrite with slice
            grid[
                row_range_low-curr_bot_r+self.obs_range : row_range_high-curr_bot_r+self.obs_range,
                col_range_low-curr_bot_c+self.obs_range : col_range_high-curr_bot_c+self.obs_range
            ] = self.grid[
                    row_range_low : row_range_high,
                    col_range_low : col_range_high
                ]

            obs_grids[i] = grid

        return obs_grids


    def get_rel_goal_positions(self, bot_index):
        result = []
        bot_r, bot_c = self.bot_locations[bot_index]
        for (goal_r, goal_c) in self.goal_locations:
            if goal_r is None:
                result.append((0, 0))
            else:
                row_diff = goal_r-bot_r
                col_diff = goal_c-bot_c
                result.append((row_diff/self.nrows, col_diff/self.ncols))
        return result


    def get_obs_states(self):
        result = []
        obs_grids = self.create_observation_grids()
        for bot_index, grid in enumerate(obs_grids):
            flat_grid = grid.flatten()
            rel_goal_positions = self.get_rel_goal_positions(bot_index)
            full_state = np.append(flat_grid, rel_goal_positions)
            result.append(full_state)
        return result


    def step(self, actions):
        assert len(actions) == self.nbots
        self.step_counter += 1

        # calculate new locations after action
        new_locations = list(map(lambda t1, i2: tuple(map(add, t1, individual_action_options[i2])), self.bot_locations, actions))

        in_collision = [False] * self.nbots
        for i, (r, c) in enumerate(new_locations):
            # make sure bots don't go out of bounds of grid
            if (not (0 <= r <= (self.nrows-1))) or (not (0 <= c <= (self.ncols-1))):
                new_locations[i] = self.bot_locations[i]
                in_collision[i] = True

            # make sure bots don't move onto obstacles
            if self.grid[new_locations[i]] == 2:
                new_locations[i] = self.bot_locations[i]
                in_collision[i] = True

            # make sure bots don't move onto each other
            for j, other_loc in enumerate(new_locations):
                if i == j:
                    continue
                if new_locations[i] == other_loc:
                    new_locations[i] = self.bot_locations[i]
                    in_collision[i] = True
                    new_locations[j] = self.bot_locations[j]
                    in_collision[j] = True

            # make sure bots don't move through each other
            for j, other_old_loc in enumerate(self.bot_locations):
                loc_diff = list(map(lambda a,b: a-b, self.bot_locations[i], other_old_loc))
                distance = abs(loc_diff[0]) + abs(loc_diff[1])
                if i == j:
                    continue
                if distance == 1 and (actions[i], actions[j]) in {(0,1),(1,0),(2,3),(3,2)}:
                    new_locations[i] = self.bot_locations[i]
                    in_collision[i] = True
                    new_locations[j] = self.bot_locations[j]
                    in_collision[j] = True

        self.bot_locations = new_locations

        # check if goals have been reached
        goals_reached = [False] * self.ngoals
        for (goal_index, goal_loc) in enumerate(self.goal_locations):
            for bot_loc in new_locations:
                if goal_loc == bot_loc:
                    self.goal_locations[goal_index] = (None, None)
            if self.goal_locations[goal_index] == (None, None):
                goals_reached[goal_index] = True

        # observation grids
        obs_states = self.get_obs_states()

        # rewards
        done = False
        if self.step_counter >= self.max_steps:
            reward = np.sum([g*50*(1-0.9*(self.step_counter/self.max_steps)) for g in goals_reached])
            done = True
        elif all(goals_reached):
            reward = 200 * (1 - 0.9 * (self.step_counter / self.max_steps))
            done = True
        elif any(in_collision):
            reward = -1
        else:
            reward = np.sum([(a == 4)*(-0.1) for a in actions])

        # extra information for debug
        info = {
            'in_collision': in_collision,
            'goals_reached': goals_reached
        }

        return obs_states, reward, done, info


    '''
    if randomize = True, randomly place robots and goals in non-obstacle and non-overlapping positions
    '''
    def reset(self, randomize=False):
        self.step_counter = 0
        if randomize:
            occupied_locations = []

            for bot_index in range(self.nbots):
                rand_loc = (randint(0,self.nrows-1), randint(0,self.ncols-1))
                while (self.grid[rand_loc] == 2) or (rand_loc in occupied_locations):
                    rand_loc = (randint(0,self.nrows-1), randint(0,self.ncols-1))

                self.bot_locations[bot_index] = rand_loc
                occupied_locations.append(rand_loc)

            for goal_index in range(self.ngoals):
                rand_loc = (randint(0,self.nrows-1), randint(0,self.ncols-1))
                while (self.grid[rand_loc] == 2) or (rand_loc in occupied_locations):
                    rand_loc = (randint(0,self.nrows-1), randint(0,self.ncols-1))

                self.goal_locations[goal_index] = rand_loc
                occupied_locations.append(rand_loc)
        else:
            self.bot_locations = copy.deepcopy(self._start_bot_locations)
            self.goal_locations = copy.deepcopy(self._start_goal_locations)

        obs_states = self.get_obs_states()

        return obs_states

    '''
    0 -> empty
    1 -> bot
    2 -> obstacle
    3 -> goal
    '''
    def render(self, render_obs_grids=False):
        rendered_grid = copy.deepcopy(self.grid)
        for loc in self.bot_locations:
            rendered_grid[loc] = 1
        for (goal_r, goal_c) in self.goal_locations:
            if goal_r is None:
                continue
            rendered_grid[goal_r, goal_c] = 3
        print('------------------------------------------')
        print(rendered_grid)
        print('------------------------------------------')
        if render_obs_grids:
            obs_grids = self.create_observation_grids()
            for bot_index, grid in enumerate(obs_grids):
                print('Observation grid for robot {}:'.format(bot_index))
                print(grid)
            print('------------------------------------------')
