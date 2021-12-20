import numpy as np
import copy
from operator import add
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
    def __init__(self, map_filename="map1.txt", obs_range=2, max_steps=100):
        with open(map_filename, 'r') as f:
            self.obs_range = obs_range
            self.max_steps = max_steps
            self.step_counter = 0

            self.ncols, self.nrows = map(int, f.readline()[:-1].split())

            self.nbots = int(f.readline()[:-1])
            self._start_locations = [tuple(map(int, f.readline()[:-1].split())) for i in range(self.nbots)]
            self.bot_locations = copy.deepcopy(self._start_locations)

            self.ngoals = int(f.readline()[:-1])
            self.goal_locations = [tuple(map(int, f.readline()[:-1].split())) for i in range(self.ngoals)]

            map_list = [None] * self.nrows
            for i in range(self.nrows):
                map_list[i] = list(map(int, f.readline().rstrip('\n').split()))
            self.grid = np.array(map_list)

            '''
            0 -> empty
            1 -> bot
            2 -> obstacle
            3 -> goal
            '''
            # fill grid with bots and goals
            for loc in self.bot_locations:
                self.grid[loc] = 1
            for loc in self.goal_locations:
                self.grid[loc] = 3
            self._start_grid = copy.deepcopy(self.grid)


    def sample_actions(self):
        actions = [np.random.choice(len(individual_action_options)) for i in range(self.nbots)]
        return np.array(actions)


    '''
    observation grid:
        - one for each bot
        - same size as self.grid
        - only other bots and obstacles within obs_range from the current bot are visible
        - unreached goals are always visible regardless of obs_range
        - rest of grid marked as empty(0)
        - current bot's location marked as 9 to differentiate from other robots
    '''
    def create_observation_grids(self):
        obs_grids = [None] * self.nbots
        for i in range(self.nbots):
            grid = np.zeros((self.nrows, self.ncols))
            curr_bot_r, curr_bot_c = self.bot_locations[i]
            grid[
                max((curr_bot_r-self.obs_range),0) : min((curr_bot_r+self.obs_range+1),self.nrows+1),
                max((curr_bot_c-self.obs_range),0) : min((curr_bot_c+self.obs_range+1),self.ncols+1)
            ] = self.grid[
                    max((curr_bot_r-self.obs_range),0) : min((curr_bot_r+self.obs_range+1),self.nrows+1),
                    max((curr_bot_c-self.obs_range),0) : min((curr_bot_c+self.obs_range+1),self.ncols+1)
                ]

            for goal_loc in self.goal_locations:
                grid[goal_loc] = self.grid[goal_loc]

            grid[(curr_bot_r, curr_bot_c)] = 9

            obs_grids[i] = grid

        return obs_grids


    '''
    updates bot locations and grid based on action and returns data visible to RL

    NOTE: when bot moves onto goal, goal will be overwritten (i.e. 3 -> 1);
            this way goals that have already been reached no longer show on grid.
    '''
    def step(self, actions):
        assert len(actions) == self.nbots

        # calculate new locations after action
        new_locations = list(map(lambda t1, i2: tuple(map(add, t1, individual_action_options[i2])), self.bot_locations, actions))

        in_collision = [False] * self.nbots
        for i, (r, c) in enumerate(new_locations):
            # make sure bots don't go out of bounds of grid
            if (not (0 <= r <= (self.nrows-1))) or (not (0 <= c <= (self.ncols-1))):
                new_locations[i] = self.bot_locations[i]

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

        # remove old locationss from grid
        for loc in self.bot_locations:
            self.grid[loc] = 0
        # add new locations to grid
        for loc in new_locations:
            self.grid[loc] = 1

        self.bot_locations = new_locations
        self.step_counter += 1

        goals_reached = [False] * self.ngoals
        # check if goals have been reached
        for i, goal_loc in enumerate(self.goal_locations):
            if self.grid[goal_loc] != 3:
                goals_reached[i] = True

        # observation grids
        obs_grids = self.create_observation_grids()

        # rewards
        if any(in_collision):
            reward = 0
            done = True
        elif all(goals_reached):
            reward = 500 * (1 - 0.9 * (self.step_counter / self.max_steps))
            done = True
        elif self.step_counter >= self.max_steps:
            reward = 10 + np.sum([g*20 for g in goals_reached])
            done = True
        else:
            reward = np.sum([(a != 4)*0.05 for a in actions])
            done = False

        # extra information for debug
        info = {
            'in_collision': in_collision,
            'goals_reached': goals_reached
        }

        return obs_grids, reward, done, info


    def reset(self):
        self.step_counter = 0
        self.grid = copy.deepcopy(self._start_grid)
        self.bot_locations = copy.deepcopy(self._start_locations)

        obs_grids = self.create_observation_grids()

        return obs_grids


    def render(self):
        print('------------------------------------------')
        print(self.grid)
        print('------------------------------------------')


    @staticmethod
    def render_obs_grids(obs_grids):
        for idx, o in enumerate(obs_grids):
            print('Observation grid for robot {}:'.format(idx))
            print(o)
