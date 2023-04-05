import numpy as np
import gym
from gym import spaces
from ray.rllib.env import MultiAgentEnv
import copy

from configs import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.colors import ListedColormap

# ground, ore, player 1 factory, player 1 robot, player 2 factory, player 2 robot
cmap = ListedColormap(['#FFFFFF',   # ground
                       '#4d2600',   # ore
                       '#0000ff',   # player 1 factory
                       '#3399ff',   # player 1 robot
                       '#ff0000',   # player 2 factory
                       '#ff8080'])  # player 2 robot

class GridWorldEnv(MultiAgentEnv):

        
    def __init__(self, config=None):
        self.agents = ['player_1', 'player_2']
        
        # observation space        
        # two factory positions, six robot positions
        player_1_factory_position = spaces.MultiDiscrete([10, 10])
        player_2_factory_position = spaces.MultiDiscrete([10, 10])

        player_1_robot_positions = spaces.MultiDiscrete([101, 101, 101]) # (100) for dead robots
        player_2_robot_positions = spaces.MultiDiscrete([101, 101, 101])
        
        
        player_1_robot_ore = spaces.MultiDiscrete([ROBOT_MAX_ORE_CAPACITY, ROBOT_MAX_ORE_CAPACITY, ROBOT_MAX_ORE_CAPACITY])
        player_2_robot_ore = spaces.MultiDiscrete([ROBOT_MAX_ORE_CAPACITY, ROBOT_MAX_ORE_CAPACITY, ROBOT_MAX_ORE_CAPACITY])

        # common observations
        NUM_OF_ORES = 5
        ore_positions = spaces.MultiDiscrete([100,]*NUM_OF_ORES)

        self.observation_space = spaces.Dict({
            'common': spaces.Dict({
                'ore_positions': ore_positions,
            }),
            'player_1': spaces.Dict({
                'factory_position': player_1_factory_position,
                'robot_positions': player_1_robot_positions,
                'robot_ore': player_1_robot_ore
            }),
            'player_2': spaces.Dict({
                'factory_position': player_2_factory_position,
                'robot_positions': player_2_robot_positions,
                'robot_ore': player_2_robot_ore
            })
        })
        
        # action space
        self.action_space = spaces.MultiDiscrete(1+  # 0:     stay still
                                                 1+  # 1:     mine
                                                 4+  # 2-5:   move in 4 directions
                                                 4   # 6-9:   pass resource in 4 directions
                                                 )
        
        # TODO for now, factory automatically convert ore to score

    def reset(self):
        # initialization
        def check_initial_state_validity(initial_state):
            # check factory positions are not the same
            # use .all() to compare two numpy arrays
            if (initial_state['player_1']['factory_position'] == initial_state['player_2']['factory_position']).all():
                return False
            
            # check duplicate robot positions
            positions = []
            for agent in self.agents:
                for robot in initial_state[agent]['robot_positions']:
                    positions.append(robot)
            if len(positions) != len(set(positions)):
                return False
            
            return True
        
        # sample observation space
        while True:
            initial_observation = self.observation_space.sample()
            # set robot ore to 0
            initial_observation['player_1']['robot_ore'] = [0, 0, 0]
            initial_observation['player_2']['robot_ore'] = [0, 0, 0]
            if check_initial_state_validity(initial_observation):
                break
        
        self.current_state = initial_observation
        
        # for recording game history
        self.history_observation = []
        self.scores_history = []
        
        
        # accumulated reward design
        self.reward = {
            'player_1': {
                0: 0,
                1: 0,
                2: 0
            },
            'player_2': {
                0: 0,
                1: 0,
                2: 0
            }
        }
        
        self.info = {
            'scores': {'player_1': 0, 'player_2': 0},
            'time_left': 100
        }
        
        return initial_observation
    
    def _sample_observation(self):
        '''
        for testing render function
        '''
        observation = self.observation_space.sample()
        
        return observation
    
    def _convert_position_to_xy(self, robot_position):
        '''convert 0-99 position to (x, y)'''
        return (robot_position // 10, robot_position % 10)
        
    def record(self, new_obs=None):
        if new_obs is None:
            new_obs = self.current_state
        
        '''record the observation for rendering'''
        frame = np.zeros((10, 10))
        
        # ore positions
        ores = new_obs['common']['ore_positions']
        for pos in ores:
            x, y = self._convert_position_to_xy(pos)
            frame[x][y] = 1
        
        # player 1
        x, y = new_obs['player_1']['factory_position']
        frame[x][y] = 2
        
        for pos in new_obs['player_1']['robot_positions']:
            if pos == 100:
                continue
            x, y = self._convert_position_to_xy(pos)
            frame[x][y] = 3
        
        # player 2
        x, y= new_obs['player_2']['factory_position']
        frame[x][y] = 4
        
        for pos in new_obs['player_2']['robot_positions']:
            if pos == 100:
                continue
            x, y = self._convert_position_to_xy(pos)
            frame[x][y] = 5
        
        self.history_observation.append(frame)
        self.scores_history.append(self.info['scores'])
    
    def render(self, framerate=2):
        '''render observation recorded in history'''
        # print the grid world as array
        n_frames = len(self.history_observation)
        print("Rendering %d frames..." % n_frames)
        fig = plt.figure(figsize=(6, 2))
        fig_grid = fig.add_subplot(121)
        fig_1_score = fig.add_subplot(243)
        fig_2_score = fig.add_subplot(244)
        
        def render_frame(i):
            fig_grid.matshow(self.history_observation[i], cmap=cmap)
        
        self.anim = matplotlib.animation.FuncAnimation(
            fig, render_frame, frames=n_frames, interval=1000/framerate
        )
        fig.show()
    
    def get_observation(self, agent_id):
        '''not using this function for now'''
        if agent_id == 'player_1':
            other_agent_id = 'player_2'
            # observation = 
        else:
            other_agent_id = 'player_1'
    
    def _robot_on_ore(self, intermediate_state, agent_id, robot_id):
        '''check if robot is on ore'''
        robot_pos = intermediate_state[agent_id]['robot_positions'][robot_id]
        if robot_pos in intermediate_state['common']['ore_positions']:
            return True
        else:
            return False
    
    def _robot_add_ore(self, intermediate_state, agent_id, robot_id, amount):
        '''
        add ore to robot if it is on ore
        '''
        # check if already full capacity
        if intermediate_state[agent_id]['robot_ore'][robot_id] == ROBOT_MAX_ORE_CAPACITY:
            return -amount
        
        # add ore
        intermediate_state[agent_id]['robot_ore'][robot_id] += amount
        
        # check ore capacity
        if intermediate_state[agent_id]['robot_ore'][robot_id] > ROBOT_MAX_ORE_CAPACITY:
            intermediate_state[agent_id]['robot_ore'][robot_id] = ROBOT_MAX_ORE_CAPACITY
        return (amount - intermediate_state[agent_id]['robot_ore'][robot_id]) * REWARD_PER_RESOURCE_TRANSFERRED
    
    def _robot_move(self, intermediate_state, agent_id, robot_id, robot_action):
        '''
        move robot to new position
        '''
        # get robot position
        robot_position = intermediate_state[agent_id]['robot_positions'][robot_id]
        
        # get new position
        x, y = self._convert_position_to_xy(robot_position)
        if robot_action == 2:
            y += 1
        elif robot_action == 3:
            y -= 1
        elif robot_action == 4:
            x -= 1
        elif robot_action == 5:
            x += 1
        
        # check if new position is valid
        if x < 0 or x > 9 or y < 0 or y > 9:
            return REWARD_INVALID_ACTION
        
        # check if new position is occupied
        new_position = x * 10 + y
        
        # check if new position is occupied by other robots
        for other_agent in self.agents:
            if new_position in intermediate_state[other_agent]['robot_positions']:
                return REWARD_INVALID_ACTION
        
        # update robot position
        intermediate_state[agent_id]['robot_positions'][robot_id] = new_position
        return 0
    
    def _robot_pass_resource(self, intermediate_state, agent_id, robot_id, robot_action):
        '''
        pass resource to other robots
        
        Return:
        '''
        # if no resources to pass
        if intermediate_state[agent_id]['robot_ore'][robot_id] == 0:
            return REWARD_INVALID_ACTION
        
        # get new position
        x, y = self._convert_position_to_xy(intermediate_state[agent_id]['robot_positions'][robot_id])
        if robot_action == 2:
            y += 1
        elif robot_action == 3:
            y -= 1
        elif robot_action == 4:
            x -= 1
        elif robot_action == 5:
            x += 1
        
        # check if new position is valid
        if x < 0 or x > 9 or y < 0 or y > 9:
            return REWARD_INVALID_ACTION
        
        # check if new position is occupied by friendly robots or factory
        new_position = x * 10 + y
        factory_x, factory_y = intermediate_state[agent_id]['factory_position']
        if new_position not in intermediate_state[agent_id]['robot_positions'] and \
            new_position != factory_x * 10 + factory_y:
            return REWARD_INVALID_ACTION
    
        transferred_resource = intermediate_state[agent_id]['robot_ore'][robot_id]
        intermediate_state[agent_id]['robot_ore'][robot_id] = 0
        
        # pass resource to factory
        if new_position == factory_x * 10 + factory_y:
            
            # convert resource to score
            self.info['scores'][agent_id][robot_id] += transferred_resource
            
            # return reward
            return transferred_resource * REWARD_PER_RESOURCE_TRANSFERRED
        
        # pass to robots
        target_robot = np.where(intermediate_state[agent_id]['robot_positions'] == new_position)
        target_robot = target_robot[0][0]
        transfer_result_reward = self._robot_add_ore(intermediate_state, agent_id, target_robot, transferred_resource)
        return transfer_result_reward
        
    
    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.
        
        e.g.
        action={'player_1':
                    {0: 0, 1: 0, 2: 0},
                'player_2':
                    {0: 0, 1: 0, 2: 0}
                }
        """
        # check game is done
        self.time_left -= 1
        is_done = {
            '__all__': self.time_left <= 0,
        }
        
        intermediate_state = copy.deepcopy(self.current_state)
        reward = copy.deepcopy(self.reward)
        
        # move agents
        for agent in self.agents:
            for robot_id, robot_action in action[agent].items():
                # check if robot is dead
                if intermediate_state[agent]['robot_positions'][robot_id] == 100:
                    continue
                
                # reward per turn
                reward[agent][robot_id] += REWARD_PER_TURN
                
                # stay still
                if robot_action == 0:
                    pass
                
                # mine
                if robot_action == 1:
                    if self._robot_on_ore(intermediate_state, agent, robot_id):
                        reward[agent][robot_id] = self._robot_add_ore(intermediate_state, agent, robot_id, ROBOT_ORE_PER_TURN)
                    else:
                        # robot is not on ore
                        reward[agent][robot_id] += REWARD_INVALID_ACTION
                
                # move
                if 2 <= robot_action <=5:
                    reward[agent][robot_id] = self._robot_move(intermediate_state, agent, robot_id, robot_action)
                        
                # pass resource
                if 6 <= robot_action <= 9:
                    reward[agent][robot_id] = self._robot_pass_resource(intermediate_state, agent, robot_id, robot_action)
        
        
        if is_done['__all__']:
            if self.info['scores']['player_1'] > self.info['scores']['player_2']:
                reward['player_1'] += REWARD_SUCCESS
            if self.info['scores']['player_1'] < self.info['scores']['player_2']:
                reward['player_2'] += REWARD_SUCCESS
        
        self.reward = reward
        self.current_state = intermediate_state
        return self.current_state, self.reward, is_done, {}