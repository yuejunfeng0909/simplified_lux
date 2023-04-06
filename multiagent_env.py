import numpy as np
from gymnasium import spaces
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
cmap = ListedColormap(['#ffffff',   # ground
                       '#4d2600',   # ore
                       '#0000ff',   # player 1 factory
                       '#3399ff',   # player 1 robot
                       '#ff0000',   # player 2 factory
                       '#ff8080'])  # player 2 robot

class GridWorldEnv(MultiAgentEnv):

        
    def __init__(self, config=None):
        super().__init__()
        self.agents = ['team_0', 'team_1']
        
        # observation space
        NUM_OF_ORES = 5
        self.observation_space = spaces.Dict({
            'common': spaces.Dict({
                'ore_positions': spaces.MultiDiscrete([100,]*NUM_OF_ORES, dtype=np.int64),
            }),
            'friendly': spaces.Dict({
                'factory_position': spaces.MultiDiscrete([100]),
                'robot_positions': spaces.MultiDiscrete([101, 101, 101]),
                'robot_ore': spaces.MultiDiscrete([ROBOT_MAX_ORE_CAPACITY + 1, 
                                                   ROBOT_MAX_ORE_CAPACITY + 1, 
                                                   ROBOT_MAX_ORE_CAPACITY + 1, ]),
            }),
            'opponent': spaces.Dict({
                'factory_position': spaces.MultiDiscrete([100]),
                'robot_positions': spaces.MultiDiscrete([101, 101, 101]),
                'robot_ore': spaces.MultiDiscrete([ROBOT_MAX_ORE_CAPACITY + 1, 
                                                   ROBOT_MAX_ORE_CAPACITY + 1, 
                                                   ROBOT_MAX_ORE_CAPACITY + 1, ]),
            })
        })
        
        # action space
        num_of_actions = 1+1+4+4
        # 0: stay still
        # 1: mine
        # 2-5: move in 4 directions
        # 6-9: pass resource in 4 directions

        self.action_space = spaces.MultiDiscrete([num_of_actions, num_of_actions, num_of_actions])
        
        # TODO for now, factory automatically convert ore to score

    def _get_observations_for_agent(self, agent_id):
        opponent_id = 'team_0' if agent_id == 'team_1' else 'team_1'
        state = {
            'common': {
                'ore_positions': self.ores_position,
            },
            'friendly': {
                'factory_position': self.factory_positions[agent_id],
                'robot_positions': self.robot_positions[agent_id],
                'robot_ore': self.robot_ores[agent_id],
            },
            'opponent': {
                'factory_position': self.factory_positions[opponent_id],
                'robot_positions': self.robot_positions[opponent_id],
                'robot_ore': self.robot_ores[opponent_id],
            }
        }
        return state

    def _get_observation(self):
        observation = {
            'team_0': self._get_observations_for_agent('team_0'),
            'team_1': self._get_observations_for_agent('team_1'),
        }
        return observation
    
    def _randomize_state(self):
        # initialize ores
        # randomly choose 5 positions, 0-99
        # initialize factory positions, cannot be on ores
        random_positions = np.random.choice(100, 7, replace=False)
        self.ores_position = random_positions[:5]
        self.factory_positions = {
            "team_0": np.array([random_positions[5],]),
            "team_1": np.array([random_positions[6],]),
        }
        
        # initialize positions of robots, 3 per team, 0-99
        self.robot_positions = {}
        # example: {'team_0': [1, 2, 3], 'team_1': [4, 5, 6]}
        
        for team in self.agents:
            # get factory position
            factory_position = self.factory_positions[team][0]
            
            # put the agent anywhere on the 8 positions around the factory
            factory_x, factory_y = self._convert_position_to_xy(factory_position)
            positions_around_factory = []
            for x in range(factory_x-1, factory_x+2):
                for y in range(factory_y-1, factory_y+2):
                    # check if the position is valid
                    if x < 0 or x >= 10 or y < 0 or y >= 10:
                        continue
                    # cannot be on factory
                    if x == factory_x and y == factory_y:
                        continue
                    positions_around_factory.append(x*10+y)
            
            # randomly choose 3 positions
            random_positions = np.random.choice(positions_around_factory, 3, replace=False)
            self.robot_positions[team] = random_positions
        
        # initialize robots with zero ore
        self.robot_ores = {
            "team_0": np.array([0, 0, 0]),
            "team_1": np.array([0, 0, 0])
        }

    def reset(self, *, seed=None, options=None):
        # initialization of internal state
        self._randomize_state()
        initial_observation = self._get_observation()
        
        # for recording game history
        self.history_observation = []
        self.scores_history = []
        
        
        # accumulated reward design
        self.reward={'team_0': 0, 'team_1': 0}
        
        self.info = {
            'scores': {'team_0': 0, 'team_1': 0},
            'time_left': 100
        }
        return initial_observation, {}
    
    def _sample_observation(self):
        '''
        for testing render function
        '''
        self._randomize_state()
        observation = self._get_observation()
        
        return observation
    
    def _convert_position_to_xy(self, robot_position):
        '''convert 0-99 position to (x, y)'''
        return (robot_position // 10, robot_position % 10)
        
    def record(self):
        '''record the observation for rendering'''
        frame = np.zeros((10, 10), dtype=np.int32)
        
        # ore positions
        ores = self.ores_position
        for pos in ores:
            x, y = self._convert_position_to_xy(pos)
            frame[x][y] = 1
        
        # team_0
        x, y = self._convert_position_to_xy(self.factory_positions['team_0'])
        frame[x][y] = 2
        
        for pos in self.robot_positions['team_0']:
            if pos == 100:
                continue
            x, y = self._convert_position_to_xy(pos)
            frame[x][y] = 3
        
        # team_1
        x, y= self._convert_position_to_xy(self.factory_positions['team_1'])
        frame[x][y] = 4
        
        for pos in self.robot_positions['team_1']:
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
    
    def _robot_on_ore(self, agent_id, robot_id):
        '''check if robot is on ore'''
        robot_pos = self.robot_positions[agent_id][robot_id]
        if robot_pos in self.ores_position:
            return True
        else:
            return False
    
    def _robot_add_ore(self, agent_id, robot_id, amount):
        '''
        add ore to robot if it is on ore
        '''
        # check if already full capacity
        if self.robot_ores[agent_id][robot_id] == ROBOT_MAX_ORE_CAPACITY:
            return -amount * REWARD_PER_RESOURCE_WASTED
        
        # add ore
        self.robot_ores[agent_id][robot_id] += amount
        
        # check ore capacity
        if self.robot_ores[agent_id][robot_id] > ROBOT_MAX_ORE_CAPACITY:
            self.robot_ores[agent_id][robot_id] = ROBOT_MAX_ORE_CAPACITY
        return (amount - self.robot_ores[agent_id][robot_id]) * REWARD_PER_RESOURCE_TRANSFERRED
    
    def _robot_move(self, agent_id, robot_id, robot_action):
        '''
        move robot to new position
        '''
        # get robot position
        robot_position = self.robot_positions[agent_id][robot_id]
        
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
        for team in self.robot_positions:
            for pos in self.robot_positions[team]:
                if pos == new_position:
                    return REWARD_INVALID_ACTION
        
        # update robot position
        self.robot_positions[agent_id][robot_id] = new_position
        return 0
    
    def _robot_pass_resource(self, agent_id, robot_id, robot_action):
        '''
        pass resource to other robots
        
        Return:
        '''
        # if no resources to pass
        if self.robot_ores[agent_id][robot_id] == 0:
            return REWARD_INVALID_ACTION
        
        # get new position
        x, y = self._convert_position_to_xy(self.robot_positions[agent_id][robot_id])
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
        factory_x, factory_y = self._convert_position_to_xy(self.factory_positions[agent_id])
        if new_position not in self.robot_positions[agent_id] and\
            new_position != factory_x * 10 + factory_y:
            return REWARD_INVALID_ACTION
    
        transferred_resource = self.robot_ores[agent_id][robot_id]
        self.robot_ores[agent_id][robot_id] = 0
        
        # pass resource to factory
        if new_position == factory_x * 10 + factory_y:
            
            # convert resource to score
            self.info['scores'][agent_id] += transferred_resource
            
            # return reward
            return transferred_resource * REWARD_PER_RESOURCE_TRANSFERRED
        
        # pass to robots
        target_robot = np.where(self.robot_positions[agent_id] == new_position)[0][0]
        transfer_result_reward = self._robot_add_ore(agent_id, target_robot, transferred_resource)
        return transfer_result_reward   
    
    def step(self, actions: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.
        
        e.g.
        action={
            'team_0': [0, 1, 2]
            'team_1': [2, 1, 5]
        }
        """
        
        # convert robot action to player action:
        
        # check game is done
        self.info['time_left'] -= 1
        truncateds = {
            '__all__': self.info['time_left'] <= 0,
        }
        terminateds = {
            '__all__': self.info['time_left'] <= 0,
        }
        
        reward = self.reward
        
        # move agents
        for team in actions:
            for agent in range(3):
                agent_action = actions[team][agent]
                # check if robot is dead
                if self.robot_positions[team][agent] == 100:
                    continue
                
                # reward per turn
                reward[team] += REWARD_PER_TURN
                
                # stay still
                if agent_action == 0:
                    pass
                
                # mine
                if agent_action == 1:
                    if self._robot_on_ore(team, agent):
                        reward[team] = self._robot_add_ore(team, agent, ROBOT_ORE_PER_TURN)
                    else:
                        # robot is not on ore
                        reward[team] += REWARD_INVALID_ACTION
                
                # move
                if 2 <= agent_action <=5:
                    reward[team] = self._robot_move(team, agent, agent_action)
                        
                # pass resource
                if 6 <= agent_action <= 9:
                    reward[team] = self._robot_pass_resource(team, agent, agent_action)
        
        
        if truncateds['__all__']:
            if self.info['scores']['team_0'] > self.info['scores']['team_1']:
                reward[team] += REWARD_SUCCESS
            if self.info['scores']['team_0'] < self.info['scores']['team_1']:
                reward[team] += REWARD_SUCCESS
        
        new_obs = self._get_observation()
        return new_obs, reward, terminateds, truncateds, {}