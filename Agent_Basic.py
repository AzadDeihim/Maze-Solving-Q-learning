import numpy as np


class Agent:

    def __init__(self, epsilon, gamma, lr, maze):
        #init params
        self.q_table = np.zeros((maze.nx * maze.ny, maze.nx * maze.ny))
        self.maze = maze
        self.rewards_table = self.createRewardTable()
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.current_x = 14
        self.current_y = 14


    def getPossibleActions(self):
        '''
        Calculates all possible actions based on current state

        :return:
        list of all possible actions in the current state
        '''

        walls = self.maze.maze_map[self.current_x][self.current_y].walls
        actions = []
        if not walls['N']:
            actions.append([self.current_x, self.current_y - 1])
        if not walls['S']:
            actions.append([self.current_x, self.current_y + 1])
        if not walls['E']:
            actions.append([self.current_x + 1, self.current_y])
        if not walls['W']:
            actions.append([self.current_x - 1, self.current_y])

        return actions

    def move(self, action):
        '''
        Moves agent to a new location

        :param action: direction to move

        :return:
        nothing
        '''
        self.current_x = action[0]
        self.current_y = action[1]

    def isGoal(self):
        '''
        Boolean function to determine if the current state is the terminal state

        :return:
        boolean denoting if current state is terminal or not
        '''
        if (self.current_x == 0) and (self.current_y == 0):
            return True
        return False

    def reset(self):
        '''
        Places the agent back in the starting position

        :return:
        nothing
        '''
        self.current_x = 14
        self.current_y = 14

    def updateQTable(self, prev_state, new_state, reward):
        '''
        Updates the Q table using the Q update equation

        :param prev_state: previous state
        :param new_state: new state
        :param reward: reward obtained from state transition

        :return:
        nothing
        '''
        prev_state = self.stateToIndex(prev_state)
        new_state = self.stateToIndex(new_state)
        self.q_table[prev_state, new_state] = self.q_table[prev_state, new_state] + self.lr * (reward + self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[prev_state, new_state])

    def stateToIndex(self, state):
        '''
        Converts a state to an index in the Q or rewards table

        :param state: any state

        :return:
        Q/reward table index
        '''
        return state[1]*15+state[0]


    def createRewardTable(self):
        '''
        Creates an n*m by n*m rewards table

        :return:
        rewards table
        '''
        X, Y = [0, 1], [0, 1]
        goal_state = self.stateToIndex([self.maze.ix, self.maze.iy])
        rewards = np.zeros((self.maze.nx * self.maze.ny, self.maze.nx * self.maze.ny))
        for x in X:
            for y in Y:
                if (x == 0) and (y == 0):
                    continue
                prev_state = self.stateToIndex([x, y])
                rewards[prev_state, goal_state] = 1000
        return rewards

    def currentState(self):
        '''

        :return:
        current state
        '''
        return [self.current_x, self.current_y]


    def getReward(self, prev_state, new_state):
        '''
        Gives reward for state transition

        :param prev_state: previous state
        :param new_state: new state

        :return:
        reward
        '''
        prev_state = self.stateToIndex(prev_state)
        new_state = self.stateToIndex(new_state)
        return self.rewards_table[prev_state, new_state]


    def getQValue(self, prev_state, new_state):
        '''
        Gives q value for state transition

        :param prev_state: previous state
        :param new_state: new state

        :return:
        q value
        '''
        prev_state = self.stateToIndex(prev_state)
        new_state = self.stateToIndex(new_state)
        return self.q_table[prev_state, new_state]

