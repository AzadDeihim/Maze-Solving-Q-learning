import numpy as np
from Maze import Maze
import random
from Agent_Basic import Agent
from hyperopt import Trials, fmin, STATUS_OK, tpe, hp

nx, ny = 15, 15
ix, iy = 0, 0

param_history = []


def objective(hyperparameters):
    #dictionary containing
    dict = {'loss': 0,
            'status': STATUS_OK
            }


    episodes_to_optimal = []

    #loop for 100 iterations
    for k in range(100):

        episodes = 500

        #create maze
        maze = Maze(nx, ny, ix, iy)
        maze.maze_map = np.load('maze.npy', allow_pickle=True)
        #maze.write_svg('maze.svg')

        #create agent with parameters given by bayesian search
        agent = Agent(epsilon=hyperparameters['epsilon'], gamma=hyperparameters['gamma'],
                            lr=hyperparameters['lr'], maze=maze)


        for _ in range(episodes):
            agent.reset()
            num_steps = 0

            #while not terminal state
            #training loop
            while not agent.isGoal():
                num_steps += 1

                #generate all possible actions
                moves = agent.getPossibleActions()
                prev_state = agent.currentState()

                #explore or exploit
                if random.uniform(0, 1) < agent.epsilon:
                    new_state = random.choice(moves)

                else:
                    Qs = []
                    #generate q values for all possible actions
                    for move in moves:
                        Qs.append(agent.getQValue(prev_state, move))

                    #choose the action with the q value, if theres a tie pick one at random
                    if len(set(Qs)) <= 1:
                        new_state = random.choice(moves)
                    else:
                        new_state = moves[np.argmax(Qs)]

                #make action
                agent.move(new_state)
                #get reward
                r = agent.getReward(prev_state, new_state)
                #update q
                agent.updateQTable(prev_state, new_state, r)




            num_steps = 0
            agent.reset()

            #testing loop
            while not agent.isGoal():
                num_steps += 1
                moves = agent.getPossibleActions()
                prev_state = agent.currentState()
                Qs = []

                #always exploit
                for move in moves:
                    Qs.append(agent.getQValue(prev_state, move))

                if len(set(Qs)) <= 1:
                    new_state = random.choice(moves)
                else:
                    new_state = moves[np.argmax(Qs)]

                agent.move(new_state)

            #if the agent found the optimal solution, store episode number and end iteration
            if num_steps == 116:
                episodes_to_optimal.append(_)
                break


            #decay
            if agent.epsilon >= 0.5:
                agent.epsilon *= 0.99999
            else:
                agent.epsilon *= 0.9999
    current_params = []

    #average score across all 100 iterations
    dict['loss'] = np.average(episodes_to_optimal)
    print('Number of episodes to reach solution: ' + str(dict['loss']))
    print('Epsilon: ' + str(hyperparameters['epsilon']))
    print('Gamma: ' + str(hyperparameters['gamma']))
    print('Alpha: ' + str(hyperparameters['lr']))

    #store and save params
    current_params.append([hyperparameters['epsilon'], hyperparameters['gamma'],
                            hyperparameters['lr'], dict['loss']])
    param_history.append(current_params)
    np.save('bayes_search', param_history)

    return dict



#defines the search space for parameters search
space = {
    'lr': hp.uniform('lr', 0, 1),
    'epsilon': hp.uniform('epsilon', 0, 1),
    'gamma': hp.uniform('gamma', 0, 1),
}


bayes_trials = Trials()
#bayes search will last 100 iterations
MAX_EVALS = 100
# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS)




