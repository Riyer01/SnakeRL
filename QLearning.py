import gym
import gym_snake
import numpy as np
import itertools 
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import pickle

def createEpsilonGreedyPolicy(Q, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state, epsilon): 
   
        Action_probabilities = np.ones(num_actions, 
                dtype = float) * epsilon / num_actions 
        best_action = np.argmax(Q[state]) 
        Action_probabilities[best_action] += (1.0 - epsilon) 
        return Action_probabilities 
   
    return policyFunction 

env = gym.make('snake-v0', grid_size=[8, 8])

def dd():
    return np.zeros(env.action_space.n)

Q = defaultdict(dd) 

exists = os.path.isfile('qtable.p')
if exists:
    # Store configuration file values
    with open('qtable.p', 'rb') as fp:
        Q = pickle.load(fp)


def qLearning(Q, env, num_episodes, discount_factor = .9, 
                            alpha = 0.7, epsilon = 0.05, min_epsilon=.001, decay_rate=.99): 
    """ 
    Q-Learning algorithm: Off-policy TD control. 
    Finds the optimal greedy policy while improving 
    following an epsilon-greedy policy"""
       
    # Action value function 
    # A nested dictionary that maps 
    # state -> (action -> action-value). 
    #Q = defaultdict(lambda: np.zeros(env.action_space.n)) 
   
       
    # Create an epsilon greedy policy function 
    # appropriately for environment action space 
    policy = createEpsilonGreedyPolicy(Q, env.action_space.n) 
    
    scores = [0] * num_episodes
    # For every episode 
    for ith_episode in range(num_episodes): 
        epsilon
        # Reset the environment and pick the first action 
        state = env.reset() 
        done = False
        t = 0
        totalReward = 0
        while not done:
            if ith_episode > num_episodes - 9 or ith_episode < 9:
                env.render(frame_speed=.01) 
            #env.render(frame_speed=.005)
            
            # get probabilities of all actions from current state 
            action_probabilities = policy(state, epsilon) 
   
            # choose action according to  
            # the probability distribution 
            action = np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done, _ = env.step(int(action))
   
            totalReward += reward

            # TD Update 
            best_next_action = np.argmax(Q[next_state])     
            td_target = reward + discount_factor * Q[next_state][best_next_action] 
            td_delta = td_target - Q[state][action] 
            Q[state][action] += alpha * td_delta 

            t += 1
            # done is True if episode terminated    
            if done:
                scores[ith_episode] = env.controller.score
                print("Episode finished after {} timesteps with reward {} and score {}".format(t, totalReward, env.controller.score))
                break
                   
            state = next_state 
        epsilon = epsilon * decay_rate if epsilon > min_epsilon else min_epsilon
    return Q, scores

numGames = 1000

Q, scores = qLearning(Q, env, numGames)

with open('qtable.p', 'wb') as fp:
    pickle.dump(Q, fp, protocol=pickle.HIGHEST_PROTOCOL)

#Plot score after each game
plt.plot(list(range(len(scores))), scores, 'ro')
plt.ylabel('Score')
plt.show()

#Plot average score over previous 500 games
averages = []
for i in range(500, len(scores)):
    averages.append(sum(scores[i-500:i])/float(500))

plt.plot(list(range(500, len(scores))), averages)
plt.ylabel('Average Score over Last 500 Games')
plt.xlabel('Number of Games')

# max_so_far = [max(scores[:i]) for i in range(1, len(scores)+1)]
# plt.plot(list(range(len(scores))), max_so_far)
# plt.ylabel('Max Score Attained by Given Game')
plt.show()
print(sum(scores)/float(len(scores)))

env.close()