import gym
import gym_snake
import numpy as np
import itertools 
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import pickle

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state): 
   
        Action_probabilities = np.ones(num_actions, 
                dtype = float) * epsilon / num_actions 
        best_action = np.argmax(Q[state]) 
        Action_probabilities[best_action] += (1.0 - epsilon) 
        return Action_probabilities 
   
    return policyFunction 

env = gym.make('snake-v0', grid_size=[20, 20])
# observation = env.reset()

# eta = .628
# gma = .9
# epis = 5000
# rev_list = [] # rewards per episode calculate
# Q = {}

# for i in range(epis):
#     # Reset environment
#     s = env.reset()
#     done = False

#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
#     #The Q-Table learning algorithm
#     while not done:
#         env.render()
#         j+=1
#         # Choose action from Q table
#         a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
#         #Get new state & reward from environment
#         s1,r,d, _ = env.step(int(a))
#         #Update Q-Table with new knowledge
#         Q[s][a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
#         rAll += r
#         s = s1
#         if d == True:
#             break
#     rev_list.append(rAll)
#     env.render()

def dd():
    return np.zeros(env.action_space.n)

Q = defaultdict(dd) 

exists = os.path.isfile('qtable.p')
if exists:
    # Store configuration file values
    with open('qtable.p', 'rb') as fp:
        Q = pickle.load(fp)


def qLearning(Q, env, num_episodes, discount_factor = .9, 
                            alpha = 0.85, epsilon = 0.05): 
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
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n) 
    
    scores = [0] * num_episodes
    # For every episode 
    for ith_episode in range(num_episodes): 
           
        # Reset the environment and pick the first action 
        state = env.reset() 
        done = False
        t = 0
        totalReward = 0
        while not done:
            # if ith_episode > num_episodes - 9:
            #     env.render(frame_speed=.01) 
            # env.render(frame_speed=.01)
            # print(state)
            # get probabilities of all actions from current state 
            action_probabilities = policy(state) 
   
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
      
    return Q, scores

Q, scores = qLearning(Q, env, 10000)
# print(sum(scores[950:])/float(len(scores[950:])))

with open('qtable.p', 'wb') as fp:
    pickle.dump(Q, fp, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(list(range(len(scores))), scores, 'ro')
plt.ylabel('Score')
plt.show()

env.close()