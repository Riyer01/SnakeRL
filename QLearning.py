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

# A nested dictionary that maps 
# state -> (action -> action-value). 
Q = defaultdict(dd) 

exists = os.path.isfile('qtable.p')
if exists:
    # Store configuration file values
    with open('qtable.p', 'rb') as fp:
        Q = pickle.load(fp)


def qLearning(Q, env, num_episodes, discount_factor = .9, 
                            alpha = .1, epsilon = 0.5, min_epsilon=.001, decay_rate=.99): 
    """ 
    Q-Learning algorithm: Off-policy TD control. 
    Finds the optimal greedy policy while improving 
    following an epsilon-greedy policy"""
    
    
    policy = createEpsilonGreedyPolicy(Q, env.action_space.n) 
    
    scores = [0] * num_episodes
    for ith_episode in range(num_episodes): 
        epsilon
        state = env.reset() 
        done = False
        t = 0
        totalReward = 0
        while not done:
            #Uncomment to run first 10 and last 10 iterations graphically
            # if ith_episode > num_episodes - 9 or ith_episode < 9:
            #     env.render(frame_speed=.01) 
            
            # get probabilities of all actions from current state 
            action_probabilities = policy(state, epsilon) 
   
            action = np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done, _ = env.step(int(action))
   
            totalReward += reward

            # Q-learning update Update 
            best_next_action = np.argmax(Q[next_state])     
            td_target = reward + discount_factor * Q[next_state][best_next_action] 
            td_delta = td_target - Q[state][action] 
            Q[state][action] += alpha * td_delta 

            t += 1 
            if done:
                scores[ith_episode] = env.controller.score
                print("Episode finished after {} timesteps with reward {} and score {}".format(t, totalReward, env.controller.score))
                break
                   
            state = next_state 
        epsilon = epsilon * decay_rate if epsilon > min_epsilon else min_epsilon
    return Q, scores

numGames = 5000

Q, scores = qLearning(Q, env, numGames)

# with open('qtable.p', 'wb') as fp:
#     pickle.dump(Q, fp, protocol=pickle.HIGHEST_PROTOCOL)


#Misc data visualization code 

# #Plot score after each game
# plt.plot(list(range(len(scores))), scores, 'ro')
# plt.ylabel('Score')
# plt.show()

#Plot average score over previous 500 games
# averages = []
# for i in range(500, len(scores)):
#     averages.append(sum(scores[i-500:i])/float(500))

# plt.plot(list(range(500, len(scores))), averages)
# plt.ylabel('Average Score over Last 500 Games')
# plt.xlabel('Number of Games')

averages = []
for i in range(100, len(scores)):
    averages.append(sum(scores[i-100:i])/float(100))

plt.plot(list(range(100, len(scores))), averages)
plt.ylabel('Average Score over Last 100 Games')
plt.xlabel('Number of Games')

n = len(scores)
print(sum(scores[n-100:])/float(len(scores[n-100:])))

plt.show()

# epsilon_values = []
# ave_scores = []
# numGames = 1100
# for epsilon in np.linspace(0.0, .9, 20):
#     print("TRAINING ON EPSILON = %s" % epsilon)
#     Q, scores = qLearning(defaultdict(dd), env, numGames, epsilon=epsilon)
#     ave_scores.append(sum(scores[numGames-100:])/float(len(scores[numGames-100:])))
#     epsilon_values.append(epsilon)

# (m, b) = np.polyfit(epsilon_values, ave_scores, 1)

# yp = np.polyval([m, b], epsilon_values)

# plt.scatter(epsilon_values, ave_scores)
# #plt.plot(epsilon_values, yp, color='red')
# plt.ylabel('Mean Score over Final 100 Games')
# plt.xlabel('Epsilon Value')
# plt.show()


# alpha_values = []
# ave_scores = []
# numGames = 1100
# for alpha in np.linspace(.1, .9, 18):
#     print("TRAINING ON ALPHA = %s" % alpha)
#     Q, scores = qLearning(defaultdict(dd), env, numGames, alpha=alpha)
#     ave_scores.append(sum(scores[numGames-100:])/float(len(scores[numGames-100:])))
#     alpha_values.append(alpha)

# (m, b) = np.polyfit(alpha_values, ave_scores, 1)

# yp = np.polyval([m, b], alpha_values)

# plt.scatter(alpha_values, ave_scores)
# plt.plot(alpha_values, yp, color='red')
# plt.ylabel('Mean Score over Final 100 Games')
# plt.xlabel('Alpha Value')
# plt.show()



env.close()