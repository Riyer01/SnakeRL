import gym
import gym_snake
from hamilton import Hamilton
from random import randint

env = gym.make('snake-v0', grid_size = [8, 8])

observation = env.reset()

num_games = 100
scores = [0] * num_games

for i_episode in range(num_games):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    while not done:
        # env.render()
        action = randint(0, 4)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        t += 1
        if done:
            scores[i_episode] = env.controller.score
            print("Episode finished after {} timesteps with reward {} and score {}".format(t, totalReward, env.controller.score))
            break
print(sum(scores)/float(len(scores)))
env.close()