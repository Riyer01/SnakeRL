import gym
import gym_snake
from hamilton import Hamilton

env = gym.make('snake-v0')
observation = env.reset()

def getGreedyAction(obs):
    snake_head, food_location = observation[0], observation[1]
    


for i_episode in range(20):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    while not done:
        env.render()
        action = getGreedyAction(observation)
        observation, reward, done, info = env.step(1)
        totalReward += reward
        t += 1
        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, totalReward))
            break
env.close()