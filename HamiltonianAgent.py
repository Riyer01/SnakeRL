import gym
import gym_snake
from hamilton import Hamilton
import random



size_x = 8
size_y = 8

env = gym.make('snake-v0', grid_size=[size_x, size_y])
observation = env.reset()

moves = [[-1, 0], [1, 0], [0, 1], [0, -1]]

def equalsGoal(cycle, x, y, goal):
    if x >= len(cycle) or y >= len(cycle[0]) or x < 0 or y < 0:
        return False
    return cycle[x][y] == goal
    

def findNextMove(cycle, x, y):
    curr = cycle[x][y]
    goal = (curr % (size_x*size_y)) + 1
    if equalsGoal(cycle, x+1, y, goal):
        return 1
    if equalsGoal(cycle, x-1, y, goal):
        return 3
    if equalsGoal(cycle, x, y-1, goal):
        return 0
    if equalsGoal(cycle, x, y+1, goal):
        return 2

for i_episode in range(20):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    snake = Hamilton(moves, size_x, size_y, observation[0][0], observation[0][1], closed=True)
    snake.solve()
    while not done:
        env.render(frame_speed=.005)
        #snake_head, food_location = observation[0], observation[1]
        print(observation)
        snake_head, food_location = tuple(env.controller.snakes[0].head), env.controller.grid.foodLocations[0] 
        action = findNextMove(snake.board, snake_head[0], snake_head[1])
        observation, reward, done, info = env.step(action)
        totalReward += reward
        t += 1
        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, totalReward))
            break
env.close()