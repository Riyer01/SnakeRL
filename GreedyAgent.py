import gym
import gym_snake
from hamilton import Hamilton
from random import randint

env = gym.make('snake-v0', grid_size = [8, 8])

observation = env.reset()

num_games = 100
scores = [0] * num_games

def simulate_move(coord, direc):
    move_map = {3:(-1, 0), 1:(1,0), 2:(0,1), 0:(0,-1)}
    x_delta, y_delta = move_map[direc]
    new_coord = (coord[0]+x_delta, coord[1]+y_delta)
    return new_coord

def manhattanDistance(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return abs(x2-x1) + abs(y2-y1)

def getGreedyAction(snake_head, food_location, curr_direction, grid):
    valid_directions = {0:[0, 1, 3], 1:[0, 1, 2], 2:[1, 2, 3], 3:[0, 2, 3]}
    moves = []
    for direc in valid_directions[curr_direction]:
        new_head = simulate_move(snake_head, direc)
        if not grid.check_death(new_head):
            moves.append((manhattanDistance(new_head, food_location), direc))
    if moves:
        return min(moves)[1]
    return 0


for i_episode in range(num_games):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    while not done:
        #env.render(frame_speed=.001)
        snake_head, food_location = tuple(env.controller.snakes[0].head if env.controller.snakes and env.controller.snakes[0] else [0, 0]), env.controller.grid.foodLocations[0], 
        curr_direction = env.controller.snakes[0].direction if env.controller.snakes and env.controller.snakes[0] else 0
        action = getGreedyAction(snake_head, food_location, curr_direction, env.controller.grid)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        t += 1
        if done:
            scores[i_episode] = env.controller.score
            print("Episode finished after {} timesteps with reward {} and score {}".format(t, totalReward, env.controller.score))
            break
print(sum(scores)/float(len(scores)))
env.close()