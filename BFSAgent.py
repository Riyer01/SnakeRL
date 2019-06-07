import gym
import gym_snake
from hamilton import Hamilton
from copy import deepcopy
import matplotlib.pyplot as plt

env = gym.make('snake-v0', grid_size = [8, 8])
observation = env.reset()

def generateMoveQueue(path):
    move_map = {(-1, 0):3, (1,0):1, (0,1):2, (0,-1):0}
    if path:
        return [move_map[(y[0]-x[0], y[1]-x[1])] for x, y in zip(path, path[1:])]
    return [0] * 1000

def getGreedyAction(snake_head, food_location, grid):
    start = snake_head
    end = food_location

    path = BFS(grid, start, end)
    queue = generateMoveQueue(path)
    return queue

def BFS(maze, start, end):
    '''"Brute-Force Search"
    :param maze(list): the maze to be navigated
    :param start(tuple): the starting coordinates (row, col)
    :param end(tuple): the end coordinates (row, col)
    :return: shortest path from start to end
    '''
    queue = [start]
    visited = set()

    while len(queue) != 0:
        if queue[0] == start:
            path = [queue.pop(0)]  # Required due to a quirk with tuples in Python
        else:
            path = queue.pop(0)
        front = path[-1]
        if front == end:
            return path
        elif front not in visited:
            for adjacentSpace in getAdjacentSpaces(maze, front, visited):
                newPath = list(path)
                newPath.append(adjacentSpace)
                queue.append(newPath)
            visited.add(front)
    return None  

def getAdjacentSpaces(maze, space, visited):
    ''' Returns all legal spaces surrounding the current space
    :param space: tuple containing coordinates (row, col)
    :return: all legal spaces
    '''
    spaces = list()
    spaces.append((space[0]-1, space[1]))  # Up
    spaces.append((space[0]+1, space[1]))  # Down
    spaces.append((space[0], space[1]-1))  # Left
    spaces.append((space[0], space[1]+1))  # Right

    final = list()
    for i in spaces:
        if not maze.check_death(i) and i not in visited:
            final.append(i)
    return final

num_games = 100
scores = [0] * num_games
for i_episode in range(num_games):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    move_queue = []
    while not done:
        #env.render(frame_speed=.0001)
        snake_head, food_location = tuple(env.controller.snakes[0].head if env.controller.snakes and env.controller.snakes[0] else [0, 0]), env.controller.grid.foodLocations[0]
        action = None
        # if move_queue:
        #     action = move_queue.pop(0)
        # else:
        #     move_queue = getGreedyAction(snake_head, food_location, deepcopy(env.controller.grid))
        #     action = move_queue.pop(0)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        t += 1
        if done:
            scores[i_episode] = env.controller.score
            print("Episode finished after {} timesteps with reward {} and score {}".format(t, totalReward, env.controller.score))
            break
#Plot average score over previous 500 games
averages = []
for i in range(500, len(scores)):
    averages.append(sum(scores[i-500:i])/float(500))

plt.plot(list(range(500, len(scores))), averages)
plt.ylabel('Average Score over Last 500 Games')
plt.xlabel('Number of Games')
plt.show()
print(sum(scores)/float(len(scores)))

env.close()