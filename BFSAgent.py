import gym
import gym_snake
from hamilton import Hamilton
from copy import deepcopy

env = gym.make('snake-v0')
observation = env.reset()

def simulate_move_snake(direction, snake, grid):
    """
    Moves the specified snake according to the game's rules dependent on the direction.
    Does not draw head and does not check for reward scenarios. See move_result for these
    functionalities.
    """

    if type(snake) == type(None):
        return

    # Cover old head position with body
    grid.cover(snake.head, grid.BODY_COLOR)
    # Erase tail without popping so as to redraw if food eaten
    grid.erase(snake.body[0])
    # Find and set next head position conditioned on direction
    snake.action(direction)

def simulate_move_result(direction, snake, grid):
    """
    Checks for food and death collisions after moving snake. Draws head of snake if
    no death scenarios.
    """

    if type(snake) == type(None):
        return 0

    # Check for death of snake
    if grid.check_death(snake.head):
        grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
        grid.connect(snake.body.popleft(), snake.body[0], grid.SPACE_COLOR)
        reward = -1
    # Check for reward
    elif grid.food_space(snake.head):
        grid.draw(snake.body[0], grid.BODY_COLOR) # Redraw tail
        grid.connect(snake.body[0], snake.body[1], grid.BODY_COLOR)
        grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
        reward = 1
        grid.foodLocations.remove(tuple(snake.head))
        grid.new_food()
    else:
        reward = 0
        empty_coord = snake.body.popleft()
        grid.connect(empty_coord, snake.body[0], grid.SPACE_COLOR)
        grid.draw(snake.head, snake.head_color)

    grid.connect(snake.body[-1], snake.head, grid.BODY_COLOR)

    return reward
def simulate_step(direction, snake, grid):
    """
    Takes an action for each snake in the specified direction and collects their rewards
    and dones.
    """
    simulate_move_snake(direction, snake, grid)
    return simulate_move_result(direction, snake, grid)

def getAction(snake, grid):
    bfs_queue = [(snake, grid)] # Add the first state
    visited = [[0 for x in range(8)] for y in range(8)]
    print('Body: ', snake.body)
    print('Head: ', snake.head)

    # shortest_path = []
    # while bfs_queue:
    #     cur_state = bfs_queue.pop(0)
    #     for direc in range(4):
    #         if (): # check if already visited
    #             bfs_queue.append()
    #             visited.append()
    # return shortest_path


for i_episode in range(20):
    observation = env.reset()
    totalReward = 0
    done = False
    t = 0
    while not done:
        env.render()
        action = getAction(deepcopy(env.controller.snakes[0]), deepcopy(env.controller.grid))
        observation, reward, done, info = env.step(1)
        totalReward += reward
        t += 1
        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, totalReward))
            break
env.close()
