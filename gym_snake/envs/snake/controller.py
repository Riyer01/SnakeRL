from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np
import random

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)
        self.score = 0

        self.snakes = []
        self.dead_snakes = []
        for i in range(1,n_snakes+1):
            #start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            start_coord = [random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1)]
            self.snakes.append(Snake(start_coord, snake_size))
            color = [self.grid.HEAD_COLOR[0], i*10, 0]
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.score += 1
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 1
            self.grid.foodLocations.remove(tuple(snake.head))
            self.grid.new_food()
        else:
            reward = -.1
            food = self.grid.foodLocations[0]
            # distToFood = abs(snake.head[0] - food[0]) + abs(snake.head[1] - food[1])
            # reward = -.2*distToFood
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def generateObservationTuple(self, direction):
        #Observation = [danger up, danger left, danger right, snake up, snake down, snake left,
        # snake right, (food_up, food_down)]
        observation = []
        if self.snakes and self.snakes[0]:
            directionMap = {0:[0, 3, 1], 1:[1, 0, 2], 2:[2, 1, 3], 3:[3, 2, 0]}
            for relativeDirec in directionMap[self.snakes[0].direction]:
                    observation.append(int(self.grid.check_death(self.snakes[0].step(self.snakes[0].head, relativeDirec))))

            for direction in range(4):
                if self.snakes[0].direction == direction:
                    observation.append(1)
                else:
                    observation.append(0)

            snake_to_fruit = np.sign([self.snakes[0].head[0] - self.grid.foodLocations[0][0], self.snakes[0].head[1] - self.grid.foodLocations[0][1]])
            if snake_to_fruit[0] == 1:
                observation = observation + [0, 1]
            elif snake_to_fruit[0] == -1:
                observation = observation + [1, 0]
            else:
                observation = observation + [0, 0]
            if snake_to_fruit[1] == 1:
                observation = observation + [0, 1]
            elif snake_to_fruit[1] == -1:
                observation = observation + [1, 0]
            else:
                observation = observation + [0, 0]
            #observation.append(tuple(snake_to_fruit))
        else:
            #observation = [0] * 7 + [(0, 0)]
            observation = [0] * 11
        
        # if self.snakes and self.snakes[0]:
        #     head = self.snakes[0].head
        #     last_tail = self.snakes[0].body[-1]
        #     food = self.grid.foodLocations[0]
        #     return ((last_tail[0] - head[0], last_tail[1] - head[1]), (food[0] - head[0], food[1] - head[1]))
        # else:
        #     return ((0, 0), (0, 0))
        return observation

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) is 1:
                return self.grid.grid.copy(), 0, True, {"snakes_remaining":self.snakes_remaining}
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, {"snakes_remaining":self.snakes_remaining}

        rewards = []

        if type(directions) == type(int()):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction,i)
            rewards.append(self.move_result(direction, i))

        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        # #Observation = [danger left, danger right, danger straight, snake up, snake down, snake left,
        # # snake right, (food_up, food_down)]
        # observation = []
        # if self.snakes and self.snakes[0]:
        #     for direction in range(4):
        #         if np.abs(self.snakes[0].direction-direction) != 2:
        #             observation.append(int(self.grid.off_grid(self.snakes[0].step(self.snakes[0].head, direction))))
        #         if self.snakes[0].direction == direction:
        #             observation.append(1)
        #         else:
        #             observation.append(0)
        #     snake_to_fruit = np.sign([self.snakes[0].head[0] - self.grid.foodLocations[0][0], self.snakes[0].head[1] - self.grid.foodLocations[0][1]])
        #     observation.append(tuple(snake_to_fruit))
        # else:
        #     observation = [0] * 7 + [(0, 0)]
        #observation = [tuple(self.snakes[0].head) if self.snakes and self.snakes[0] else None, self.grid.foodLocations[0]]
        observation = self.generateObservationTuple(direction)
        if len(rewards) is 1:
            #used to be self.grid.grid.copy()
            return tuple(observation), rewards[0], done, {"snakes_remaining":self.snakes_remaining}
        else:
            return tuple(observation), rewards, done, {"snakes_remaining":self.snakes_remaining}
