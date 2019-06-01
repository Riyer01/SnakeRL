import gym
import gym_snake
import numpy as np

env = gym.make('snake-v0')
observation = env.reset()

eta = .628
gma = .9
epis = 5000
rev_list = [] # rewards per episode calculate
Q = {}

for i in range(epis):
    # Reset environment
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        env.render()
        j+=1
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1,r,d, _ = env.step(int(a))
        #Update Q-Table with new knowledge
        Q[s][a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rev_list.append(rAll)
    env.render()
env.close()