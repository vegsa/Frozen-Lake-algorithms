
import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=True) # is_slippery=True if stochastic environment

gamma = 0.99
alpha = 0.1

def qlearning(env):
    n_episodes = 10000
    max_iterations = 100
    
    explore = 1
    min_explore = 0.01
    
    Q_table = np.zeros((env.env.nS, env.env.nA))
    
    for e in range(n_episodes):
        
        state = env.reset()
        done = False
        
        for i in range(max_iterations):
            
            if np.random.uniform(0,1) < explore:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state, :])
                
            next_state, reward, done, _ = env.step(action)
            
            TD = reward + gamma*np.max(Q_table[next_state, :]) - Q_table[state, action]

            Q_table[state, action] += alpha*TD
                        
            if done:
                break
            state = next_state
        
        explore = max(min_explore, np.exp(-0.001*e))
            
    return Q_table

#print(env.render())
Q_table = qlearning(env)
print(Q_table.round(2))

pol = np.zeros(env.env.nS)
for i in range(len(Q_table)):
    pol[i] = np.argmax(Q_table[i])
    
actions = np.reshape(pol, (4,4))
print(actions)

e = 0
for i in range(100):
    u = env.reset()
    for t in range(500):
        u, reward, done, info = env.step(pol[u])
        if done:
            if reward == 1:
                e += 1
            break
        
print("The agent succeeded to reach goal {} out of 100 episodes using this policy".format(e))
