import gym
import numpy as np
import copy

env = gym.make('FrozenLake-v1', is_slippery=True)

DISCOUNT = 0.99
EPSILON = 0.00001

def find_policy(env, V, pi, s):
    e = np.zeros(env.env.nA)
    for a in range(env.env.nA):
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        for i in range(x):
            s_next = int(P[i][1]) #s'
            p = P[i][0]
            r = P[i][2]
    
            e[a] += p*(r + DISCOUNT*V[s_next])
            
    m = np.argmax(e)
    pi[s][m] = 1
    return pi
    

def sum_utility(env, V, s):
    Vsum = np.zeros(env.env.nA) 
    
    for a in range(env.env.nA):
        P = np.array(env.env.P[s][a])
        (x,y) = np.shape(P)
        for i in range(x):
            s_next = int(P[i][1]) #s'
            p = P[i][0]
            r = P[i][2]
    
            Vsum[a] += p*(r + DISCOUNT*V[s_next])
    
    maxV = 0
    
    m = np.argmax(Vsum)
    maxV = Vsum[m]
    
    
    return maxV


def value_iteration(env):

    V = np.zeros(env.env.nS)

    delta = 1

    while delta > EPSILON:
        v = copy.deepcopy(V)
        delta = 0
        for s in range(env.env.nS):
            V[s] = sum_utility(env, v, s)
        delta = max(delta, np.max(np.abs(v-V)))
        
        
    return V


V = value_iteration(env)
print("Values: ", V)

# Find policy.
pi = np.zeros((env.env.nS, env.env.nA))
for s in range(env.env.nS):
    pi = find_policy(env, V, pi, s)
    
pol = np.zeros(env.env.nS)

for i in range(len(pi)):
    for k in range(len(pi[0])):
        if (pi[i][k] == 1):
            pol[i] = k

# Print policy.
policy = np.reshape(pol, (4,4))
print(policy)

def test_policy(env, policy):
    e = 0
    for i in range(100):
        state = env.reset()
        for t in range(500):
            state, reward, done, _ = env.step(policy[state])
            if done:
                if reward == 1:
                    e += 1
                break
            
            
    print("The agent succeeded to reach goal {} out of 100 episodes using this policy".format(e))


test_policy(env, pol)