# The Free-energy principle for Frozen lake. 
# Based on the pymdp environment from infer-actively: https://github.com/infer-actively/pymdp 

# Full observability and known transition dynamics.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pymdp import control
from pymdp.maths import spm_log_single as log_stable # Numerically stable version of np.log() based on the function from MATLAB's SPM library, spm_log.m

from fep_functions import KL_divergence, softmax

import gym

# Generative model
    
# Actions: LEFT, RIGHT, UP, DOWN, STAY

xdim = 4
ydim = 4

# 4x4:
state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (3,0), 4: (0,1), 5: (1,1), 6: (2,1), 7: (3,1), 8: (0,2), 9: (1,2), 10: (2,2), 11: (3,2), 12: (0,3), 13: (1,3), 14: (2,3), 15: (3,3)}

#3x3:
#state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5: (2,1), 6: (0,2), 7: (1,2), 8: (2,2)}

#4x3
#state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (3,0), 4: (0,1), 5: (1,1), 6: (2,1), 7: (3,1), 8: (0,2), 9: (1,2), 10: (2,2), 11: (3,2)}


num_states = len(state_mapping)

actions = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3}

num_actions = len(actions)

# Visualization of grid space
def grid_visualization(state_mapping): 
    grid = np.zeros((ydim,xdim))
    for linear_index, xy_coordinates in state_mapping.items():
        x, y = xy_coordinates
        grid[y,x] = linear_index
        
    fig = plt.figure(figsize = (3,3))
    sns.set(font_scale=1.5)
    sns.heatmap(grid, annot=True,  cbar = False, fmt='.0f', cmap='crest')

grid_visualization(state_mapping) 

labels = [state_mapping[i] for i in range(num_states)]
# Plot of linkelihood matrix
def plot_likelihood(A):
    fig = plt.figure(figsize=(6,6))
    ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
    plt.title("Likelihood distribution (A)")
    plt.show()
    
#plot_likelihood(A)

# Plots transitions corresponding to the different actions.
# Add xdim and ydim here if necessary 
def plot_transition(actions, xdim, ydim):
    fig, axes = plt.subplots(2,3, figsize = (15,8))
    a = list(actions.keys())
    count = 0
    for i in range(ydim-1):
        for j in range(xdim):
            if count >= 5:
                break 
            g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(a[count])
            count +=1 
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()
    
#plot_transition(actions, xdim, ydim)

def plot_beliefs(Qs, title=""):
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()

#plot_beliefs(Qs)

def learning_transition(env):
    n_episodes = 5000
    max_iterations = 100
    
    B = np.zeros([env.env.nS, env.env.nA, env.env.nS])
    b = np.zeros([env.env.nA, env.env.nS, env.env.nS])
    
    for e in range(n_episodes):
        
        state = env.reset()
        done = False
        for i in range(max_iterations):
            
            action = env.action_space.sample()
            
            new_state, _, done, _ = env.step(int(action))
            
            for a in range(env.env.nA):
                b[a, :, :] += 0.125*np.kron(state, new_state)
            
            if done:
                break
            
            state = new_state
            
        for i in range(env.env.nA):
            for j in range(env.env.nS):
                B[j,i,:] = softmax(b[i,:,j])
            
    return B
    

# Transition matrix B. Determines how the agent can move.
# Will be a 9x9x5 matrix. Corresponds to an end state, a starting state, and the action that defines the transition
# p(s_t | s_(t-1), a_(t-1))
def transition_matrix(env):
    B = np.zeros([env.env.nS, env.env.nS, env.env.nA])
    
    for s in range(env.env.nS):
        for a in range(env.env.nA):
            P = np.array(env.env.P[s][a])
            (x,y) = np.shape(P)
            for i in range(x):
                new_state = int(P[i][1])
                B[new_state, s, a] = P[i][0]
    
    return B
 

# Evaluate the negative expected free energy for a given policy, -G(pi).
def expected_free_energy(policy, Qs, A, B, C):
    # Initialization of expected free energy (EFE).
    G = 0
    # Loop through the policy.
    Qs_pi = Qs
    for t in range(len(policy)):
        
        # Get the action from the policy at timestep t
        u = int(policy[t])
        
        # Expected state, given policy. q(x_t|pi)
        
        Qs_pi = np.dot(B[:, :, u],Qs_pi)
        
        # Expected observation, given policy. q(o_t | pi)
        Qo_pi = A.dot(Qs_pi)
        
        
        # Fix this for is_slippery = True?
        # Find predicted divergence
        H = - (A * log_stable(A)).sum(axis = 0)

        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = KL_divergence(Qo_pi, C)
        
        # compute the expected uncertainty or ambiguity 
        uncertainty = Qs_pi.dot(H) #H.dot(Qs_pi)
        
        """
        
        # Find predicted divergence
        divergence = KL_divergence(Qs_pi, C)

        # Expected uncertainty
        H = -(Qs_pi*log_stable(A)).sum(axis=1)
        uncertainty = Qo_pi.dot(H)
        """
        
        G += (divergence + uncertainty)

    # To minimize G, find the maximum of -G.
    return -G


# Inference of the most likely action.
def infer_action(Qs, A, B, C, num_actions, policies):
    
    # Initialization of the negative EFE.
    neg_G = np.zeros(len(policies))
    
    # Loop over every possible policy and compute EFE of each policy.
    for i, policy in enumerate(policies):
        neg_G[i] = expected_free_energy(policy, Qs, A, B, C)
    
    # Get the distribution over policies
    Q_pi = softmax(neg_G)
    
    # Initialize probabilities of action states. Convert from policies to actions.
    P_a = np.zeros(num_actions)
    
    # Sum probabilities of control states or actions.
    for i, policy in enumerate(policies):
        
        # Control state given by policy.
        u = int(policy[0])
        
        # Add probability of the policy
        P_a[u] += Q_pi[i]
    
    
    # Sample control from the marginal probability of the action with largest probability.
    print(P_a)
    #u = utils.sample(P_a)
    
    u = np.argmax(P_a)

    # Print action.
    key_list = list(actions.keys())
    val_list = list(actions.values())
    pos = val_list.index(u)
    print("Action", key_list[pos])
    
    return u

def active_inference(env, A, B, C, Qs):
    # Number of time steps
    T = 10
    
    # Length of policies we consider
    policy_len = 6
    
    # Generate all possible combinations of policies.
    policies = control.construct_policies([B.shape[0]], [env.env.nA], policy_len)

    env.reset()
    
    for t in range(T):
        
        # Infer with action to take
        a = infer_action(Qs, A, B, C, num_actions, policies)
        # Perform action in environment and update the environment
        o, _, done, _ = env.step(int(a))

        print("o: ", o)
        # Infer the new hidden state
        likelihood = A[o,:]
        prior = B[:,:,int(a)].dot(Qs)
        
        # Optimal approximate posterior
        Qs = softmax(log_stable(likelihood) + log_stable(prior))
        
        print(Qs.round(3))
        
        plot_beliefs(Qs, "Belief (Qs) at time {}".format(t))
        print("\n")
        
        if done: 
            break


# Make a function that iterates through every state
# and infer the action to take in that position

def find_policy(env, A, B, C, Qs):
    # Length of policies we consider
    policy_len = 3 # If policy_len > 3 it will not reach the goal at all if is_slippery = True
                   # If is_slippery = False, policy_len must be equal to 6 or larger
    
    # Generate all possible combinations of policies.
    policies = control.construct_policies([B.shape[0]], [env.env.nA], policy_len)

    env.reset()
    
    policy = np.zeros(env.env.nS)
    
    for t in range(env.env.nS):
        a = infer_action(Qs, A, B, C, num_actions, policies)
        
        policy[t] = int(a)
        # Perform action in environment and update the environment
        #o, _, done, _ = env.step(int(a))
        #print("o: ", o)
        
        # Infer the new hidden state
        #likelihood = A[o,:]
        #prior = B[:,:,int(a)].dot(Qs)
        
        # Optimal approximate posterior
        #Qs = softmax(log_stable(likelihood) + log_stable(prior))

        if (t+1) == 16:
            break
        Qs = A[t+1,:]*np.ones(env.env.nS)
        
        #print(Qs.round(3))
        
        #plot_beliefs(Qs, "Belief (Qs) at time {}".format(t))
        #print("\n")
        
        #if done: 
        #    break

    return policy        


env = gym.make("FrozenLake-v1", is_slippery=True)
start_state = env.reset()

# Likelihood matrix A. Represents p(o_t | x_t) = p(s | external states)
# In a grid world, we assume that the likelihood matrix of the agent is identical to that of the environment.
# The agent has probability 1 of observing that it is occupying any state x, given that it is in state x.
# The agent has full transparency over its own location. 
A = np.eye(env.env.nS)

# Transition matrix B.
#P = P(xdim, ydim,actions,state_mapping)
#print("P: ", P)
B = transition_matrix(env)
#print(B[:,:,1])

# Desired state vector, C. The goal vector.
REWARD_LOCATION = 15
reward_state = state_mapping[REWARD_LOCATION]

C = np.zeros(num_states)
C[REWARD_LOCATION] = 1
plot_beliefs(C, "Desired location")


# Prior beliefs. The agent is uncertain of which state it is in. q(x_0 | o_0)
#Qs = np.ones(num_states)*1/num_states
Qs = np.zeros(env.env.nS)
Qs[start_state] = 1


pol = find_policy(env, A, B, C, Qs)
policy = np.reshape(pol, (4,4))
print(policy)

#active_inference(env,A,B,C,Qs)

"""
print(policy)
def print_policy(policy, actions):
    pol_written = []
    key_list = list(actions.keys())
    val_list = list(actions.values())
    for u in range(len(policy)):
        pos = val_list.index(policy[u])
        pol_written.append(key_list[pos])
        
    print(pol_written)
    
print_policy(policy, actions)
"""

e = 0
for i in range(100):
    u = env.reset()
    for t in range(500):
        u, reward, done, info = env.step(pol[u])
        if done:
            if reward == 1:
                e += 1
            break
        
print(e)
        
print("The agent succeeded to reach goal {} out of 100 episodes using this policy".format(e))

