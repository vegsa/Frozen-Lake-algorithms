# The Free-energy principle for grid world. 
# Based on the pymdp environment from infer-actively: https://github.com/infer-actively/pymdp 
# And the equations from https://colab.research.google.com/drive/13XfDDh2m-nHf8I_BPbQHhwUGxPgQLpjv?usp=sharing#scrollTo=tpF7GjPx62hb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from pymdp import utils
from pymdp import control
from pymdp.maths import spm_log_single as log_stable # Numerically stable version of np.log() based on the function from MATLAB's SPM library, spm_log.m

from fep_functions import KL_divergence, softmax

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

actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4}

num_actions = len(actions)


# Class that represents the environment
class GridWorld:
    
    def __init__(self, A, B, start_state, dim):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        self.state = np.zeros(dim)
        
        # Initial state. Starting point in grid
        self.state[start_state] = 1
        
        
    # Updates the environment a single step when given an action
    def step(self, a):
        self.state = np.dot(self.B[:,:,a], self.state) # Transition
        obs = utils.sample(np.dot(self.A, self.state)) # Observation. utils.sample extracts the state with value 1.
        return obs 
    
    # Resets the environment to the initial conditions
    def reset(self, dim):
        self.state = np.zeros(dim)
        
        obs = utils.sample(np.dot(self.A, self.state))
        return obs


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


#P maps each state to its next state given an action
def P(xdim, ydim, actions, state_mapping):
    P = {}
    
    for state_index, xy_coordinates in state_mapping.items():
        P[state_index] = {a: [] for a in range(len(actions))}
        x, y = xy_coordinates
        
        # If at the top of the grid, y = 0
        if (y == 0):
            P[state_index][actions['UP']] = state_index
        else: 
            P[state_index][actions['UP']] = state_index - xdim # Move one up
        
        # If all the way to the right, x = xdim - 1
        if (x == (xdim-1)):
            P[state_index][actions['RIGHT']] = state_index
        else:
            P[state_index][actions['RIGHT']] = state_index + 1
            
        # If at the bottom of the grid, y = ydim - 1
        if (y == (ydim-1)):
            P[state_index][actions['DOWN']] = state_index
        else:
            P[state_index][actions['DOWN']] = state_index + xdim
            
        # If all the way to the left, x = 0
        if (x == 0):
            P[state_index][actions['LEFT']] = state_index
        else: 
            P[state_index][actions['LEFT']] = state_index - 1
            
        # Stay in the same place
        P[state_index][actions['STAY']] = state_index
        
    return P


# Transition matrix B. Determines how the agent can move.
# Will be a 9x9x5 matrix. Corresponds to an end state, a starting state, and the action that defines the transition
# p(s_t | s_(t-1), a_(t-1))
def transition_matrix(num_states, num_actions, P):
    B = np.zeros([num_states, num_states, num_actions])
    
    for s in range(num_states):
        for a in range(num_actions):
            new_state = int(P[s][a])
            B[new_state, s, a] = 1
    
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
        
        """
        # Find predicted divergence
        divergence = KL_divergence(Qs_pi, C)

        # Expected uncertainty
        H = -(Qs_pi*log_stable(A)).sum(axis=1)
        uncertainty = Qo_pi.dot(H)
        
        """
        H = - (A * log_stable(A)).sum(axis = 0)

        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = KL_divergence(Qo_pi, C)
        
        # compute the expected uncertainty or ambiguity 
        uncertainty = H.dot(Qs_pi)
        
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
    
    
    # Sample control from the marginal probability of the action with larget probability.
    print(P_a)
    
    
    # Issues with calculating the correct P_a. Try with another calculation of G.
    u = np.argmax(P_a)
        
    #u = utils.sample(P_a)

    # Print action.
    key_list = list(actions.keys())
    val_list = list(actions.values())
    pos = val_list.index(u)
    print("Action", key_list[pos])
    
    return u

def active_inference(env, A, B, C, Qs, start_state):
    # Number of time steps
    T = 10
    
    num_actions = 5
    
    # Length of policies we consider
    policy_len = 6
    
    # Generate all possible combinations of policies.
    policies = control.construct_policies([B.shape[0]], [num_actions], policy_len)
    
    print("Starting state:", env.state)
    
    for t in range(T):
        
        # Infer with action to take
        a = infer_action(Qs, A, B, C, num_actions, policies)
        
        # Perform action in environment and update the environment
        o = env.step(int(a))
        
        # Infer the new hidden state
        likelihood = A[o,:]
        prior = B[:,:,int(a)].dot(Qs)
        
        Qs = softmax(log_stable(likelihood) + log_stable(prior))
        
        print(Qs.round(3))
        
        plot_beliefs(Qs, "Belief (Qs) at time {}".format(t))
        print("\n")

       

# Likelihood matrix A. Represents p(o_t | x_t) = p(s | external states)
# In a grid world, we assume that the likelihood matrix of the agent is identical to that of the environment.
# The agent has probability 1 of observing that it is occupying any state x, given that it is in state x.
# The agent has full transparency over its own location. 
A = np.eye(xdim*ydim)

# Transition matrix B.
P = P(xdim, ydim,actions,state_mapping)
B = transition_matrix(num_states, num_actions, P)

# Desired state vector, C. The goal vector.
REWARD_LOCATION = 15
reward_state = state_mapping[REWARD_LOCATION]

C = np.zeros(num_states)
C[REWARD_LOCATION] = 1
plot_beliefs(C, "Desired location")


# Initialize grid world.
start_state = 1
env = GridWorld(A,B, start_state, xdim*ydim)

# Prior beliefs. The agent is uncertain of which state it is in. q(x_0 | o_0)
Qs = np.ones(num_states)*1/num_states

active_inference(env,A,B,C,Qs, start_state)

# Reset environment.
env.reset(xdim*ydim)

