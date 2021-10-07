# Active inference using the free energy principle for Frozen lake. 
# Based on the pymdp environment from infer-actively: https://github.com/infer-actively/pymdp 

# Full observability and known transition dynamics.

import numpy as np
import matplotlib.pyplot as plt

from pymdp import control
from pymdp.maths import spm_log_single as log_stable # Numerically stable version of np.log() based on the function from MATLAB's SPM library, spm_log.m

import gym

    
# Actions: LEFT, RIGHT, UP, DOWN, STAY
actions = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3}


def plot_beliefs(Qs, title=""):
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()

def KL_divergence(q, p):
    return np.sum(q*(log_stable(q) - log_stable(p)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

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
        a = infer_action(Qs, A, B, C, env.env.nA, policies)
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
        a = infer_action(Qs, A, B, C, env.env.nA, policies)
        
        policy[t] = int(a)

        if (t+1) == 16:
            break
        Qs = A[t+1,:]*np.ones(env.env.nS)   #Approximate posterior.

    return policy        


env = gym.make("FrozenLake-v1", is_slippery=True)
start_state = env.reset()

# Likelihood matrix A. Represents p(o_t | x_t) = p(s | external states)
# In a grid world, we assume that the likelihood matrix of the agent is identical to that of the environment.
# The agent has full transparency over its own location. 
A = np.eye(env.env.nS)

# Transition matrix B.
B = transition_matrix(env)

# Desired state vector, C. The goal vector.
REWARD_LOCATION = 15

C = np.zeros(env.env.nS)
C[REWARD_LOCATION] = 1
plot_beliefs(C, "Desired location")


# Prior beliefs.
Qs = np.zeros(env.env.nS)
Qs[start_state] = 1

# Perform active inference.
#active_inference(env,A,B,C,Qs)

pol = find_policy(env, A, B, C, Qs)
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