import numpy as np
from pymdp.maths import spm_log_single as log_stable # Numerically stable version of np.log() based on the function from MATLAB's SPM library, spm_log.m

def KL_divergence(q, p):
    return np.sum(q*(log_stable(q) - log_stable(p)))
    
def free_energy(q, A, B):
    return np.sum(q*(log_stable(q)-log_stable(A)-log_stable(B)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Likelihood is not the entire A matrix, but the row of the A matrix corresponding to the current observation, P(o_t = o | x_t)
def perform_inference(likelihood, prior):   # q*
    return softmax(log_stable(likelihood) + log_stable(prior))

