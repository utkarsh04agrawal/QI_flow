import numpy as np

L = 10


def evolve_layer(state,t,q):
    N = len(state)
    if t%2==0: # even layer
        x_list = np.array(range(N)[1::2])
        weight = q/(q**2+1) * state[x_list]
        state[x_list-1] += weight
        state[x_list+1] += weight
        state[x_list] = 0 
    else: # odd layer
        x_list = np.array(range(N-2)[2::2])
        weight = q/(q**2+1) * state[x_list]
        state[x_list-1] += weight
        state[x_list+1] += weight
        state[x_list] = 0

    # return state


def evolve_dissipation(state,p,q):
    state[1] += ((1-p**2)/q) * state[0]
    state[0] = (p**2) * state[0]


def state_evolution(L,T,q,p,ancilla_structure):
    """
    L: system size
    T: time for which to run the dynamics
    ancilla_structure: a list. 0 at index t means no channel at the boundary at time t; element 1 implies a channel.
    """
    state = np.zeros(L+1) # state stores the weight corresponding to DW being at x=0,..,L
    state[0] = 1 # initially there is no DW, that is the DW is at x=0

    theta_array = np.zeros((T,L)) # this stores probability of domain wall being at position x
    for t in range(T):
        evolve_layer(state,t,q)
        if ancilla_structure[t] == 1:
            evolve_dissipation(state,p,q)
        state = state/np.sum(state) # normalizing to stop the elements becoming too small
        theta_array[t,:] = np.cumsum(state)[:-1]
    
    return state, theta_array


print(state_evolution(L=10,T=10,q=1,p=0,ancilla_structure=[1]*10))





