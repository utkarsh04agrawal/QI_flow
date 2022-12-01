import numpy as np
import matplotlib.pyplot as pl
import time
import pickle
import os

class State:
    def __init__(self,L:int,p:float,q:int,initial_state) -> None:
        assert L%2 == 0
        self.p = p
        self.q = q
        self.L = L
        # index = np.arange(0,2**L,1)
        self.data = initial_state.copy()
        self.log_Z = []
     


def eventransfer(State,a_structure):
    L = State.L
    os = State.data.reshape((4,)*(L//2))
    for x in range(L//2):
        U = State.U_haar
        # if a_structure[2*x] == -1:
        #     U_1 = State.U_t_1
        # elif a_structure[2*x] == 1:
        #     U_1 = State.U_k_1
        # elif a_structure[2*x] == 0:
        #     U_1 = np.eye(2)
        # if a_structure[2*x+1] == -1:
        #     U_2 = State.U_t_1
        # elif a_structure[2*x+1] == 1:
        #     U_2 = State.U_k_1
        # elif a_structure[2*x+1] == 0:
        #     U_2 = np.eye(2)
        # U = np.kron(U_1,U_2)@U
        os = np.tensordot(U,os,axes=(-1,x))
        os = np.moveaxis(os,0,x)
    os = np.reshape(os,(2,)*L)
    
    for x in range(L):
        if a_structure[x] == -1:
            U_1 = State.U_t_1
        elif a_structure[x] == 1:
            U_1 = State.U_k_1
        elif a_structure[x] == 0:
            continue
        os = np.tensordot(U_1,os,axes=(-1,x))
        os = np.moveaxis(os,0,x)
    State.data = os


def oddtransfer(State,a_structure,BC='PBC'):
    L = State.L
    os = State.data
    os = np.moveaxis(os,0,-1) #bringing 1st spin to the last position
    os = np.reshape(os,(4,)*(L//2))

    for x in range(L//2):
        if x==L//2-1 and BC == 'OBC':
            continue
        U = State.U_haar
        os = np.tensordot(U,os,axes=(-1,x))
        os = np.moveaxis(os,0,x)
    os = np.reshape(os,(2,)*L)
    os = np.moveaxis(os,-1,0) #bringing back the first spin to the starting point.

    for x in range(L):
        if a_structure[x] == -1:
            U_1 = State.U_t_1
        elif a_structure[x] == 1:
            U_1 = State.U_k_1
        elif a_structure[x] == 0:
            continue
        os = np.tensordot(U_1,os,axes=(-1,x))
        os = np.moveaxis(os,0,x)

    State.data = os


def no_of_down_spins(N):
    temp = np.arange(0,2**N,1,dtype=int)
    no_of_ones = np.zeros(2**N,dtype=float)
    for i in range(N):
        no_of_ones += temp%2
        temp = (temp/2).astype(int)
    return N - no_of_ones

def TopLayer(state,no_of_downs):
    """
    This function contracts the evolved state with the top layer.
    """
    ## Caluclate no. of down spins in each configuration
    # no_of_downs = no_of_down_spins(state.L)
    fac = np.sum(state.data.reshape(2**state.L)*((state.q)**(no_of_downs-(state.L)//2)))
    
    return np.log(fac)


def get_U_haar(q):
    U = np.zeros((4,4))
    U[0,0] = 1
    U[3,0] = 0
    U[3,3] = 1
    U[0,3] = 0
    U[3,1] = q/(q**2+1)
    U[3,2] = q/(q**2+1)
    U[0,1] = q/(q**2+1)
    U[0,2] = q/(q**2+1)
    return U

def get_U_t(p,q):
    U = np.zeros((4,4))
    U[0,0] = p**2
    U[3,0] = (1-p**2)/q**2
    U[3,3] = 1
    U[0,3] = 0
    U[3,1] = q*p**2/(q**2+1) + (1-p**2)/q
    U[3,2] = q*p**2/(q**2+1) + (1-p**2)/q
    U[0,1] = p**2*q/(q**2+1)
    U[0,2] = p**2*q/(q**2+1)
    return U

def get_U_k(p,q):
    U = np.zeros((4,4))
    U[0,0] = 1
    U[3,0] = 0
    U[3,3] = p**2
    U[0,3] = (1-p**2)/q**2
    U[0,2] = q*p**2/(q**2+1) + (1-p**2)/q
    U[0,1] = q*p**2/(q**2+1) + (1-p**2)/q
    U[3,2] = p**2*q/(q**2+1)
    U[3,1] = p**2*q/(q**2+1)
    return U


def get_U_t_1(p,q): #this returns transfer matrix for single site channel with ancilla traced out
    U_1 = np.zeros((2,2))
    U_1[1,1] = 1
    U_1[0,0] = p**2
    U_1[1,0] = (1-p**2)/q
    return U_1

def get_U_k_1(p,q): #this returns transfer matrix for single site channel with ancilla kept with the system
    U_1 = np.zeros((2,2))
    U_1[1,1] = p**2
    U_1[0,0] = 1
    U_1[0,1] = (1-p**2)/q
    return U_1


def encoded_bit_state(L,depth,q,BC='up'): 
    """
    This function makes a single LOCAL bell pair a global bit by evolving by Haar dynamics. BC specify whether the initial boundary condition of the bell pair is 'up' or 'down' corresponding to Identity or Swap permutation respectively.
    """
    # up and down correspond to bottom boundary condition for the bell pair.
    
    initial_state = np.zeros((2,)*L)
    if BC == 'up':
        initial_state[1,:] = 1
    elif BC == 'down':
        initial_state[0,:] = 1
    else:
        print("BC parameter is wrong. It can only take 'up' or 'down' value")
    
    state = State(L=L,p=None,q=q,initial_state=initial_state)
    log_Z = []
     # Scrambling
    state.U_haar = get_U_haar(q)
    for t in range(depth):
        start = time.time()
        if t%2 == 0:
            eventransfer(state,[0]*(L))
        else:
            oddtransfer(state,[0]*(L))
        sd = np.sum(state.data)
        log_Z.append(np.log(sd))
        state.data = state.data/sd
    state.log_Z.append(np.sum(log_Z)) # storing the Z (partition function) for the encoding process

    return state


def load_bit_state(L,depth,q,BC='up'):
    """
    Load the state after the scrambling steps.
    """
    filedir = 'data/encoded_bell_pairs'
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    
    filename = filedir + '/L='+str(L)+'_T='+str(depth)+'_q='+str(q)+'_'+BC
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            state = pickle.load(f)
        return state
    
    state = encoded_bit_state(L,depth,q,BC)
    with open(filename,'wb') as f:
        pickle.dump(state,f)
    
    return state


## function to get free energy when the strength of coupling between S and ancillas is same everywhere
def free_energy_uniform(state: State, depth,ancilla_array : np.ndarray, intermediate_time=False,BC='PBC'):
    """
    ancilla_string is a list whose elements are string of length L//2. For the element 't','k','h', the transfer matrix corresponding to having the ancilla traced out, kept in system, no ancilla respectively are applied. 
    """
    p = state.p
    q = state.q
    L = state.L
    assert np.shape(ancilla_array) == (depth,L)
    
    top_layer_factor = []
    ##
    state.U_t_1 = get_U_t_1(p,q)
    state.U_k_1 = get_U_k_1(p,q)
    state.U_haar = get_U_haar(q)

    no_of_downs = no_of_down_spins(L)
    for t in range(depth):
        # start = time.time()
        
        if t%2 == 0:
            eventransfer(state,ancilla_array[t])
        else:
            oddtransfer(state,ancilla_array[t],BC=BC)
        sd = np.sum(state.data)

        if sd==0:
            print(L,p,t,'2')

        state.log_Z.append(np.log(sd))
        state.data = state.data/sd
        
        # print(time.time()-start)
        if intermediate_time:
            top_layer_factor.append(TopLayer(state,no_of_downs))
    if not intermediate_time:
        top_layer_factor.append(TopLayer(state,no_of_downs))
    

    return state,top_layer_factor