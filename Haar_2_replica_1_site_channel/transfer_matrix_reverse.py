"""
This code implements the transfer matrix in reverse direction. This is implemented to check the correntness of the other code.
"""

import numpy as np
import matplotlib.pyplot as pl
import time
import pickle
import os
# import scipy.sparse as sparse

class State:
    def __init__(self,L:int,p:float,q:int) -> None:
        assert L%2 == 0
        self.p = p
        self.q = q
        self.L = L
        # index = np.arange(0,2**L,1)
        self.log_Z = [0]
        self.data = np.zeros((2,)*L)
        self.data[(0,)*L] = 1 #initial state, that is the top boundary is always all down


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

def TopLayer(state,x): #target =0,1 specify the boundary condition of the Bell pair at position x.
    """
    This function contracts the evolved state with the top layer.
    """
    temp = np.moveaxis(state.data,x,0)
    fac_1 = state.q**2*np.sum(temp[1,:]) + state.q*np.sum(temp[0,:]) # this is for Bell pair having up spin.
    fac_0 = state.q**2*np.sum(temp[0,:]) + state.q*np.sum(temp[1,:]) # this is for Bell pair having down spin.
    return np.log(fac_1), np.log(fac_0) 


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


## function to get free energy when the strength of coupling between S and ancillas is same everywhere
def free_energy_uniform(state: State,depth,ancilla_array : np.ndarray,bell_pair_position:int, intermediate_time=False,BC='PBC'):
    """
    ancilla_string is a list whose elements are string of length L//2. For the element 't','k','h', the transfer matrix corresponding to having the ancilla traced out, kept in system, no ancilla respectively are applied. 
    """
    p = state.p
    q = state.q
    L = state.L
    top_layer_factor = []
    ##
    state.U_t_1 = get_U_t_1(p,q)
    state.U_k_1 = get_U_k_1(p,q)
    state.U_haar = get_U_haar(q)

    p_cross = [] #prob. for DW to cross across the system starting from left boundary (with dissipation) to the right boundary without disspipation

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
        p_cross.append(state.data[(1,)*L])
        # print(time.time()-start)
        if intermediate_time:
            top_layer_factor.append(TopLayer(state,bell_pair_position))
    if not intermediate_time:
        top_layer_factor.append(TopLayer(state,bell_pair_position))
    

    return state,top_layer_factor,p_cross