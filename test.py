#!/bin/env python 

import numpy as np
import os,sys,inspect


import tensorflow as tf
import numpy as np
import scipy.linalg as la

#from core.Convergence import Convergence
#from core.run_session import run_session


import random as rd
import time
import math

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from time import sleep
from numpy.random import random_sample
from sync_opt import SyncReplicasOptimizer as sync


def get_avg(av,average_path,iterations):
    #Averaging trajectory gradients
    a1 = iterations * average_path
    a2 = np.add(av,a1)
    return np.divide(a2,(iterations + 1))

def dressed_unitary(U,v,dressed_id):
    # get unitary matrix in dressed basis
    conversion_U = sort_ev(v,dressed_id)
    return np.dot(np.dot(conversion_U,U),np.conjugate(np.transpose(conversion_U)))

def get_dressed_info(H0):
    # assign index of the dressed state according to the overall with bare state
    w_c, v_c = la.eig(H0)
    dressed_id=[]
    for ii in range(len(v_c)):
        index = np.argmax(np.abs(v_c[:, ii]))
        if index not in dressed_id:
            dressed_id.append(index)
        else:
            temp = (np.abs(v_c[:, ii])).tolist()
            while index in dressed_id:
                temp[index] = 0
                index = np.argmax(temp)
            dressed_id.append(index)
            
    return w_c, v_c, dressed_id

def qft(N):
    # quantum fourier transform operator
    phase = 2.0j * np.pi / (2**N)
    L, M = np.meshgrid(np.arange(2**N), np.arange(2**N))
    L = np.exp(phase * (L * M))
    q = 1.0 / np.sqrt(2**N) * L
    return q
    
def hamming_distance(x):
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot

def Hadamard (N=1):
    # Hadamard gate
    Had = (2.0 ** (-N / 2.0)) * np.array([[((-1) ** hamming_distance(i & j))
                                      for i in range(2 ** N)]
                                     for j in range(2 ** N)])
    return Had

def concerned(N,levels):
    concern = []
    for ii in range (levels**N):
        ii_b = Basis(ii,N,levels)
        if is_binary(ii_b):
            concern.append(ii)
    return concern
        
def is_binary(num):
    flag = True
    for c in num: 
        if c!='0' and c!='1':
            flag = False
            break
    return flag
    
def transmon_gate(gate,levels):
    N = int(np.log2(len(gate)))
    result = np.identity(levels**N,dtype=complex)
    for ii in range (len(result)):
        for jj in range(len(result)):
            ii_b = Basis(ii,N,levels)
            jj_b = Basis(jj,N,levels)
            if is_binary(ii_b) and is_binary(jj_b):
                result[ii,jj]=gate[int(ii_b, 2),int(jj_b, 2)]
                
    return result
def rz(theta):
    return [[np.exp(-1j * theta / 2), 0],[0, np.exp(1j * theta / 2)]]
def rx (theta):
    return [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]]


def Bin(a,N):
    a_bin = np.binary_repr(a)
    while len(a_bin) < N:
        a_bin = '0'+a_bin
    return a_bin

def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

def Basis(a,N,r):
    a_new = baseN(a,r)
    while len(a_new) < N:
        a_new = '0'+a_new
    return a_new
    
    
def kron_all(op,num,op_2): 
    # returns an addition of sth like xii + ixi + iix for op =x and op_2 =i
    total = np.zeros([len(op)**num,len(op)**num])
    a=op
    for jj in range(num):
        if jj != 0:
            a = op_2
        else:
            a = op
            
        for ii in range(num-1):
            if (jj - ii) == 1:
                
                b = op
            else:
                b = op_2
            a = np.kron(a,b)
        total = total + a
    return a    

def multi_kron(op,num): 
    #returns xx...x
    a=op
    for ii in range(num-1):
        a = np.kron(a,op)
    return a

def append_separate_krons(op,name,num,state_num,Hops,Hnames,ops_max_amp,amp=4.0): 
    #appends xii,ixi,iix separately
    string = name
    I_q = np.identity(state_num)
    x = 1
    y = 1
    z = 1
    X1 = op
    while(x < num):
        X1 = np.kron(X1, I_q)
        x = x + 1
    Hops.append(X1)
    ops_max_amp.append(amp)
    x = 1
    while(x < num):
        string = string + 'i'
        x = x+1
    Hnames.append(string)

    x = 1

    while(x < num):
        X1 = I_q
        string = 'i'
        while(y<num):
            if(y==x):
                X1 = np.kron(X1, op)
                y = y + 1
                string = string + name
            else:
                X1 = np.kron(X1, I_q)
                y = y + 1
                string = string + 'i'
        x = x + 1
        y=1
        Hops.append(X1)
        ops_max_amp.append(amp)
        Hnames.append(string)
    return Hops,Hnames,ops_max_amp

def nn_chain_kron(op, op_I, qubit_num, qubit_state_num): 
    # nearest neighbour kron: e.g. xxii + ixxi + iixx
    op_list = ['I']*(qubit_num-2)
    op_list = ['OP','OP'] + op_list
    
    
    a_all = np.zeros([qubit_state_num**qubit_num,qubit_state_num**qubit_num])
    for ii in range(qubit_num-1):
        
        if op_list[0] == 'I':
            a = op_I
        else:
            a = op
            
        for kk in range(1,qubit_num):
            if op_list[kk] == 'I':
                b = op_I
            else:
                b = op
            a = np.kron(a,b)
        
        a_all = a_all + a
        
        op_list = [op_list[-1]] + op_list[:-1]
        
    
    return a_all


def sort_ev(v,dressed_id):
    # sort the eigenstates according to bare states
    v_sorted=[]
    
    for ii in range (len(dressed_id)):
        v1 = v[:,get_state_index(ii,dressed_id)]
        v_sorted.append(v1)
    
    return np.transpose(np.reshape(v_sorted, [len(dressed_id),len(dressed_id)]))

def get_state_index(bareindex,dressed_id):
    # get the index of dressed state, with maximum overlap with the corresponding bare state
    if len(dressed_id) > 0:
        return dressed_id.index(bareindex)
    else:
        return bareindex
    
def c_to_r_mat(M):
    # complex to real isomorphism for matrix
    return np.asarray(np.bmat([[M.real,-M.imag],[M.imag,M.real]]))

def c_to_r_vec(V):
    # complex to real isomorphism for vector
    new_v =[]
    new_v.append(V.real)
    new_v.append(V.imag)
    return np.reshape(new_v,[2*len(V)])
        



class SystemParameters:

    def __init__(self,H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxA, draw,initial_guess, show_plots,Unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms,use_gpu,use_inter_vecs,sparse_H,
                sparse_U,sparse_K, c_ops,trajectories, do_all, expect_op):
        # Input variable
        self.sparse_U = sparse_U
        self.sparse_H = sparse_H
        self.sparse_K = sparse_K
        self.use_inter_vecs = use_inter_vecs
        self.use_gpu = use_gpu
        self.Taylor_terms = Taylor_terms
        self.dressed_info = dressed_info
        self.reg_coeffs = reg_coeffs
        self.file_path = file_path
        self.state_transfer = state_transfer
        self.no_scaling = no_scaling
        self.save = save
        self.H0_c = H0
        self.ops_c = Hops
        self.ops_max_amp = maxA
        self.Hnames = Hnames
        self.Hnames_original = Hnames #because we might rearrange them later if we have different timescales 
        self.total_time = total_time
        self.steps = steps
        self.show_plots = show_plots
        self.Unitary_error= Unitary_error
        self.trajectories = trajectories
        self.c_ops = c_ops
        self.traj = False
        self.do_all = do_all
        self.expect_op = expect_op
        self.expect = False
        if self.expect_op != []:
            self.expect = True


        if initial_guess is not None:
            # transform initial_guess to its corresponding base value
            self.u0 = initial_guess
            self.u0_base = np.zeros_like(self.u0)
            for ii in range (len(self.u0_base)):
                self.u0_base[ii]= self.u0[ii]/self.ops_max_amp[ii]
                if max(self.u0_base[ii])> 1.0:
                    raise ValueError('Initial guess has strength > max_amp for op %d' % (ii) )
            self.u0_base = np.arcsin(self.u0_base) #because we take the sin of weights later
                
                

            
        else:
            self.u0 =[]
        self.states_concerned_list = states_concerned_list

        self.is_dressed = False
        self.U0_c = U0
        self.initial_unitary = c_to_r_mat(U0) #CtoRMat is converting complex matrices to their equivalent real (double the size) matrices
        if self.expect:
            self.expect_op = c_to_r_mat(self.expect_op)
        
        if draw is not None:
            self.draw_list = draw[0]
            self.draw_names = draw[1]
        else:
            self.draw_list = []
            self.draw_names = []
        
        
        if dressed_info !=None:
            self.v_c = dressed_info['eigenvectors']
            self.dressed_id = dressed_info['dressed_id']
            self.w_c = dressed_info['eigenvalues']
            self.is_dressed = dressed_info['is_dressed']
            self.H0_diag=np.diag(self.w_c)
            
        self.init_system()
        self.init_vectors()
        
        if self.c_ops !=None:
            self.traj = True
            self.state_transfer = True
            if len(U) != len(states_concerned_list):
                full_U = U
                U=[]
                for ii in range(len(states_concerned_list)):
                    U.append(np.dot(full_U,self.initial_vectors_c[ii]))
        
        if self.state_transfer == False:
            self.target_unitary = c_to_r_mat(U)
        else:
            self.target_vectors=[]
            self.target_vectors_c=[]

            for target_vector_c in U:
                self.target_vector = c_to_r_vec(target_vector_c)
                self.target_vectors.append(self.target_vector)
                self.target_vectors_c.append(target_vector_c)
        
        if self.traj:
            self.cdaggerc=[]
            self.c_ops_real=[]
            
                   
            #ceating the effective hamiltonian that describes the evolution of states if no jumps occur
            for ii in range (len(self.c_ops)):
                cdaggerc_c = np.dot(np.transpose(np.conjugate(self.c_ops[ii])),self.c_ops[ii])
                self.c_ops_real.append(c_to_r_mat(self.c_ops[ii]))
                self.cdaggerc.append(c_to_r_mat(cdaggerc_c))
                self.H0_c= self.H0_c + ((0-1j)/2)* ( cdaggerc_c)
                
        self.init_operators()
        self.init_one_minus_gaussian_envelope()
        self.init_guess()

    def approx_expm(self,M,exp_t, scaling_terms): 
        #approximate the exp at the beginning to estimate the number of taylor terms and scaling and squaring needed
        U=np.identity(len(M),dtype=M.dtype)
        Mt=np.identity(len(M),dtype=M.dtype)
        factorial=1.0 #for factorials
        
        for ii in range(1,exp_t):
            factorial*=ii
            Mt=np.dot(Mt,M)
            U+=Mt/((2.**float(ii*scaling_terms))*factorial) #scaling by 2**scaling_terms

        
        for ii in range(scaling_terms):
            U=np.dot(U,U) #squaring scaling times
        
        return U
    
    def approx_exp(self,M,exp_t, scaling_terms): 
        # the scaling and squaring of matrix exponential with taylor expansions
        U=1.0
        Mt=1.0
        factorial=1.0 #for factorials
        
        for ii in range(1,exp_t):
            factorial*=ii
            Mt=M*Mt
            U+=Mt/((2.**float(ii*scaling_terms))*factorial) #scaling by 2**scaling_terms

        
        for ii in range(scaling_terms):
            U=np.dot(U,U) #squaring scaling times
        
        return U
    
    def Choose_exp_terms(self, d): 
        #given our hamiltonians and a number of scaling/squaring, we determine the number of Taylor terms
        

        exp_t = 30 #maximum

        H=self.H0_c
        U_f = self.U0_c
        for ii in range (len(self.ops_c)):
            H = H + self.ops_max_amp[ii]*self.ops_c[ii]
        if d == 0:
            self.scaling = max(int(2*np.log2(np.max(np.abs(-(0+1j) * self.dt*H)))),0) 

        else:
            self.scaling += d

        if self.state_transfer or self.no_scaling:
            self.scaling =0
        while True:

            if len(self.H0_c) < 10:
                for ii in range (self.steps):
                    U_f = np.dot(U_f,self.approx_expm((0-1j)*self.dt*H, exp_t, self.scaling))
                Metric = np.abs(np.trace(np.dot(np.conjugate(np.transpose(U_f)), U_f)))/(self.state_num)
            else:
                max_term = np.max(np.abs(-(0+1j) * self.dt*H))
                
                Metric = 1 + self.steps *np.abs((self.approx_exp(max_term, exp_t, self.scaling) - np.exp(max_term))/np.exp(max_term))

            if exp_t == 3:
                break
            if np.abs(Metric - 1.0) < self.Unitary_error:
                exp_t = exp_t-1
            else:
                break
        
        return exp_t



        
    def init_system(self):
        self.dt = float(self.total_time)/self.steps        
        self.state_num= len(self.H0_c)
        
        
    def init_vectors(self):
        # initialized vectors used for propagation
        self.initial_vectors=[]
        self.initial_vectors_c=[]

        for state in self.states_concerned_list:
            if self.is_dressed:
                self.initial_vector_c= self.v_c[:,get_state_index(state,self.dressed_id)]
            else:
                self.initial_vector_c=np.zeros(self.state_num)
                self.initial_vector_c[state]=1
            
            self.initial_vectors_c.append(self.initial_vector_c)
            self.initial_vector = c_to_r_vec(self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)
        
        

    def init_operators(self):
        # Create operator matrix in numpy array

        self.ops=[]
        for op_c in self.ops_c:
            op = c_to_r_mat(-1j*self.dt*op_c)
            self.ops.append(op)
        
        self.ops_len = len(self.ops)

        self.H0 = c_to_r_mat(-1j*self.dt*self.H0_c)
        self.identity_c = np.identity(self.state_num)
        self.identity = c_to_r_mat(self.identity_c)
        
        if self.Taylor_terms is None:
            self.exps =[]
            self.scalings = []
            if self.state_transfer or self.no_scaling:
                comparisons = 1
            else:
                comparisons = 6
            d = 0
            while comparisons >0:

                self.exp_terms = self.Choose_exp_terms(d)
                self.exps.append(self.exp_terms)
                self.scalings.append(self.scaling)
                comparisons = comparisons -1
                d = d+1
            self.complexities = np.add(self.exps,self.scalings)
            a = np.argmin(self.complexities)

            self.exp_terms = self.exps[a]
            self.scaling = self.scalings[a]
        else:
            self.exp_terms = self.Taylor_terms[0]
            self.scaling = self.Taylor_terms[1]
            
        
        
        print ("Using "+ str(self.exp_terms) + " Taylor terms and "+ str(self.scaling)+" Scaling & Squaring terms")
        
        i_array = np.eye(2*self.state_num)
        op_matrix_I=i_array.tolist()
        
        self.H_ops = []
        for op in self.ops:
            self.H_ops.append(op)
        self.matrix_list = [self.H0]
        for ii in range(self.ops_len):
            self.matrix_list = self.matrix_list + [self.H_ops[ii]]
        self.matrix_list = self.matrix_list + [op_matrix_I]
        
        self.matrix_list = np.array(self.matrix_list)
        
    def init_one_minus_gaussian_envelope(self):
        # Generating the Gaussian envelope that pulses should obey
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        opsnum=self.ops_len
        for ii in range(opsnum):
            constraint_shape = np.ones(self.steps)- self.gaussian(np.linspace(-2,2,self.steps)) - offset
            constraint_shape = constraint_shape * (constraint_shape>0)
            constraint_shape = constraint_shape + overall_offset* np.ones(self.steps)
            one_minus_gauss.append(constraint_shape)


        self.one_minus_gauss = np.array(one_minus_gauss)


    def gaussian(self,x, mu = 0. , sig = 1. ):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def init_guess(self):
        # initail guess for control field
        if self.u0 != []:
            
            self.ops_weight_base = np.reshape(self.u0_base, [self.ops_len,self.steps])
        else:
            initial_mean = 0
            index = 0
            
            initial_stddev = (1./np.sqrt(self.steps))
            self.ops_weight_base = np.random.normal(initial_mean, initial_stddev, [self.ops_len ,self.steps])
        
        self.raw_shape = np.shape(self.ops_weight_base)
        
        





#Defining time scales
total_time = 5
steps = 50
state_transfer = True
RWA = True
RFT = True

#Defining H0





qubit_state_num = 2

fq= 4.6/(2*np.pi)
kappa = 0.05
gamma = 0.001
g = 0.05

mode_state_num = 5
#g = 2.*np.pi*0.1 #GHz
fc = 5.0/(2*np.pi) #GHz
state_num = qubit_state_num * mode_state_num
if RFT:
    fq = fq-fc
    fc = 0
    
wc = 2*np.pi*fc
wa = 2*np.pi*fq


alpha = 0.224574
ens = np.array([ 2*np.pi*ii*(fq - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
H0q = np.kron(np.identity(mode_state_num),np.diag(ens))

a   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),1),np.identity(qubit_state_num))
adag   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),-1),np.identity(qubit_state_num))
sm = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),1))
smdag = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1))

if RWA:
     H0 = wc * np.dot(adag,a) + H0q + g * (np.dot(adag,sm) + np.dot(a,smdag))
else:
     H0 = wc * np.dot(adag,a) + H0q +  g * np.dot((adag + a),(sm + smdag))
#Defining Forbidden sates


#Defining Concerned states (starting states)
psi0=[0,1]

#Defining states to include in the drawing of occupation
states_draw_list = [0,1,2]
states_draw_names = ['g0','e0','g1']



U =[]
U1 = np.zeros(state_num,dtype=complex)
U1[1]=1
U1[0]=0
U.append(U1)
U2 = np.zeros(state_num,dtype=complex)
U2[0]=1
U.append(U2)
    

#Defining U0 (Initial)
q_identity = np.identity(qubit_state_num)
U0= q_identity

#Defining control Hs
IX = a + adag
IY = (0+1j)* (a-adag)
Hops = [IX]
ops_max_amp = [0.05]
Hnames =['HI']

#Defining convergence parameters
max_iterations = 5000
decay = max_iterations/2
convergence = {'rate':0.005, 'update_step':10, 'max_iterations':max_iterations,\
               'conv_target':1e-6,'learning_rate_decay':decay}
reg_coeffs = {'envelope' : 0,  'dwdt':0,'d2wdt2':0}


c_ops=[]
c_ops.append(np.sqrt(gamma)*sm)
c_ops.append(np.sqrt(kappa)*a)

u0 = None

print ("Parameters Defined")



      
draw = [states_draw_list,states_draw_names]
                    
show_plots = False
initial_guess = u0
use_gpu = False,
unitary_error = 1e-4
maxA=ops_max_amp
method ='Adam'
expect_op = IX
file_name='JC'
trajectories = 12000
do_all_traj = False,
data_path = None
save = False
use_inter_vecs=True

def get_inner_product(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for single vector
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num])
    psi_1_imag = (psi1[state_num:2*state_num])
    psi_2_real = (psi2[0:state_num])
    psi_2_imag = (psi2[state_num:2*state_num])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.multiply(psi_1_real,psi_2_real)
        bd = tf.multiply(psi_1_imag,psi_2_imag)
        bc = tf.multiply(psi_1_imag,psi_2_real)
        ad = tf.multiply(psi_1_real,psi_2_imag)
        reals = tf.square(tf.add(tf.reduce_sum(ac),tf.reduce_sum(bd)))
        imags = tf.square(tf.subtract(tf.reduce_sum(bc),tf.reduce_sum(ad)))
        norm = tf.add(reals,imags)
    return norm

def get_loss_list(sys_para,psi1,psi2):
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude

    ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
    bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
    bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
    ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
    ac_bd = tf.square(tf.add(ac,bd))
    bc_ad = tf.square(tf.subtract(bc,ad))

    loss_list = tf.add(ac_bd,bc_ad)
    return loss_list

def get_inner_product_2D(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors
    # psi1 and psi2 are shaped as (2*state_num, number of vectors)
    state_num=sys_para.state_num


    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)

        ac_bd = tf.square(tf.add(ac,bd))
        bc_ad = tf.square(tf.subtract(bc,ad))
        reals = tf.reduce_sum(ac_bd) # first trace inner product of all vectors, then squared
        imags = tf.reduce_sum(bc_ad)
        norm = (tf.add(reals,imags))/(tf.cast(num_vecs,tf.float32))
    return norm

def get_inner_product_3D(sys_para,psi1,psi2, num_vecs):
    #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors and timesteps
    # psi1 and psi2 are shaped as (2*state_num, time_steps, number of vectors)
    state_num=sys_para.state_num

    psi_1_real = (psi1[0:state_num,:])
    psi_1_imag = (psi1[state_num:2*state_num,:])
    psi_2_real = (psi2[0:state_num,:])
    psi_2_imag = (psi2[state_num:2*state_num,:])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)
        reals = tf.reduce_sum(tf.square(tf.reduce_sum(tf.add(ac,bd),1)))
        # first trace inner product of all vectors, then squared, then sum contribution of all time steps
        imags = tf.reduce_sum(tf.square(tf.reduce_sum(tf.subtract(bc,ad),1)))
        norm = (tf.add(reals,imags))/(len(sys_para.states_concerned_list)**2)
    return norm

def get_avgd_inner_product ( sys_para, psi1, psi2, start, end):
    state_num=sys_para.state_num


    psi_1_real = (psi1[0:state_num,start:end])
    psi_1_imag = (psi1[state_num:2*state_num,start:end])
    psi_2_real = (psi2[0:state_num,start:end])
    psi_2_imag = (psi2[state_num:2*state_num,start:end])
    # psi1 has a+ib, psi2 has c+id, we wanna get Sum ((ac+bd) + i (bc-ad)) magnitude
    with tf.name_scope('inner_product'):
        ac = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_real),0)
        bd = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_imag),0)
        bc = tf.reduce_sum(tf.multiply(psi_1_imag,psi_2_real),0)
        ad = tf.reduce_sum(tf.multiply(psi_1_real,psi_2_imag),0)

        ac_bd = tf.add(ac,bd)
        bc_ad = tf.subtract(bc,ad)
        reals = tf.reduce_sum(ac_bd)/tf.cast((end-start), tf.float32) # first trace inner product of all vectors, then squared
        imags = tf.reduce_sum(bc_ad)/tf.cast((end-start), tf.float32)

    return reals, imags
def my_print(text):
    sys.stdout.write(str(text)+'\n')
    sys.stdout.flush()
def expect (sys_para, num_trajs,  op, psis):
    result = []
    psis2 = tf.matmul(tf.cast(op,tf.float32),psis)
    if num_trajs[0] !=0:

        expect1 = get_avgd_inner_product (sys_para, psis, psis2, 0, num_trajs[0])
        if not sys_para.do_all:
            result.append(expect1)
    else:
        expect1 = 0
    if num_trajs[1] !=0:
        expect2 = get_avgd_inner_product (sys_para, psis, psis2, num_trajs[0], num_trajs[0] + num_trajs[1])
        if not sys_para.do_all:
            result.append(expect2)
    else:
        expect2 = 0
    if sys_para.do_all:
        return expect1, expect2
    else:
        return tf.stack(result)

def normalize(sys_para, psi, num_vecs):
    state_num=sys_para.state_num
    new_norms = tf.reshape(get_norms(sys_para, psi, num_vecs),[num_vecs])
    weights = 1/tf.sqrt(new_norms)
    x = []
    for ii in range (2*state_num):
        x.append(weights)
    return tf.multiply(psi,tf.stack(x))




def get_norms(sys_para, psi, num_vecs):
    state_num=sys_para.state_num
    psi1 = tf.reshape(psi,[2*state_num,num_vecs])
    return tf.reduce_sum(tf.square(psi1),0)

def get_norm(sys_para, psi):
    state_num=sys_para.state_num
    psi1 = tf.reshape(psi,[2*state_num,1])
    return tf.reduce_sum(tf.square(psi1),0)
def get_one_random(sys_para, num_trajs, start,end,index):
    vec_type = tf.constant(0)
    sums = []
    s = 0
    for jj in range (len(sys_para.initial_vectors)):
        #create a list of their summed probabilities
        s=s+num_trajs[jj]
        sums=tf.concat([sums,tf.reshape(s,[1])],0)

    r2 = tf.cast(index,tf.int32)
    rvector=r2 * tf.ones_like(sums)
    cond2= tf.greater_equal(sums,rvector)
    b=tf.where(cond2)
    final =tf.reshape(b[0,:],[])
    return tf.random_uniform([1],tf.gather(start,final),tf.gather(end,final))



def get_random(sys_para,num_trajs,  start,end,length=1):

    #Returns a random number between 0 & 1 to tell jumps when to occur
    ii =0
    rand = []
    for initial_vector in sys_para.initial_vectors:
        new = tf.random_uniform([num_trajs[ii]],start[ii],end[ii])
        if rand == []:
            rand = new
        else:
            rand = tf.concat([rand,new],0)
        ii = ii+1

    #rand=tf.random_uniform([length],start,end)
    return rand

def divide(needed,max_traj):
    returned = []

    end = False

    while not end:
        summation = 0
        trial = np.zeros_like(needed)
        flag = True

        for ii in range (len(needed)):
            if flag and ((summation + needed[ii]) <= max_traj):
                summation = summation + needed[ii]
                trial[ii] = needed[ii]
                if ii == len(needed)-1:
                    end = True
            else:

                trial[ii] = max_traj - summation
                summation = max_traj
                flag = False
        returned.append(trial)
        needed = needed-trial
    return returned
    


# start time
grape_start_time = time.time()
freq_unit = "GHz"
# set timing unit used for plotting
freq_time_unit_dict = {"GHz": "ns", "MHz": "us","KHz":"ms","Hz":"s"}
time_unit = freq_time_unit_dict[freq_unit]

# make sparse_{H,U,K} False if use_gpu is True, as GPU Sparse Matmul is not supported yet.
if use_gpu:
    sparse_H = False
    sparse_U = False
    sparse_K = False

file_path = None



if U0 is None:
    U0 = np.identity(len(H0))
if convergence is None:
    convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,'conv_target':1e-8,'learning_rate_decay':2500}


if maxA is None:
    if initial_guess is None:
        maxAmp = 4*np.ones(len(Hops))
    else:
        maxAmp = 1.5*np.max(np.abs(initial_guess))*np.ones(len(Hops))
else:
    maxAmp = maxA

Taylor_terms = None
if state_transfer:
    no_scaling = True
else:
    no_scaling = False
dressed_info = None
# pass in system parameters
sys_para = SystemParameters(H0,Hops,Hnames,U,U0,total_time,steps,psi0,dressed_info,maxAmp, draw,initial_guess,  show_plots,unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms, use_gpu, use_inter_vecs,sparse_H,sparse_U,sparse_K, c_ops, trajectories, do_all_traj, expect_op)

#run_python = Python_evolve(sys_para)


tf_hostlist = str(os.environ['SLURM_NODELIST'])
node_name = str(os.environ["SLURMD_NODENAME"])
foo = ( [pos for pos, char in enumerate(tf_hostlist) if char == ','])
clusters = tf_hostlist.count(',')
start_i = 9
hosts =[]
server = tf_hostlist[0:8]
for ii in range (clusters+1):
    if ii>0:
        start_i = foo[ii-1]+1
    if clusters==0:
        end_i = len(tf_hostlist)-1
    elif ii != clusters:
        end_i = foo[ii]
    else:
        end_i = len(tf_hostlist)-1
    string = tf_hostlist[start_i:end_i]
    if (len(string) <5):
        hosts.append(server+string+":2222")
    else:
        s = int(string[0:4])
        e = int(string[6:10])
        for jj in range (e-s+1):
            hosts.append(server+str(s+jj).zfill(4)+":2222")

idx = hosts.index(node_name+":2222") 
if (idx==0):
    job_name = "ps"
    task_index = 0
else:
    job_name = "worker"
    task_index = idx -1
cluster = tf.train.ClusterSpec( {"ps" : [hosts[0]], "worker": hosts[1:] } )
server = tf.train.Server(server_or_cluster_def=cluster,
                         job_name=job_name, task_index=task_index)

print(hosts)
print(node_name,job_name,task_index)
sys.stdout.flush()
norms=[]
jumps=[]

input_num = len(sys_para.Hnames) +1
taylor_terms = sys_para.exp_terms 
scaling = sys_para.scaling
num_vecs = len(sys_para.initial_vectors)
print ("Building graph:")
sys.stdout.flush()
if job_name == "ps":
    print ("PS Joined")
    sys.stdout.flush()
    server.join()
else:
    print ("Worker running")
    sys.stdout.flush()
    is_chief = (task_index == 0 and job_name == "worker")
    with tf.Graph().as_default():  
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster = cluster)) :            
                    


            if sys_para.traj:
                tf_c_ops = tf.constant(np.reshape(sys_para.c_ops_real,[len(sys_para.c_ops),2*sys_para.state_num,2*sys_para.state_num]),dtype=tf.float32)
                tf_cdagger_c = tf.constant(np.reshape(sys_para.cdaggerc,[len(sys_para.c_ops),2*sys_para.state_num,2*sys_para.state_num]),dtype=tf.float32)
                
                if sys_para.expect:
                    expect_op = tf.constant(sys_para.expect_op)


            def get_matexp(uks,H_all):
                # matrix exponential
                I = H_all[input_num]
                matexp = I
                uks_Hk_list = []
                for ii in range(input_num):
                    uks_Hk_list.append((uks[ii]/(2.**scaling))*H_all[ii])

                H = tf.add_n(uks_Hk_list)
                H_n = H
                factorial = 1.

                for ii in range(1,taylor_terms+1):      
                    factorial = factorial * ii
                    matexp = matexp + H_n/factorial
                    if not ii == (taylor_terms):
                        H_n = tf.matmul(H,H_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_U)

                for ii in range(scaling):
                    matexp = tf.matmul(matexp,matexp,a_is_sparse=sys_para.sparse_U,b_is_sparse=sys_para.sparse_U)

                return matexp


            @function.Defun(tf.float32,tf.float32,tf.float32)
            def matexp_op_grad(uks,H_all, grad):  
                # gradient of matrix exponential
                coeff_grad = []

                coeff_grad.append(tf.constant(0,dtype=tf.float32))


                ### get output of the function
                matexp = get_matexp(uks,H_all)          
                ###

                for ii in range(1,input_num):
                    coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                           tf.matmul(H_all[ii],matexp,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_U))))

                return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32)]                                         

            global matexp_op


            @function.Defun(tf.float32,tf.float32, grad_func=matexp_op_grad)                       
            def matexp_op(uks,H_all):
                # matrix exponential defun operator
                matexp = get_matexp(uks,H_all)

                return matexp 

            def get_matvecexp(uks,H_all,psi):
                # matrix vector exponential
                I = H_all[input_num]
                matvecexp = psi

                uks_Hk_list = []

                for ii in range(input_num):
                    uks_Hk_list.append(uks[ii]*H_all[ii])

                H = tf.add_n(uks_Hk_list)    

                psi_n = psi
                factorial = 1.

                for ii in range(1,taylor_terms):      
                    factorial = factorial * ii
                    psi_n = tf.matmul(H,psi_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K)
                    matvecexp = matvecexp + psi_n/factorial

                return matvecexp


            @function.Defun(tf.float32,tf.float32,tf.float32,tf.float32)
            def matvecexp_op_grad(uks,H_all,psi, grad):  
                # graident of matrix vector exponential
                coeff_grad = []

                coeff_grad.append(tf.constant(0,dtype=tf.float32))

                ### get output of the function
                matvecexp = get_matvecexp(uks,H_all,psi)
                #####


                for ii in range(1,input_num):
                    coeff_grad.append(tf.reduce_sum(tf.multiply(grad,
                           tf.matmul(H_all[ii],matvecexp,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K))))



                I = H_all[input_num]
                vec_grad = grad
                uks_Hk_list = []
                for ii in range(input_num):
                    uks_Hk_list.append((-uks[ii])*H_all[ii])

                H = tf.add_n(uks_Hk_list)
                vec_grad_n = grad
                factorial = 1.

                for ii in range(1,taylor_terms):      
                    factorial = factorial * ii
                    vec_grad_n = tf.matmul(H,vec_grad_n,a_is_sparse=sys_para.sparse_H,b_is_sparse=sys_para.sparse_K)
                    vec_grad = vec_grad + vec_grad_n/factorial

                return [tf.stack(coeff_grad), tf.zeros(tf.shape(H_all),dtype=tf.float32),vec_grad]                                         

            global matvecexp_op

            @function.Defun(tf.float32,tf.float32,tf.float32, grad_func=matvecexp_op_grad)                       
            def matvecexp_op(uks,H_all,psi):
                # matrix vector exponential defun operator
                matvecexp = get_matvecexp(uks,H_all,psi)

                return matvecexp





            tf_one_minus_gaussian_envelope = tf.constant(sys_para.one_minus_gauss,dtype=tf.float32, name = 'Gaussian')

            

            if sys_para.traj:
                tf_initial_vectors=[]
                num_trajs = tf.placeholder(tf.int32, shape = [num_vecs])
                vecs = tf.reshape(tf.constant(sys_para.initial_vectors[0],dtype=tf.float32),[1,2*sys_para.state_num])
                targets = tf.reshape(tf.constant(sys_para.target_vectors[0],dtype =tf.float32),[1,2*sys_para.state_num])
                ii = 0
                counter = tf.constant(0)
                for initial_vector in sys_para.initial_vectors:

                    tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
                    target_vector = tf.reshape(tf.constant(sys_para.target_vectors[ii],dtype =tf.float32),[1,2*sys_para.state_num])
                    tf_initial_vectors.append(tf_initial_vector)
                    tf_initial_vector = tf.reshape(tf_initial_vector,[1,2*sys_para.state_num])
                    i = tf.constant(0)

                    c = lambda i,vecs,targets: tf.less(i, num_trajs[ii])

                    def body(i,vecs,targets):

                        def f1(): return tf.concat([vecs,tf_initial_vector],0), tf.concat([targets,target_vector],0)
                        def f2(): return tf_initial_vector, target_vector
                        vecs,targets = tf.cond(tf.logical_and(tf.equal(counter,tf.constant(0)),tf.equal(i,tf.constant(0))), f2, f1)



                        return [tf.add(i,1), vecs,targets]

                    r,vecs,targets = tf.while_loop(c, body, [i,vecs,targets],shape_invariants = [i.get_shape(), tf.TensorShape([None,2*sys_para.state_num]), tf.TensorShape([None,2*sys_para.state_num])])
                    counter = tf.add(counter,r)
                    ii = ii+1
                vecs = tf.transpose(vecs)
                targets = tf.transpose(targets)
                packed_initial_vectors = vecs
                target_vecs = targets
                num_vecs = counter

            else:
                tf_initial_vectors=[]
                for initial_vector in sys_para.initial_vectors:
                    tf_initial_vector = tf.constant(initial_vector,dtype=tf.float32)
                    tf_initial_vectors.append(tf_initial_vector)
                packed_initial_vectors = tf.transpose(tf.stack(tf_initial_vectors))

            H0_weight = tf.Variable(tf.ones([sys_para.steps]), trainable=False) #Just a vector of ones needed for the kernel
            weights_unpacked=[H0_weight] #will collect all weights here
            ops_weight_base = tf.Variable(tf.constant(sys_para.ops_weight_base, dtype = tf.float32), dtype=tf.float32,name ="weights_base")

            ops_weight = tf.sin(ops_weight_base,name="weights")
            for ii in range (sys_para.ops_len):
                weights_unpacked.append(sys_para.ops_max_amp[ii]*ops_weight[ii,:])

            #print len(sys_para.ops_max_amp)
            H_weights = tf.stack(weights_unpacked,name="packed_weights")



            print ("Operators weight initialized.")
            sys.stdout.flush()



            global_step = tf.get_variable('global_step', [], 
                              initializer = tf.constant_initializer(0), 
                              trainable = False,
                              dtype = tf.int32)


            jump_vs = []
            tf_matrix_list = tf.constant(sys_para.matrix_list,dtype=tf.float32)
            # Create a trajectory for each initial state
            Evolution_states=[]
            inter_vecs=[]
            inter_lst = []
            #start = tf.placeholder(tf.float32,shape=[])
            #end = tf.placeholder(tf.float32,shape=[])
            start = tf.placeholder(tf.float32,shape=[len(sys_para.initial_vectors)])
            end = tf.placeholder(tf.float32,shape=[len(sys_para.initial_vectors)])
            psi0 = packed_initial_vectors
            old_psi = psi0
            new_psi = psi0
            norms = tf.ones([num_vecs],dtype = tf.float32)
            r=get_random(sys_para,num_trajs,  start,end,num_vecs)
            operator = tf_c_ops[0] # temporary
            expects = []
            inter_vecs_list=[]
            inter_vecs_list.append(old_psi)
            all_jumps= []
            all_norms = []
            all_norms.append(norms)
            vecs = tf.cast(num_vecs, tf.int64)
            for ii in np.arange(0,sys_para.steps):
                old_psi = new_psi        
                new_psi = matvecexp_op(H_weights[:,ii],tf_matrix_list,old_psi)
                new_norms = tf.reshape(get_norms(sys_para, new_psi, num_vecs),[num_vecs])

                norms = tf.multiply(norms,new_norms)
                all_norms.append(norms)

                cond= tf.less(norms,r)
                a=tf.where(cond)
                state_num=sys_para.state_num
                reshaped_new = tf.reshape(new_psi,[2*state_num*num_vecs])

                c = tf.constant(0)
                def while_condition(c,old,new,norms,randoms):
                    return tf.less(c, tf.size(a))
                def jump_fn(c,old,new,norms,randoms):


                    index = tf.reshape(tf.gather(a,c),[])
                    idx = []

                    for kk in range (2*state_num):
                        idx.append(index + kk*vecs)

                    vector = tf.gather(reshaped_new,idx)
                    #vector = tf.gather(tf.transpose(old),index)


                    #####


                    if len(sys_para.c_ops)>1:
                        weights=[]
                        sums=[]
                        s=0
                        for ii in range (len(sys_para.c_ops)):

                            temp=tf.matmul(tf.transpose(tf.reshape(vector,[2*state_num,1])),tf_cdagger_c[ii,:,:])
                            temp2=tf.matmul(temp,tf.reshape(vector,[2*state_num,1])) #get the jump expectation value
                            weights=tf.concat([weights,tf.reshape(temp2,[1])],0)
                        weights=tf.abs(weights/tf.reduce_sum(tf.abs(weights))) #convert them to probabilities

                        for jj in range (len(sys_para.c_ops)):
                            #create a list of their summed probabilities
                            s=s+weights[jj]
                            sums=tf.concat([sums,tf.reshape(s,[1])],0)

                        r2 = tf.random_uniform([1],0,1)
                        #tensorflow conditional graphing, checks for the first time a summed probability exceeds the random number
                        rvector=r2 * tf.ones_like(sums)
                        cond2= tf.greater_equal(sums,rvector)
                        b=tf.where(cond2)
                        final =tf.reshape(b[0,:],[])
                        #final = tf.gather(b,0)

                        #apply the chosen jump operator
                        propagator2 = tf.reshape(tf.gather(tf_c_ops,final),[2*sys_para.state_num,2*sys_para.state_num])
                    else:
                        propagator2 = tf.reshape(tf_c_ops,[2*sys_para.state_num,2*sys_para.state_num])
                    inter_vec_temp2 = tf.matmul(propagator2,tf.reshape(vector,[2*sys_para.state_num,1]))
                    norm2 = get_norm(sys_para, inter_vec_temp2)
                    inter_vec_temp2 = inter_vec_temp2 / tf.sqrt(norm2)

                    #delta = tf.reshape(inter_vec_temp2 - tf.gather(tf.transpose(new),index),[2*sys_para.state_num])

                    new_vector = tf.reshape(tf.gather(tf.reshape(new,[2*state_num*num_vecs]),idx),[2*sys_para.state_num])
                    inter_vec_temp2 = tf.reshape(inter_vec_temp2,[2*sys_para.state_num])
                    #delta = inter_vec_temp2 
                    delta = inter_vec_temp2-new_vector
                    indices=[]
                    for jj in range (2*sys_para.state_num):
                        indices.append([jj,index])

                    values = delta
                    shape = tf.cast(tf.stack([2*sys_para.state_num,num_vecs]),tf.int64)
                    Delta = tf.SparseTensor(indices, values, shape)
                    new = new + tf.sparse_tensor_to_dense(Delta)


                    values = tf.reshape(1 - tf.gather(norms,index),[1])
                    shape = tf.cast(tf.stack([num_vecs]),tf.int64)
                    Delta_norm = tf.SparseTensor(tf.reshape(index,[1,1]), values, shape)
                    norms = norms + tf.sparse_tensor_to_dense(Delta_norm)

                    #new_random = get_one_random(start, end,index)
                    new_random =tf.random_uniform([1],0,1)
                    values = tf.reshape(new_random - tf.gather(randoms,index),[1])
                    #shape = tf.stack([num_vecs])
                    Delta_norm = tf.SparseTensor(tf.reshape(index,[1,1]), values, shape)
                    randoms = randoms + tf.sparse_tensor_to_dense(Delta_norm)

                    #####

                    return [tf.add(c, 1),old,new,norms,randoms]

                wh,old_psi,new_psi,norms,r = tf.while_loop(while_condition, jump_fn, [c,old_psi,new_psi,norms,r])
                all_jumps.append(wh)


                new_psi = normalize(sys_para, new_psi, num_vecs)

                inter_vecs_list.append(new_psi)
                if sys_para.expect:

                    expects.append(expect(sys_para, num_trajs, expect_op, new_psi))

            inter_vecs_packed = tf.stack(inter_vecs_list, axis=1)
            inter_vecs = inter_vecs_packed
            all_norms = tf.stack(all_norms)
            if sys_para.expect:
                if sys_para.do_all:
                    expectations = tf.stack(expects, axis=1)
                else:
                    expectations = tf.stack(expects)
            else:
                expectations = 0

            #####

            #inter_vecs_packed.set_shape([2*sys_para.state_num,sys_para.steps,num_vecs] )
            #inter_vecs2 = tf.unstack(inter_vecs_packed, axis = 2)
            #indices = tf.stack(indices)






            #inter_vec = tf.reshape(psi0,[2*sys_para.state_num,1],name="initial_vector")
            #psi0 = inter_vec


            all_jumps = tf.stack(all_jumps)
            jumps.append(jumps)
            #jumps = tf.stack(jumps)
            #for tf_initial_vector in tf_initial_vectors:
                #Evolution_states.append(One_Trajectory(tf_initial_vector)) #returns the final state of the trajectory
            packed = inter_vecs_packed
            print ("Trajectories Initialized")
            sys.stdout.flush()


            if sys_para.state_transfer == False:

                final_vecs = tf.matmul(final_state, packed_initial_vectors)

                loss = 1-get_inner_product_2D(sys_para, final_vecs,target_vecs, num_vecs)

            else:
                #loss = tf.constant(0.0, dtype = tf.float32)
                final_state = inter_vecs_packed[:,sys_para.steps,:]
                a = []
                for ii in range (sys_para.steps):
                    a.append(tf.constant((sys_para.steps-ii), dtype = tf.float32))
                accelerate = tf.stack(a)
                accelerate = tf.ones([sys_para.steps])
                #
                if sys_para.expect:

                    Il1 = tf.reduce_sum(expectations[:,0,0])  
                    Il2 = -tf.reduce_sum(expectations[:,1,0])
                    Il = Il1 + Il2
                    Il1d = tf.gradients(Il1, [ops_weight_base])[0]
                    Il2d = tf.gradients(Il2, [ops_weight_base])[0]
                    loss = - tf.square(Il)
                    quad = tf.gradients(loss, [ops_weight_base])[0]




                unitary_scale = get_inner_product_2D(sys_para, final_state,final_state, num_vecs)


            reg_loss = loss

            print ("Training loss initialized.")
            sys.stdout.flush()
            #learning_rate = tf.placeholder(tf.float32,shape=[])
            opt = tf.train.GradientDescentOptimizer(0.05)
            #opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(hosts)-1,total_num_replicas=len(hosts)-1)
            opt = sync(opt,replicas_to_aggregate=len(hosts)-1,total_num_replicas=len(hosts)-1,y1 = -2*Il1, y2 = -2*Il2,g1 = IL1d, g2 = IL2d  )
            sync_replicas_hook = opt.make_session_run_hook(is_chief)

            #Here we extract the gradients of the pulses
            

            
            optimizer = opt.minimize(reg_loss, global_step = global_step)


            print ("Optimizer initialized.")
            sys.stdout.flush()





            print ("Graph " +str(task_index) + " built!")
            sys.stdout.flush()
            
            local_init_op = opt.local_step_init_op
            if is_chief:
                  local_init_op = opt.chief_init_op
            ready_for_local_init_op = opt.ready_for_local_init_op
            # Initial token and chief queue runners required by the sync_replicas mode
            
            init_op = tf.global_variables_initializer()
            init_token_op = opt.get_init_tokens_op()
            chief_queue_runner = opt.get_chief_queue_runner()


            sv = tf.train.Supervisor(is_chief=is_chief, local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                                 init_op=init_op,
                                 recovery_wait_secs=0.001,
                                 global_step=global_step)

            config = tf.ConfigProto(allow_soft_placement = True, device_filters=["/job:ps", "/job:worker/task:0", "/job:worker/task:%d" % task_index])

            sess = sv.prepare_or_wait_for_session(server.target, config = config)



            itera = 0
            if is_chief:
                sess.run(init_token_op)
                sv.start_queue_runners(sess, [chief_queue_runner])
                


            traj_num = sys_para.trajectories
            max_traj = 1000
            num_psi0 = len(sys_para.initial_vectors)
            needed_traj = []
            for kk in range (num_psi0):
                needed_traj.append(traj_num)
            jump_traj = np.sum(needed_traj)
            num_batches = len(hosts)-1
            num_traj_batch = int(traj_num/num_batches)
            lrate = 0.005
            fd_dict = {start: np.zeros([num_psi0]), end: np.ones([num_psi0]), num_trajs:num_traj_batch*np.ones([num_psi0])}
            print ("Entering iterations_"+str(task_index))
            sys.stdout.flush()
            if is_chief:
                sleep(0.01)
            for ii in range(100):
                
                sleep(random_sample())


                my_print('\r'+' Iteration: ' +str(ii) + ": Running batch #" +str(task_index+1)+" out of "+str(num_batches)+ " with "+str(num_traj_batch)+" jump trajectories")
                #sys.stdout.flush()

                #nos, exs, l1d,l2d,  q, l1, l2, int_vecs,step = sess.run([norms, expectations, Il1d, Il2d,quad, Il1, Il2, inter_vecs, global_step], feed_dict=fd_dict)
                _, step, rl = sess.run([optimizer, global_step, reg_loss], feed_dict=fd_dict)
                #print (np.square(l1 + l2))
                #sys.stdout.flush()
                my_print (ii)
                my_print(task_index)
                my_print(rl)
                #time.sleep( np.random.random_sample())
                with open('out.txt', 'a') as the_file:
                    the_file.write('\r'+' Iteration: ' +str(ii) + ": Running batch #" +str(task_index+1)+" out of "+str(num_batches)+ " with "+str(num_traj_batch)+" jump trajectories " + "step:" + str(step) +  " " + str(os.environ["SLURMD_NODENAME"]) + "Area: " + str(-rl) + "\n")

                #sys.stdout.flush()


    #conv = Convergence(sys_para,time_unit,convergence)

    # run the optimization
    #SS = run_session(tfs,graph,conv,sys_para,method, show_plots = sys_para.show_plots, use_gpu = use_gpu)
    #return SS.uks,SS.Uf

######################################################################################################################
