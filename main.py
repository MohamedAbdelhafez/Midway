#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--nodes", help="assign nodes id for this run",default='0')
#args = parser.parse_args()
#nodes = str(args.nodes)
#start_node = int(nodes[7:10])
#end_node = int(nodes[11:14])
#f = open('out.txt','w')
#for ii in range (end_node - start_node):
    
    #f.write(str(start_node+ii) + " ")
    
#f.close()
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir+"/Packages") 
from main_grape.Grape_original import Grape

#Defining time scales
total_time = 20
steps = 100
state_transfer = True
RWA = True
RFT = True

#Defining H0
qubit_state_num = 2


kappa = 1.0/160
gamma = 0.001
mode_freq1 = 10
fq = 9
g = 0.1*np.pi


mode_state_num = 5

state_num = qubit_state_num * mode_state_num

wc = 2*np.pi*mode_freq1
wa = 2*np.pi*fq
chi = g*g/(wc - wa)
wd = wc -chi
a   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),1),np.identity(qubit_state_num))
adag   = np.kron(np.diag(np.sqrt(np.arange(1,mode_state_num)),-1),np.identity(qubit_state_num))
sm = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),1))
smdag = np.kron(np.identity(mode_state_num),np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1))
if RFT:
    if RWA: 
        H0 = (wc-wd) * np.dot(adag,a) +(wa -wd) * np.dot(smdag,sm) + g * (np.dot(adag,sm) + np.dot(a,smdag))
    else:
        H0 =  (wc-wd) * np.dot(adag,a) +(wa-wd) * np.dot(smdag,sm) +  g * np.dot((adag + a),(sm + smdag))
    
else:
    if RWA:
         H0 = wc * np.dot(adag,a) + wa * np.dot(smdag,sm) + g * (np.dot(adag,sm) + np.dot(a,smdag))
    else:
         H0 = wc * np.dot(adag,a) + wa * np.dot(smdag,sm) +  g * np.dot((adag + a),(sm + smdag))
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



      
uks,U_final = Grape(H0,Hops,Hnames,U,total_time,steps,psi0,convergence=convergence, draw = [states_draw_list,states_draw_names],  
                    
                    show_plots = False, c_ops = c_ops, initial_guess = u0, use_gpu = True,
       unitary_error = 1e-4,  maxA=ops_max_amp, state_transfer = state_transfer, method ='Adam', expect_op = IY,
                    reg_coeffs=reg_coeffs, file_name='JC', trajectories = 2000, do_all_traj = False,
                    data_path = str(currentdir)+'/Data')

######################################################################################################################
