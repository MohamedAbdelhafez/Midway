import numpy as np
import h5py
import matplotlib.pyplot as plt
import qutip as qt

def open_qutip_verification(datafile):
    
    
    # load data from file
    with h5py.File(datafile,'r') as hf:
    
        gate_time = np.array(hf.get('total_time'))
        gate_steps = np.array(hf.get('steps'))
        H0 = np.array(hf.get('H0'))
        error = np.array(hf.get('error'))
        Hops = np.array(hf.get('Hops'))
        initial_vectors_c = np.array(hf.get('initial_vectors_c'))
        target_vectors_c = np.array(hf.get('target_vectors_c'))
        c_ops = np.array(hf.get('c_ops'))
        uks = np.array(hf.get('uks'))[-1]

        
    
    max_abs_diff_list = []
    all_close_list = []
    
    # H0 and Hops
    H0_qobj = qt.Qobj(H0)
    Hops_qobj = []

    for Hop in Hops:
        Hops_qobj.append(qt.Qobj(Hop))
            
            
    cops_qobj = []

    for cop in c_ops:
        cops_qobj.append(qt.Qobj(cop))
    # define time    
    tlist = np.linspace(0,gate_time,gate_steps+1)
    dt = gate_time/gate_steps
    
        
    # append zero control pulse at the end of uks (final timestep)
    uks_t0 = np.zeros((uks.shape[0],1))
    uks = np.hstack([uks,uks_t0])
    
    
    # looping over each initial vector
    for init_vector_id in range(len(initial_vectors_c)):
        
        print "Verifying init vector id: %d" %(init_vector_id)
        
        # initial vector
        psi0 = qt.Qobj(initial_vectors_c[init_vector_id])
        target = qt.Qobj(target_vectors_c[init_vector_id])
        # make functions to return uks field
        def make_get_uks_func(id):
            def _function(t,args=None):
                time_id = int(t/dt)
                return uks[id][time_id]
            return _function
        
        # create the time-dependent Hamiltonian list
        Ht_list = []
        Ht_list.append(H0_qobj)
        for ii in range(len(Hops)):
            Ht_list.append([Hops_qobj[ii],make_get_uks_func(ii)])
            #us = np.asarray(np.reshape(uks[ii],[len(uks[ii])]))
            
            #Ht_list.append([Hops_qobj[ii],us])
        
        opts = qt.Odeoptions(method='adams', nsteps=100000, atol=1e-10, rtol=1e-10)
        
        #args = {}
#output = mesolve(hamiltonian_JC, psi0, time_list, jump_op_list, [sm.dag()*sm, a.dag()*a, a, sm], args=args, options=me_options)
        
        # solving the Schrodinger evolution in QuTiP's sesolve
        #output = qt.sesolve(Ht_list, psi0, tlist, [])
        
        output = qt.mesolve(Ht_list, psi0, tlist, cops_qobj,[psi0*psi0.dag(), target*target.dag()])
        
        print output.expect[1][-1],1-error[-1]
        print np.abs(output.expect[1][-1]-1+error[-1])/output.expect[1][-1]
        
        fig, ax = plt.subplots(figsize=(9,6))
        ax.plot(tlist, output.expect[0], label='initial')
        ax.plot(tlist, output.expect[1], label='target')
        ax.legend()
        #ax.set_ylim(-0.01,1.1)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Occupation probability')
        
        '''
        # obtaining the simulation result
        state_tlist = []
        for state in output.states:
            state_tlist.append(state.full())
        state_tlist = np.array(state_tlist)[:,:,0]
        state_tlist = np.transpose(state_tlist)
        
        
        # absolute difference of simulation result from Tensorflow and QuTiP
        abs_diff = np.abs(state_tlist) - np.abs(inter_vecs_raw[init_vector_id])        
        max_abs_diff_list.append(np.max(abs_diff))
        
        # if all close between simulation result from Tensorflow and QuTiP
        all_close = np.allclose(state_tlist,inter_vecs_raw[init_vector_id])        
        all_close_list.append(all_close)
    
    print "QuTiP simulation verification result for each initial state"
    print "================================================"
    print "max abs diff: " + str(max_abs_diff_list)
    print "all close: " + str(all_close_list)
    print "================================================"
    '''