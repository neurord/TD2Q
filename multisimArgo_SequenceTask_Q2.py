# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:22:45 2020

@author: kblackw1
"""

from __future__ import print_function, division

def run_sim(params,numevents,noise,results,sa_combo,trial_subset,printR=False):
    import sequence_env as env
    import agent_twoQtwoSsplit as QL
    from SequenceTask import RL
    from SequenceTaskParam import env_params, states,act,Tloc,R

    acq = RL(env.separable_T, QL.QL, states,act,R,Tloc,params,env_params,printR=printR)
    acq.episode(numevents,noise=noise,info=False)
    results=acq.count_actions(results,sa_combo,trial_subset,accum_type='count')
    del acq
    return results

def run_one_set(p):
    st1,st2,q2o=p
    import numpy as np
    from SequenceTask import save_results,construct_key

    numtrials=600 #allow agent to perform this many actions/events
    runs=10
    noise=0.01 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    printR=False #print environment Reward matrix

    from SequenceTaskParam import Hx_len,params
    if Hx_len==3:
        #MINIMUM actions for reward = 6, so maximum rewards = 1 per 6 actions/events
        state_action_combos=[(('*','*LL'), 'goR'),(('Rlever','*LL'),'press'),(('Rlever','LLR'),'press')]
        events_per_trial=6
    elif Hx_len==4:
        #MINIMUM actions for reward = 7, so maximum rewards = 1 per 7 actions/events
        state_action_combos=[(('*','**LL'), 'goR'),(('Rlever','**LL'),'press'),(('Rlever','*LLR'),'press'),(('Rlever','LLRR'),'goMag')]
        events_per_trial=7
    else:
        print('unrecognized press history length')
        
    params['events_per_trial']=events_per_trial
    alpha1=[0.2,0.3,0.4,0.5]
    alpha2=[0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    
    numevents=numtrials*events_per_trial #number of events/actions allowed for agent per run/trial
    trial_subset=int(0.05*numevents)# display mean reward and count actions over 1st and last of these number of trials 
    max_correct=trial_subset/events_per_trial/100

    epochs=['Beg','End']
    
    keys=construct_key(state_action_combos +['rwd'],epochs)
    ### allresults - store mean performance vs parameter at begining and end of trials
    allresults={k+'_'+ep:[] for k in keys.values() for ep in epochs}
    allresults['params']={p:[] for p in params} #to store list of parameters
    resultslist={k+'_'+ep:[] for k in keys.values() for ep in epochs}
    resultslist['params']={p:[] for p in params}
   
    for a1 in alpha1: #at least 2*a2, double increment
        for a2 in alpha2:
            print('************ NEW SIM *********',np.round(st1,3),np.round(st2,3),np.round(a1,3),np.round(a2,3))
            #update some parameters
            params['numQ']=2 
            params['state_thresh']=[np.round(st1,3),np.round(st2,3)] #threshold on prob for creating new state 
            # higher means more states. 
            params['alpha']=[np.round(a1,3),np.round(a2,3)]
            params['Hx_len']=Hx_len
            params['Q2other']=q2o 
            #results: initialize for each set of parameters
            for p in allresults['params'].keys():
                allresults['params'][p].append(params[p])                #
                resultslist['params'][p].append(params[p])                #
            results={sa:{'Beg':[],'End':[]} for sa in state_action_combos+['rwd']}
            for r in range(runs):
                results=run_sim(params,numevents,noise,results,state_action_combos,trial_subset,printR)
            allresults,resultslist=save_results(results,epochs,allresults,keys,resultslist)
            resultslist['params']['max_correct']=max_correct
    fname='Sequence_HxLen'+str(params['Hx_len'])+'_Q'+str(params['numQ'])+'_q2o'+'_'.join([str(params['Q2other']),str(round(st1,3)),str(round(st2,3))])+'_all'
    print('**************** End of a1,a2 loop ************', fname)
    np.savez(fname,allresults=allresults,params=params,reslist=resultslist)
    return 
    
if __name__ == "__main__":
    from multiprocessing.pool import Pool
    import os
    #loop over values for state_Thresh, alpha1,alpha2 here
    state_thresh=[0.5,0.625,0.75,0.875,1.0]
    Q2_other=[0.05,0.1,0.2]
    ############### This is not quite working - simulations randomly stop for some param combos with no error message ############
    ########## Possibly running out of memory? ###################
    params=[(round(st1,3),round(st2,3),round(q2o,2)) for st1 in state_thresh for st2 in state_thresh for q2o in Q2_other]
    max_pools=os.cpu_count()
    #num_pools=min(len(params),max_pools) #needed on single workstation
    num_pools=len(params)
    print('************* number of processors',max_pools,' params',len(params),params)
    p = Pool(num_pools,maxtasksperchild=1)
    p.map(run_one_set,params)
    print('#################### Returned from p.map ##################')
    
