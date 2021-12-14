# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:22:45 2020

@author: kblackw1
"""

import numpy as np

def run_sim(params,numevents,noise,results,sa_combo,trial_subset,printR=False):
    import sequence_env as env
    import agent_twoQtwoSsplit as QL
    from SequenceTask import RL
    from SequenceTaskParam import env_params, states,act,Tloc,R

    acq = RL(env.separable_T, QL.QL, states,act,R,Tloc,params,env_params,printR=printR)
    acq.episode(numevents,noise=noise,info=False)
    results=acq.count_actions(results,sa_combo,trial_subset,accum_type='count')
    #count number of times maximal reward obtained during trial_subset
    del acq
    return results

numtrials=600 #sim time for 600 trials, Hx_len=4, 10 runs is < 10 min on laptop.  25 sims < 250 min = 4 hours - HiPri
runs=10 #use 2 to test, 10 to simulate
noise=0.01 #make noise small enough or state_thresh small enough to minimize new states in acquisition
printR=False #print environment Reward matrix

from SequenceTask import save_results,construct_key

from SequenceTaskParam import Hx_len,params
if Hx_len==3:
    #MINIMUM actions for reward = 6, so maximum rewards = 1 per 6 "trials"
    state_action_combos=[(('*','*LL'), 'goR'),(('Rlever','*LL'),'press'),(('Rlever','LLR'),'press')]
    events_per_trial=6
elif Hx_len==4:
    #MINIMUM actions for reward = 7, so maximum rewards = 1 per 7 "trials"
    state_action_combos=[(('*','**LL'), 'goR'),(('Rlever','**LL'),'press'),(('Rlever','*LLR'),'press'),(('Rlever','LLRR'),'goMag')]
    events_per_trial=7
else:
    print('unrecognized press history length')
    
params['events_per_trial']=events_per_trial
epochs=['Beg','End']
#loop over values for state_Thresh, alpha1,alpha2 here
state_thresh=[0.5,0.625,0.75,0.875,1.0] # 5 values
alpha1=[0.1,0.2,0.3,0.4,0.5]
   
numevents=numtrials*events_per_trial #number of events/actions allowed for agent per run/trial
trial_subset=int(0.05*numevents)# display mean reward and count actions over 1st and last of these number of trials 
max_correct=trial_subset/events_per_trial/100

keys=construct_key(state_action_combos +['rwd'],epochs)
allresults={k+'_'+ep:[] for k in keys.values() for ep in epochs}
allresults['params']={p:[] for p in params} #to store list of parameters
resultslist={k+'_'+ep:[] for k in keys.values() for ep in epochs}
resultslist['params']={p:[] for p in params}

st2=0
a2=0
for st1 in state_thresh:
    for a1 in alpha1: 
        params['numQ']=1
        params['state_thresh']=[round(st1,3),round(st2,3)] #threshold on prob for creating new state 
        # higher means more states. 
        params['alpha']=[round(a1,3),round(a2,3)]
        params['Hx_len']=Hx_len
        for p in allresults['params'].keys():
            allresults['params'][p].append(params[p])                #
            resultslist['params'][p].append(params[p])                #
        results={sa:{'Beg':[],'End':[]} for sa in state_action_combos+['rwd']}
        for r in range(runs):
            results=run_sim(params,numevents,noise,results,state_action_combos,trial_subset,printR)
        allresults,resultslist=save_results(results,epochs,allresults,keys,resultslist)
        resultslist['params']['max_correct']=max_correct
        fname='Sequence_paramsHxLen'+str(params['Hx_len'])+'_Q'+'_'.join([str(params['numQ']),str(round(st1,3)),str(round(a1,3))])
        np.savez(fname,allresults=allresults,params=params,reslist=resultslist)
fname='Sequence_paramsHxLen'+str(params['Hx_len'])+'_Q'+str(params['numQ'])+'_all'
np.savez(fname,allresults=allresults,params=params,reslist=resultslist) 

