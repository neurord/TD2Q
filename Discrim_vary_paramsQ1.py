# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:54:53 2020

@author: kblackw1
"""

import numpy as np

import completeT_env as tone_discrim
import agent_twoQtwoSsplit as QL
from RL_TD2Q import RL
from RL_utils import save_results, construct_key

def run_sims(RL,phase,events,trial_subset,action_items,noise,cues,summary):    
    RL.episode(events,noise=noise,info=False,cues=cues,name=phase)
    rwd_prob=np.mean(RL.agent.learn_hist['rwd_prob'][-trial_subset:])
    summary,t2=RL.count_state_action(summary,action_items,trial_subset)
    Q={'Q':RL.agent.Q,'ideal_states':RL.agent.ideal_states,'learn_weight':RL.agent.learn_weight,'rwd_prob':rwd_prob,'name':phase}
    del RL
    return summary,Q

#control output
save_reward_array=False
printR=False #print environment Reward matrix
from DiscriminationTaskParam2 import env_params, params,states,act, Racq,Tacq, Rext,Rdis,Tdis, Rrev,Trev

events_per_trial=3  #this is task specific
trials=200   #this is task specific - ideally this value is close to the animal behavior trials
numevents= events_per_trial*trials
runs=10  # number of agents to evaluate
trial_subset=int(0.1*numevents) # display mean reward and count actions over 1st and last of these number of trials 

#additional cues that are part of the state for the agent, but not environment
#this means that they do not influence the state transition matrix
context=[[0],[1]] #set of possible context cues
#extinction context needs to be more similar to acquisition context than difference between tone/loc cues
acq_cue=context[0] #use [] for no cues
ext_cue=context[1]
dis_cue=context[0] 
params['state_units']['context']=False
#If want to add reward and time since reward to cues, need to divide by ~100
noise=0.15 #make noise small enough or state_thresh small enough to minimize new states in acquisition
#Specify which learning protocols/phases to implement

learn_phases=['acquire','extinc','renew','discrim','reverse']##'acquire','extinc']#
epochs=['Beg','End']
action_items=[(('start','blip'),'center'),(('Pport','6kHz'),'left'),(('Pport','6kHz'),'right'),(('Pport','10kHz'),'left'),(('Pport','10kHz'),'right')]

#update some parameters of the agent
params['decision_rule']=None#, 'delta', 'sumQ2', None means use direct negative of D1 rule
params['events_per_trial']=events_per_trial
   
keys=construct_key(action_items +['rwd'],epochs)
### allresults - store mean performance vs parameter
resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
allresults={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
allresults['params']={p:[] for p in params.keys()} #to store list of parameters
resultslist['params']={p:[] for p in params.keys()} 
#loop over values for state_Thresh, alpha1,alpha2 here
min_alpha=0.1
max_alpha=0.81
alpha_inc=0.05 #0.05 for sims, 0.2 to test
state_thresh=[0.5,0.625,0.75,0.875,1.0]
st2=0
a2=0
import datetime
dt=datetime.datetime.today()
date=str(dt).split()[0]
for st1 in state_thresh:
    for a1 in np.arange(min_alpha,max_alpha,alpha_inc*2): #at least 2*a2, double increment
        #update some parameters
        print('************ NEW SIM *********',np.round(st1,3),np.round(a1,3))
        params['numQ']=1
        params['state_thresh']=[round(st1,3),round(st2,3)] #threshold on prob for creating new state 
        # higher means more states. 
        params['alpha']=[round(a1,3),round(a2,3)]
        #
        for p in allresults['params'].keys():
            allresults['params'][p].append(params[p])                #
            resultslist['params'][p].append(params[p])
        results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}
        #
        for r in range(runs):
            acq = RL(tone_discrim.completeT, QL.QL, states,act,Racq,Tacq,params,env_params,printR=printR)
            results,acqQ=run_sims(acq,'acquire',numevents,trial_subset,action_items,noise,acq_cue,results)
            ext = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=acqQ)
            results,extQ=run_sims(ext,'extinc',numevents,trial_subset,action_items,noise,ext_cue,results)
            #### renewal
            ren = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=extQ)
            results,renQ=run_sims(ren,'renew',numevents,trial_subset,action_items,noise,acq_cue,results)
            ####### discrimination trials, add in 10Khz tone, + needed reward and state transitions
            dis = RL(tone_discrim.completeT, QL.QL, states,act,Rdis,Tdis, params,env_params,oldQ=acqQ)
            results,disQ=run_sims(dis,'discrim',int(1.5*numevents),trial_subset,action_items,noise,dis_cue,results) 
            rev=RL(tone_discrim.completeT, QL.QL, states,act,Rrev, Trev,params,env_params,oldQ=disQ)
            results,revQ=run_sims(rev,'reverse',int(1.5*numevents),trial_subset,action_items,noise,dis_cue,results)
        allresults,resultslist=save_results(results,epochs,allresults,keys,resultslist)
    fname='Discrim'+date+'_Q'+str(params['numQ'])+'_st'+str(params['state_thresh'][0])
    print('*************** End of a1 loop ************', fname)
    np.savez(fname,allresults=allresults,params=params,reslist=resultslist) 
fname='Discrim'+date+'_Q'+str(params['numQ'])+'_all'
print('**************** End of st1 loop ************', fname)
np.savez(fname,allresults=allresults,params=params,reslist=resultslist)
