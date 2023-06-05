# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:39:14 2020

@author: kblackw1
"""
import numpy as np

import sequence_env as task_env
import agent_twoQtwoSsplit as QL
import RL_utils as rlu
from SequenceTask import RL,accum_Qhx

if __name__ == "__main__":
    from SequenceTaskParam import Hx_len,rwd
    from SequenceTaskParam import params,states,act
    from SequenceTaskParam import Tloc,R,env_params
    numtrials=600 # 450 #
    runs=10
    #If want to add reward and time since reward to cues, need to divide by ~100
    noise=0.01 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    #control output
    printR=False #print environment Reward matrix
    Info=False #print information for debugging
    plot_hist=0#1: plot Q, 2: plot the time since last reward
    other_plots=False
    save_data=True #write output data in npz file
    Qvalues=[1,2] #simulate using these values for numQ, make this [2] to simulate inactivation
    if Hx_len==3:
        #MINIMUM actions for reward = 6, so maximum rewards = 1 per 6 "trials"
        state_action_combos=[(('*','*LL'), 'goR'),(('Rlever','*LL'),'press'),(('Rlever','LLR'),'press')]
    elif Hx_len==4:
        #MINIMUM actions for reward = 7, so maximum rewards = 1 per 7 "trials"
        state_action_combos=[(('Llever','---L'), 'press'),(('Llever','**RL'), 'press'),(('Llever','**LL'), 'goR'),(('Rlever','**LL'),'press'),(('Rlever','*LLR'),'press'),(('Rlever','LLRR'),'goMag')]
        overstay=[(('Llever','**LL'), act) for act in ['goL','goMag','press','other']]+\
            [(('Rlever','LLRR'),act) for act in ['goL','goR','press','other']]
        premature=[(('Llever','**RL'), act) for act in ['goL','goR','goMag','other']]+\
            [(('Llever','---L'), act) for act in ['goL','goR','goMag','other']]+\
            [(('Rlever','*LLR'), act) for act in ['goL','goR','goMag','other']]
        start=[(('mag','----'), act) for act in ['goL','goR','goMag','press','other']]
        state_action_combos=state_action_combos+overstay+premature+start
        sa_errors={'stay':overstay,'switch':premature,'start':start}
    else:
        print('unrecognized press history length')
    
    numevents=numtrials*params['events_per_trial'] #number of events/actions allowed for agent per run/trial
    trial_subset=int(0.05*numtrials)*params['events_per_trial']# display mean reward and count actions over 1st and last of these number of trials 
    epochs=['Beg','End']
    trials_per_block=10
    events_per_block=trials_per_block* params['events_per_trial']
    num_blocks=int((numevents+1)/events_per_block)
    plot_Qstates=[state[0] for state in state_action_combos]
    
    #update some parameters
    params['decision_rule']=None #'delta'#'combo', 'delta', 'sumQ2', None means use choose_winner
    params['Q2other']=0.0  #heterosynaptic syn plas of Q2 for other actions
    params['beta_min']=0.5#params['beta'] #0.1 is only slightly worse#
    params['beta']=3
    params['gamma']=0.9
    params['beta_GPi']=10
    params['rwd']=rwd['reward']
    #lower means more states
    state_thresh={'Q1':[0.75,0],'Q2':[0.75,0.875]} #without normalized ED, with heterosynaptic LTP 
    state_thresh={'Q1':[0.75,0],'Q2':[0.75,0.625]} #or st2= 0.875 with normalized ED, with heterosynaptic LTD
    alpha={'Q1':[0.2,0],'Q2':[0.2,0.35]}    
    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset
    sa_keys=rlu.construct_key(state_action_combos +['rwd'],epochs)
    output_summary=[]
    key_params=['numQ','Q2other','decision_rule','beta','beta_min','gamma','beta_GPi','rwd']
    header=','.join(key_params)+',rwd_mean,rwd_std,half_rwd_block'
    output_summary.append(header)
    vary_param='beta' # 'gamma' #
    for new_val in [0.9, 1.5, 2, 3, 5]: # [0, 0.3,0.45,0.6,0.75,0.82,0.9,0.95,0.98]: # 
        params[vary_param]=new_val
        output_data={q:{} for q in Qvalues}
        all_Qhx={q:[] for q in Qvalues}
        all_beta={q:[] for q in Qvalues}
        all_lenQ={q:{qq:[] for qq in range(1,q+1)} for q in Qvalues}

        results={numQ:{sa:{'Beg':[],'End':[]} for sa in state_action_combos+['rwd']} for numQ in Qvalues}
        resultslist={numQ:{k+'_'+ep:[] for k in sa_keys.values() for ep in epochs} for numQ in Qvalues}
        traject_dict={numQ:{k:[] for k in sa_keys.keys()} for numQ in Qvalues}
        for numQ in Qvalues:
            resultslist[numQ]['params']={p:[] for p in params}
            Qhx=None
            for r in range(runs):
                params['numQ']=numQ
                params['state_thresh']=state_thresh['Q'+str(numQ)]  
                params['alpha']= alpha['Q'+str(numQ)] 
                ######### acquisition trials, context A, only 6 Khz + L turn allowed #########
                acq = RL(task_env.separable_T, QL.QL, states,act,R,Tloc,params,env_params,printR=printR)
                acq.episode(numevents,noise=noise,info=Info)
                results[numQ]=acq.count_actions(results[numQ],state_action_combos,trial_subset,accum_type='mean')#,accum_type='count')
                traject_dict=acq.trajectory(traject_dict, sa_keys,num_blocks,events_per_block,numQ,accum_type='mean')#,accum_type='count')
                Qhx,state_nums=accum_Qhx(plot_Qstates,actions_colors,acq,params['numQ'],Qhx)
                all_beta[numQ].append(acq.agent.learn_hist['beta'])
                for qq in all_lenQ[numQ].keys():
                    if qq-1 in acq.agent.learn_hist['lenQ'].keys():
                        all_lenQ[numQ][qq].append(acq.agent.learn_hist['lenQ'][qq-1])
                del acq #to free up memory
                all_Qhx[numQ].append(Qhx)
            resultslist=rlu.save_results(results,sa_keys,resultslist)      
            for p in resultslist[numQ]['params'].keys():               #
                resultslist[numQ]['params'][p].append(params[p])                #
            for ta in traject_dict[numQ].keys():
                output_data[numQ][ta]={'mean':np.mean(traject_dict[numQ][ta],axis=0),'sterr':np.std(traject_dict[numQ][ta],axis=0)/np.sqrt(runs-1)}
            ####### append summary results to list #############
            newline=','.join([str(params[k]) for k in key_params])
            newline=newline+','+str(np.round(np.mean(results[numQ]['rwd']['End']),2))+','+str(np.round(np.std(results[numQ]['rwd']['End']),2))
            halfrwd=(np.max(output_data[numQ]['rwd']['mean'])+np.min(output_data[numQ]['rwd']['mean']))/2
            block=np.min(np.where(output_data[numQ]['rwd']['mean']>halfrwd))
            newline=newline+','+str(block)
            output_summary.append(newline)
            ######### Save data ###########
            import datetime
            dt=datetime.datetime.today()
            date=str(dt).split()[0]
            #fname='Sequence'+date+'_HxLen'+str(Hx_len)+'_alpha'+'_'.join([str(a) for a in params['alpha']])+'_st'+'_'.join([str(st) for st in params['state_thresh']])+\
            fname_params=key_params+['split']
            fname='Sequence'+date+'_'.join([k+str(params[k]) for k in fname_params])#+'_1step'
            np.savez(fname,par=params,results=resultslist[numQ],traject=output_data[numQ],Qhx=all_Qhx[numQ],all_beta=all_beta[numQ],all_lenQ=all_lenQ[numQ],sa_errors=sa_errors)
    fname_params.remove(vary_param)
    #fname_params.remove('beta_min') #+'_bminbmax_'
    fname='Sequence'+date+'_'.join([k+str(params[k]) for k in fname_params])+vary_param+'range'
    np.save(fname,output_summary)