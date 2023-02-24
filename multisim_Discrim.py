# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:39:14 2020
2021 march: change agent to use euclidean distance instead of Gaussian mixture
        change learning rule for Q2 to use decreases in predicted reward

@author: kblackw1
"""
import numpy as np
import copy
import completeT_env as tone_discrim
import agent_twoQtwoSsplit as QL
from RL_TD2Q import RL
import RL_utils as rlu
from TD2Q_Qhx_graphs import Qhx_multiphase         
from Discrim2stateProbT_twoQtwoS import select_phases, respond_decay

if __name__ == "__main__":
    from DiscriminationTaskParam2 import params,states,act
    events_per_trial=params['events_per_trial']  #this is task specific
    trials=200 #Iino: 180 trials for acq, then 160 trials for discrim; or discrim from the start using 60 trials of each per day (120 trials) * 3 days
    numevents= events_per_trial*trials
    runs=10 #10 for paper
    #control output
    plot_hist=0
    printR=False #print environment Reward matrix
    Info=False#print information for debugging
    #additional cues that are part of the state for the agent, but not environment
    #this means that they do not influence the state transition matrix
    context=[[0],[1]] #set of possible context cues
    noise=0.15 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    #action_items is a subset of the state-action combinations that an agent can perform
    #count number of responses to the following state-action combos:
    action_items=[(('start','blip'),'center'),(('Pport','6kHz'),'left'),(('Pport','6kHz'),'right'),(('Pport','10kHz'),'left'),(('Pport','10kHz'),'right')]
    #action_items=['center','left','right']

    block_DA_dip=False#'AIP' #AIP or no_dip blocks homosynaptic LTP, no_dip blocks heterosynaptic LTD, False - control
    PREE=0
    savings='none'#''none'#'in new context'# 'after extinction'##'none'# #- for simulating discrim and reverse
    extinct='none' #AAB: aquire and extinguish in A, try to renew in B; ABB: aquire in A, extinguish in B, re-test renewal in B
    #Specify which learning protocols/phases to implement
    learn_phases,figure_sets,traject_items,acq_cue,ext_cue,ren_cue,dis_cue,ren_cue,acq2_cue=select_phases(block_DA_dip,PREE,savings,extinct,context,action_items)
    state_sets=[[('Pport','6kHz'),('Pport','10kHz')],[('Pport','6kHz')]]
    phases=[['acquire','discrim','reverse'],['acquire','extinc','renew']] 
    trial_subset=int(0.1*numevents) #display mean reward and count actions over 1st and last of these number of trials 
    #update some parameters of the agent
    params['decision_rule']=None #'delta' #'mult' #  #'combo','sumQ2', None means use direct negative of D1 rule
    params['Q2other']=0.1
    params['numQ']=2
    params['beta_min']=0.5
    params['beta']=1.5
    params['beta_GPi']=10
    params['gamma']=0.82
    params['state_units']['context']=False
    if params['distance']=='Euclidean':
        state_thresh={'Q1':[1.0,0],'Q2':[0.75,0.625]} #For normalized Euclidean distance
        alpha={'Q1':[0.3,0],'Q2':[0.2,0.1]}    #For normalized Euclidean distance
    else:
        state_thresh={'Q1':[0.22, 0.22],'Q2':[0.20, 0.22]} #For Gaussian Mixture?, 
        alpha={'Q1':[0.4,0],'Q2':[0.4,0.2]}    #For Gaussian Mixture? [0.62,0.19] for beta=0.6, 1Q or 2Q;'

    params['state_thresh']=state_thresh['Q'+str(params['numQ'])] #for euclidean distance, no noise
    #lower means more states for Euclidean distance rule
    params['alpha']=alpha['Q'+str(params['numQ'])] #  
    params['split']=True #if False - initialize new row in Q matrix to 0; if True - initialize to Q values of best matching state   
    ######################################
    from DiscriminationTaskParam2 import Racq,Tacq,env_params,Rext,Rdis,Tdis,Rrev,Trev
    epochs=['Beg','End']
    
    keys=rlu.construct_key(action_items +['rwd'],epochs)
    
    ### to plot performance vs trial block
    trials_per_block=10
    events_per_block=trials_per_block* events_per_trial
    num_blocks=int((numevents+1)/events_per_block)
    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset

    output_summary=[]
    key_params=['numQ','Q2other','beta_GPi','decision_rule','beta_min','beta','gamma']
    header=','.join(key_params)+',rwd_mean,rwd_std,half_rwd_block,half_block_std'
    output_summary.append(header)
    vary_param='beta' #'gamma' #
    for new_val in [0.9, 1.5, 2, 3, 5]: #[0.3,0.45,0.6,0.75,0.82,0.9,0.95,0.98]:
        params[vary_param]=new_val
        #params['beta_min']=new_val
        resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
        traject_dict={phs:{ta:[] for ta in traject_items[phs]} for phs in learn_phases}
        #count number of responses to the following actions:
        results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}
        resultslist['params']={p:[] for p in params.keys()} 
        all_beta={'_'.join(k):[] for k in phases}
        all_lenQ={k:{q:[] for q in range(params['numQ'])} for k in all_beta.keys()}
        for r in range(runs):
            rl={}
            if 'acquire' in learn_phases:
                ######### acquisition trials, context A, only 6 Khz + L turn allowed #########
                rl['acquire'] = RL(tone_discrim.completeT, QL.QL, states,act,Racq,Tacq,params,env_params,printR=printR)
                results,acqQ=rlu.run_sims(rl['acquire'],'acquire',numevents,trial_subset,action_items,noise,Info,acq_cue,-1,results,phist=plot_hist,block_DA=block_DA_dip)
                traject_dict=rl['acquire'].trajectory(traject_dict, traject_items,events_per_block)
            if 'extinc' in learn_phases:
                rl['extinc'] = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=acqQ)
                results,extQ=rlu.run_sims(rl['extinc'],'extinc',numevents,trial_subset,action_items,noise,Info,ext_cue,-1,results,phist=plot_hist,block_DA=block_DA_dip)
                traject_dict=rl['extinc'].trajectory(traject_dict, traject_items,events_per_block)
            #### renewal - blocking D2 or Da Dip not tested
            if 'renew' in learn_phases:
                rl['renew'] = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=extQ)
                results,renQ=rlu.run_sims(rl['renew'],'renew',numevents,trial_subset,action_items,noise,Info,ren_cue,-1,results)
                traject_dict=rl['renew'].trajectory(traject_dict, traject_items,events_per_block)
            ####### discrimination trials, add in 10Khz tone, + needed reward and state transitions
            if 'discrim' in learn_phases:
                #use last context in the list
                rl['discrim'] = RL(tone_discrim.completeT, QL.QL, states,act,Rdis,Tdis, params,env_params,oldQ=acqQ)
                acq_first=True
                results,disQ=rlu.run_sims(rl['discrim'],'discrim',int(numevents),trial_subset,action_items,noise,Info,dis_cue,-1,results,phist=plot_hist,block_DA=block_DA_dip)
                traject_dict=rl['discrim'].trajectory(traject_dict, traject_items,events_per_block)    
            ####### reverse trials, switch contingencies ####
            if 'reverse' in learn_phases:
                rl['reverse']=RL(tone_discrim.completeT, QL.QL, states,act,Rrev, Trev,params,env_params,oldQ=disQ)
                results,revQ=rlu.run_sims(rl['reverse'],'reverse',int(numevents),trial_subset,action_items,noise,Info,dis_cue,-1,results,phist=plot_hist)
                traject_dict=rl['reverse'].trajectory(traject_dict, traject_items,events_per_block)
            all_beta,all_lenQ=rlu.beta_lenQ(rl,phases,all_beta,all_lenQ,params['numQ'])

        ######### Average over runs, also need stdev.  
        all_ta=[]; output_data={}
        for phs in traject_dict.keys():
            output_data[phs]={}
            for ta in traject_dict[phs].keys():
                all_ta.append(ta)
                output_data[phs][ta]={'mean':np.mean(traject_dict[phs][ta],axis=0),'sterr':np.std(traject_dict[phs][ta],axis=0)/np.sqrt(runs-1)}
        all_ta=list(set(all_ta))
        #move reward to front
        all_ta.insert(0, all_ta.pop(all_ta.index('rwd')))              #
        for p in resultslist['params'].keys():               #
            resultslist['params'][p].append(params[p])
        resultslist=rlu.save_results(results,keys,resultslist)
        state_act=(('Pport','6kHz'),'left') #
        half,half_block=respond_decay(['acquire','extinc','renew'],state_act,traject_dict)
        ####### append summary results to list #############
        for phase in results.keys():
            newline=','.join([str(params[k]) for k in key_params])
            newline=newline+','+phase+','+str(np.round(np.mean(results[phase]['rwd']['End']),2))+','+str(np.round(np.std(results[phase]['rwd']['End']),2))
        ############## Evaluate decay of responding #####################
            if phase in half_block.keys():
                newline=newline+','+str(round(np.nanmean(half_block[phase]),2))+','+str(round(np.nanstd(half_block[phase]),2))
            else:
                newline=newline+',,'
            output_summary.append(newline)
        newline=','.join([str(params[k]) for k in key_params])
        total_rwd=np.array(results['acquire']['rwd']['End'])+np.array(results['discrim']['rwd']['End'])+np.array(results['reverse']['rwd']['End'])
        newline=newline+',TOTAL,'+str(np.round(np.mean(total_rwd),2))+','+str(np.round(np.std(total_rwd),2))
        output_summary.append(newline)        
        ########################## Save trajectories and Qhx
        actions=['left','right']
        agents=[[rl[phs] for phs in phaseset] for phaseset in phases]
        all_Qhx={q:{} for q in range(params['numQ'])};all_bounds={q:{} for q in range(params['numQ'])};all_ideals={q:{} for q in range(params['numQ'])}
        for ij,(state_subset,phase_set,agent_set) in enumerate(zip(state_sets,phases,agents)):
            Qhx, boundaries,ideal_states=Qhx_multiphase(state_subset,actions,agent_set,params['numQ'])
            for q in Qhx.keys():
                for st in Qhx[q].keys():
                    newstate=','.join(list(st))
                    all_Qhx[q][newstate+' '+str(ij)]=copy.deepcopy(Qhx[q][st])
                    all_ideals[q][newstate+' '+str(ij)]=copy.deepcopy(ideal_states[q][st])
                    all_bounds[q][newstate+' '+str(ij)]=copy.deepcopy(boundaries[q][st])
        del rl        
        import datetime
        dt=datetime.datetime.today()
        date=str(dt).split()[0]
        fname_params=key_params+['split']
        fname='Discrim'+date+'_'.join([k+str(params[k]) for k in fname_params])
        np.savez(fname,par=params,results=resultslist,traject=output_data)
        np.savez('Qhx'+fname,all_Qhx=all_Qhx,all_bounds=all_bounds,events_per_trial=events_per_trial,phases=phases,all_ideals=all_ideals,all_beta=all_beta,all_lenQ=all_lenQ)
    fname_params.remove(vary_param)
    #fname_params.remove('beta_min') #+'_bminbmax_'
    fname='Discrim'+date+'_'.join([k+str(params[k]) for k in fname_params])+vary_param+'range'
    np.save(fname,output_summary)


