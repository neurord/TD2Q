# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:39:14 2020
2021 march: change agent to use euclidean distance instead of Gaussian mixture
        change learning rule for Q2 to use decreases in predicted reward

@author: kblackw1
"""
import numpy as np

import completeT_env as tone_discrim
import agent_twoQtwoSsplit as QL
from RL_TD2Q import RL
import RL_utils as rlu
import copy
from TD2Q_Qhx_graphs import Qhx_multiphase

def select_phases(block_DA_dip,PREE,savings,extinct,context,action_items):
    acq_cue=context[0]#use [] for no cues
    ext_cue=context[1] #1 for extinction in separate context
    dis_cue=context[0] 
    ren_cue=acq_cue
    acq2_cue=[]
    if block_DA_dip:
        learn_phases=['acquire','extinc','discrim']
        figure_sets=[['extinc','discrim']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd'],'discrim':action_items[1:]+['rwd']}
        ext_cue=context[0] #0 for extinction in same context
    elif PREE: #evaluate how reward prob during acquisition affects rate of extinction
        learn_phases=['acquire','extinc']
        figure_sets=[['acquire','extinc']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd']}
        ext_cue=context[0] #0 for extinction in same context
    elif savings=='after extinction': #evaluate if learn faster the second time            
        learn_phases=['acquire','extinc','acquire2']
        figure_sets=[['acquire','extinc','acquire2']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]],'acquire2':[action_items[1]]+['rwd']}
        ext_cue=context[0] #0 for extinction in same context
        acq2_cue=context[0]
    elif savings=='in new context':#evaluate if learn faster in new context
        learn_phases=['acquire','acquire2']
        figure_sets=[['acquire','acquire2']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'acquire2':[action_items[1]]+['rwd']}
        acq2_cue=context[1] #acquisition in different context
    elif extinct=='AAB':
        learn_phases=['acquire','extinc','renew']
        figure_sets=[['acquire','extinc','renew']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd'],'renew':[action_items[1]]+['rwd']}
        ext_cue=context[0]
        ren_cue=context[1]
    elif extinct=='ABB':
        learn_phases=['acquire','extinc','renew']
        figure_sets=[['acquire','extinc','renew']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd'],'renew':[action_items[1]]+['rwd']}
        ext_cue=context[1]
        ren_cue=context[1]
    elif extinct=='ABA':
        learn_phases=['acquire','extinc','renew']
        figure_sets=[['acquire','extinc','renew']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd'],'renew':[action_items[1]]+['rwd']}
        ext_cue=context[1]
        ren_cue=context[0]
    else: #this is ABA, but with added discrim and reverse
        learn_phases=['acquire','extinc','renew','discrim','reverse']     #
        figure_sets=[['discrim','reverse'],['acquire','extinc','renew']]
        traject_items={'acquire':[action_items[1]]+['rwd'],'extinc':[action_items[1]]+['rwd'],'renew':[action_items[1]]+['rwd'],
                       'discrim':action_items[1:]+['rwd'],'reverse':action_items[1:]+['rwd']}
    return learn_phases,figure_sets,traject_items,acq_cue,ext_cue,ren_cue,dis_cue,ren_cue,acq2_cue

def respond_decay(phases,state_act, output_data):
    half={phs:[] for phs in phases};half_block={phs:[] for phs in phases}
    for phs in phases:
        all_data=output_data[phs][state_act]
        mean_traject=np.mean(all_data,axis=0)
        for data in all_data:
            half[phs].append((np.max(data)+np.min(data))/2)
            if mean_traject.argmax()>mean_traject.argmin():
                h=np.where(data>half[phs][-1])
            else:
                h=np.where(data<half[phs][-1])
            if len(h[0]):
                half_block[phs].append(np.min(h))
            else:
                half_block[phs].append(np.nan)
        print(phs,'half responding=',np.nanmean(half[phs]),'block=',np.nanmean(half_block[phs]))
    return half,half_block

####################################################################################################################
if __name__ == "__main__":
    from DiscriminationTaskParam2 import params,states,act
    events_per_trial=params['events_per_trial']  #this is task specific
    trials=200 #Iino: 180 trials for acq, then 160 trials for discrim; or discrim from the start using 60 trials of each per day (120 trials) * 3 days
    numevents= events_per_trial*trials
    runs=10 #10 for paper
    #control output
    printR=False #print environment Reward matrix
    Info=False#print information for debugging
    plot_hist=0#1: plot final Q, 2: plot the time since last reward
    plot_Qhx=2 #2D or 3D plots of the dynamics of Q
    save_reward_array=True
    #additional cues that are part of the state for the agent, but not environment
    #this means that they do not influence the state transition matrix
    context=[[0],[1]] #set of possible context cues
    #RewHx3 means that agent estimates the reward history as part of the  context
    #extinction context needs to be more similar to acquisition context than difference between tone/loc cues
    #If want to add reward and time since reward to cues, need to divide by ~100
    noise=0.15 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    #action_items is a subset of the state-action combinations that an agent can perform
    #count number of responses to the following state-action combos:
    action_items=[(('start','blip'),'center'),(('Pport','6kHz'),'left'),(('Pport','6kHz'),'right'),(('Pport','10kHz'),'left'),(('Pport','10kHz'),'right')]
    #action_items=['center','left','right']

    block_DA_dip=False #'AIP' #AIP or no_dip blocks homosynaptic LTP, no_dip blocks heterosynaptic LTD, False - control
    PREE=0
    savings='none'#''none'#'in new context'# 'after extinction'##'none'# #- for simulating discrim and reverse
    extinct='none' #AAB: aquire and extinguish in A, try to renew in B; ABB: aquire in A, extinguish in B, re-test renewal in B
    #Specify which learning protocols/phases to implement
    learn_phases,figure_sets,traject_items,acq_cue,ext_cue,ren_cue,dis_cue,ren_cue,acq2_cue=select_phases(block_DA_dip,PREE,savings,extinct,context,action_items)
    #state_sets and phases used for saving Qhx, beta and Qlen
    if block_DA_dip:
        state_sets=[[('Pport','6kHz'),('Pport','10kHz')]]
        phases=[['acquire','discrim']] 
    elif extinct.startswith('A'):
        state_sets=[[('Pport','6kHz')]]
        phases=learn_phases
    else:
        state_sets=[[('Pport','6kHz'),('Pport','10kHz')],[('Pport','6kHz')]]
        phases=[['acquire','discrim','reverse'],['acquire','extinc','renew']] 
    trial_subset=int(0.1*numevents) #display mean reward and count actions over 1st and last of these number of trials 
    #update some parameters of the agent
    params['decision_rule']=None #'mult' #'delta' #  #'combo','sumQ2', None means use direct negative of D1 rule
    params['Q2other']=0.0
    params['numQ']=2
    params['wt_learning']=False
    params['distance']='Euclidean'
    params['beta_min']=0.5
    params['beta']=1.5
    params['beta_GPi']=10
    params['gamma']=0.82
    params['state_units']['context']=False
    params['initQ']=-1 #-1 means do state splitting.  If initQ=0, 1 or 10, it means initialize Q to that value and don't split
    params['D2_rule']= None #'Ndelta' #'Bogacz' #'Opal' #None ### Ndelta: calculate delta from N matrix to update N. Opal: Opal update without critic
    if params['distance']=='Euclidean':
        #state_thresh={'Q1':[0.875,0],'Q2':[0.875,1.0]} #For Euclidean distance
        #alpha={'Q1':[0.2,0],'Q2':[0.2,0.1]}    #For Euclidean distance
        state_thresh={'Q1':[1.0,0],'Q2':[0.75,0.625]} #For normalized Euclidean distance
        alpha={'Q1':[0.3,0],'Q2':[0.2,0.1]}    #For normalized Euclidean distance
    else:
        state_thresh={'Q1':[0.22, 0.22],'Q2':[0.20, 0.22]} #For Gaussian Mixture?, 
        alpha={'Q1':[0.4,0],'Q2':[0.4,0.2]}    #For Gaussian Mixture? [0.62,0.19] for beta=0.6, 1Q or 2Q;'

    params['state_thresh']=state_thresh['Q'+str(params['numQ'])] #for euclidean distance, no noise
    #lower means more states for Euclidean distance rule
    params['alpha']=alpha['Q'+str(params['numQ'])] #  
    traject_title='num Q: '+str(params['numQ'])+' rule:'+str( params['decision_rule'])+' forget:'+str(params['forgetting'])
    ################# For OpAL ################
    params['use_Opal']=False
    if params['use_Opal']:
        params['numQ']=2
        params['Q2other']=0
        params['decision_rule']='delta'
        params['alpha']=[0.1,0.1]#[0.2,0.2]#
        params['beta_min']=1
        params['beta']=1
        params['gamma']=0.1 #called alpha_c in OpAL
        params['initQ']=1 #do not split states, initialize Q values to 1
        params['state_thresh']=[0.75,0.625]
        params['D2_rule']='Opal'
        noise=0.05
    ######################################
    from DiscriminationTaskParam2 import Racq,Tacq,env_params, rwd
    epochs=['Beg','End']
    
    if PREE:
        traject_title+=' PREE:'+str(PREE)
        from DiscriminationTaskParam2 import loc, tone
        Racq[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],PREE),(rwd['base'],1-PREE)]   #lick in left port - 90% reward   
    keys=rlu.construct_key(action_items +['rwd'],epochs)
    resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
    traject_dict={phs:{ta:[] for ta in traject_items[phs]} for phs in learn_phases}
    #count number of responses to the following actions:
    results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}
    Qhx_actions=['left','right'] #actions of interest for Qhx
    
    ### to plot performance vs trial block
    trials_per_block=10
    events_per_block=trials_per_block* events_per_trial
    num_blocks=int((numevents+1)/events_per_block)
    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset
    resultslist['params']={p:[] for p in params.keys()} 
    all_beta={'_'.join(k):[] for k in phases}
    all_lenQ={k:{q:[] for q in range(params['numQ'])} for k in all_beta.keys()}
    all_Qhx=[];all_bounds=[];all_ideals=[]

    for r in range(runs):
        rl={}
        if 'acquire' in learn_phases:
            print('&&&&&&&&&&&&&&&&&&&& acquire for run',r, states,'\n  R:',Racq.keys(),'\n  T:',Tacq.keys(),' cues:',acq_cue)
            ######### acquisition trials, context A, only 6 Khz + L turn allowed #########
            rl['acquire'] = RL(tone_discrim.completeT, QL.QL, states,act,Racq,Tacq,params,env_params,printR=printR)
            results,acqQ=rlu.run_sims(rl['acquire'],'acquire',numevents,trial_subset,action_items,noise,Info,acq_cue,r,results,phist=plot_hist,block_DA=block_DA_dip)
            traject_dict=rl['acquire'].trajectory(traject_dict, traject_items,events_per_block)
        if 'extinc' in learn_phases:
            if runs==1:
                print('&&&&&&&&&&&&&&&&&&&& extinction for run',r, states,' cues:',ext_cue)
            from DiscriminationTaskParam2 import Rext,Tacq
            rl['extinc'] = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=acqQ)
            results,extQ=rlu.run_sims(rl['extinc'],'extinc',numevents,trial_subset,action_items,noise,Info,ext_cue,r,results,phist=plot_hist,block_DA=block_DA_dip)
            traject_dict=rl['extinc'].trajectory(traject_dict, traject_items,events_per_block)
        #### renewal - blocking D2 or Da Dip not tested
        if 'renew' in learn_phases:
            if runs==1:
                print('&&&&&&&&&&&&&&&&&&&& renewal for run',r, states,' cues:',acq_cue)
            rl['renew'] = RL(tone_discrim.completeT, QL.QL, states,act,Rext,Tacq,params,env_params,printR=printR,oldQ=extQ)
            results,renQ=rlu.run_sims(rl['renew'],'renew',numevents,trial_subset,action_items,noise,Info,ren_cue,r,results)
            traject_dict=rl['renew'].trajectory(traject_dict, traject_items,events_per_block)
        ####### discrimination trials, add in 10Khz tone, + needed reward and state transitions
        if 'discrim' in learn_phases:
            #use last context in the list
            from DiscriminationTaskParam2 import Rdis,Tdis
            if runs==1:
                print('&&&&&&&&&&&&&&&&&&&& discrimination for run',r, states,'\n  R:',Rdis.keys(),'\n  T:',Tdis.keys(),' cues:',dis_cue)
            if 'acquire' in learn_phases: #expand previous covariance matrix and Q with new states
                rl['discrim'] = RL(tone_discrim.completeT, QL.QL, states,act,Rdis,Tdis, params,env_params,oldQ=acqQ)
                acq_first=True
            else:
                rl['discrim'] = RL(tone_discrim.completeT, QL.QL, states,act,Rdis,Tdis, params,env_params)
                acq_first=False
            results,disQ=rlu.run_sims(rl['discrim'],'discrim',int(numevents),trial_subset,action_items,noise,Info,dis_cue,r,results,phist=plot_hist,block_DA=block_DA_dip)
            traject_dict=rl['discrim'].trajectory(traject_dict, traject_items,events_per_block)

             #rl['discrim'].set_of_plots('discrim, acquire 1st:'+str(acq_first),noise,t2,hist=plot_hist)
            if Info:
                print('discrim, acquire 1st:'+str(acq_first)+', mean reward=',np.round(np.mean(rl['discrim'].results['reward'][-trial_subset:]),2))
    
        if 'reverse' in learn_phases:
            from DiscriminationTaskParam2 import Rrev,Trev
            if runs==1:
                print('&&&&&&&&&&&&&&&&&&&& reversal',states,'\n  R:',Rrev.keys(),'\n  T:',Trev.keys(),' cues:',dis_cue)
            rl['reverse']=RL(tone_discrim.completeT, QL.QL, states,act,Rrev, Trev,params,env_params,oldQ=disQ)
            results,revQ=rlu.run_sims(rl['reverse'],'reverse',int(numevents),trial_subset,action_items,noise,Info,dis_cue,r,results,phist=plot_hist)
            traject_dict=rl['reverse'].trajectory(traject_dict, traject_items,events_per_block)
        if savings == 'in new context' or savings == 'after extinction':
            if savings == 'in new context' :
                rl['acquire2'] = RL(tone_discrim.completeT, QL.QL, states,act,Racq,Tacq,params,env_params,printR=printR,oldQ=acqQ)
            if savings == 'after extinction':
                rl['acquire2'] = RL(tone_discrim.completeT, QL.QL, states,act,Racq,Tacq,params,env_params,printR=printR,oldQ=extQ)
            results,acq2Q=rlu.run_sims(rl['acq2'],'acquire2',numevents,trial_subset,action_items,noise,Info,acq2_cue,r,results,phist=plot_hist)
            traject_dict=rl['acquire2'].trajectory(traject_dict, traject_items,events_per_block)
            print ('>>>>>>>>>>>>>>>>>>>> savings', savings,'acq2 cue',acq2_cue)
        all_beta,all_lenQ=rlu.beta_lenQ(rl,phases,all_beta,all_lenQ,params['numQ'])

        agents=[[rl[phs] for phs in phaseset] for phaseset in phases]
        one_Qhx={q:{} for q in range(params['numQ'])};one_ideals={q:{} for q in range(params['numQ'])};one_bounds={q:{} for q in range(params['numQ'])}
        for ij,(state_subset,phase_set,agent_set) in enumerate(zip(state_sets,phases,agents)):
            Qhx, boundaries,ideal_states=Qhx_multiphase(state_subset,Qhx_actions,agent_set,params['numQ'])
            for q in Qhx.keys():
                for st in Qhx[q].keys():
                    newstate=','.join(list(st))
                    one_Qhx[q][newstate+' '+str(ij)]=copy.deepcopy(Qhx[q][st])
                    one_ideals[q][newstate+' '+str(ij)]=copy.deepcopy(ideal_states[q][st])
                    one_bounds[q][newstate+' '+str(ij)]=copy.deepcopy(boundaries[q][st])
        all_Qhx.append(one_Qhx)
        all_ideals.append(one_ideals)
        all_bounds.append(one_bounds)
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
    interesting_combos={'acquire':['Pport_6kHz_left_End','rwd__End'],
                        'acquire2':['Pport_6kHz_left_End','rwd__End'],
                        'extinc':['Pport_6kHz_left_Beg','Pport_6kHz_left_End'],
                        'renew':['Pport_6kHz_left_Beg','Pport_6kHz_left_End'],
                        'discrim':['Pport_6kHz_left_Beg','Pport_6kHz_left_End','Pport_10kHz_left_Beg','Pport_10kHz_right_End','rwd__End'],
                        'reverse':['Pport_6kHz_left_End','Pport_10kHz_right_End','Pport_6kHz_right_End','Pport_10kHz_left_End','rwd__End']}
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(' Using',params['numQ'], 'Q, alpha=',params['alpha'],'thresh',params['state_thresh'], 'beta=',params['beta'],'runs',runs,'of total events',numevents)
    print(' apply learning_weights:',[k+':'+str(params[k]) for k in params.keys() if k.startswith('wt')])
    print(' D2_rule=',params['D2_rule'],'decision rule=',params['decision_rule'],'split=',params['initQ'],'critic=',params['use_Opal'])
    print('counts from ',trial_subset,' events (',events_per_trial,' events per trial)          BEGIN    END    std over ',runs,'runs')
    for phase in results.keys():
        for sa,counts in results[phase].items():
            print(phase.rjust(12), sa,':::',np.round(np.mean(counts['Beg']),2),'+/-',np.round(np.std(counts['Beg']),2),
                  ',', np.round(np.mean(counts['End']),2),'+/-',np.round(np.std(counts['End']),2))
        for sa in interesting_combos[phase]:
            if sa in resultslist[phase]:
                print( '            ',sa,':::',[round(val,3) for lst in resultslist[phase][sa] for val in lst] )
    trajectfig=rlu.plot_trajectory(output_data,traject_title,figure_sets)
    print('******* winner count, 1st run *****')
    for phase,ag in rl.items():     
        print(phase,[(wck,np.sum(wcvals)) for wck,wcvals in ag.agent.winner_count.items()])
    if plot_Qhx==2:
        ########## Plot Q values over time 
        ### A. Qhx_multiphase plots only select state/actions, and concatenates multiple learning phases
        from TD2Q_Qhx_graphs import plot_Qhx_2D     
        ########################## NEXT:
        for ij,(state_subset,phase_set,agent_set) in enumerate(zip(state_sets,phases,agents)):
            fig=plot_Qhx_2D(Qhx,boundaries,events_per_trial,phase_set,ideal_states)
        fig=plot_Qhx_2D(all_Qhx[0],all_bounds[0],events_per_trial,phases,all_ideals[0]) #fig 3
    elif plot_Qhx==3: 
        ### B. 3D plot Q history for selected actions, for all states, one graph per phase
        for phase in ['discrim','reverse']:
            rl[phase].agent.plot_Qdynamics(['center','left','right'],'surf',title=rl[phase].name)
    
    if save_reward_array:
        if block_DA_dip:
            fname='DiscrimD2'+block_DA_dip
        else:
            fname='Discrim'
        import datetime
        dt=datetime.datetime.today()
        date=str(dt).split()[0]
        key_params=['numQ','Q2other','beta_GPi','decision_rule','beta_min','beta','gamma','use_Opal','D2_rule']
        fname_params=key_params+['initQ']
        fname=fname+date+'_'.join([k+str(params[k]) for k in fname_params])+'_rwd'+str(rwd['reward'])
        np.savez(fname,par=params,results=resultslist,traject=output_data)
        if plot_Qhx==2:
            np.savez('Qhx'+fname,all_Qhx=all_Qhx,all_bounds=all_bounds,events_per_trial=events_per_trial,phases=phases,all_ideals=all_ideals,all_beta=all_beta,all_lenQ=all_lenQ)
    # Q value plot of subset of states 
    select_states=['success','6kHz','10kHz']
    numchars=3
    state_subset=[ss[0:numchars] for ss in select_states]
    for i in range(params['numQ']):
        rl['discrim'].agent.visual(rl['discrim'].agent.Q[i],labels=rl['discrim'].state_to_words(i,noise,chars=numchars),title='dis Q'+str(i),state_subset=state_subset)
    if save_reward_array:
        numchars=8
        allQ={i:rl['discrim'].agent.Q[i] for i in range(params['numQ'])}
        all_labels={i:rl['discrim'].state_to_words(i,noise,chars=numchars) for i in range(params['numQ'])}
        actions=rl['discrim'].agent.actions
        np.savez('staticQ'+fname,allQ=allQ,labels=all_labels,actions=actions,state_subset=[ss[0:numchars] for ss in select_states])
    
    if not block_DA_dip:
        print('beta_min',params['beta_min'],'beta_max',params['beta'],'beta_GPi',params['beta_GPi'],'rwd',\
            np.mean(results['acquire']['rwd']['End'])+np.mean(results['discrim']['rwd']['End'])+np.mean(results['reverse']['rwd']['End']) )

        ############## Evaluate decay of responding #####################
        state_act=(('Pport','6kHz'),'left') #
        half,half_block=respond_decay(['acquire','extinc','renew'],state_act,traject_dict)
        #identify which panel for plotting fit
        ax=[a for a in trajectfig.axes if 'Ppor' in a.get_ylabel() and '6kHz' in a.get_ylabel()]
        symbols={'acquire':'o','extinc':'d','renew':'X'}
        for jj,(key,x) in enumerate(half_block.items()):
            ax[0].scatter(np.mean(x),np.mean(half[key]),marker=symbols[key],color=ax[0].get_lines()[3*jj].get_c())

