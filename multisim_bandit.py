# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:14:25 2021

@author: kblackw1
"""
import numpy as np
import completeT_env as tone_discrim
import agent_twoQtwoSsplit as QL
from RL_TD2Q import RL
import RL_utils as rlu
from TD2Q_Qhx_graphs import Qhx_multiphase

from BanditTask import calc_fraction_left,plot_prob_tracking

if __name__ == "__main__":
    step1=False
    if step1:
        from Bandit1stepParam import params, env_params, states,act, Rbandit, Tbandit
        from Bandit1stepParam import loc, tone, rwd
    else:
        from BanditTaskParam import params, env_params, states,act, Rbandit, Tbandit
        from BanditTaskParam import loc, tone, rwd
    events_per_trial=params['events_per_trial']  #this is task specific
    trials=100 
    numevents= events_per_trial*trials
    runs=40 #Hamid et al uses 14 rats. 40 gives smooth trajectories
    noise=0.15 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    #control output
    printR=False #print environment Reward matrix
    Info=False#print information for debugging
    plot_hist=0#1: plot final Q, 2: plot the time since last reward, etc.
    plot_Qhx=1 #2D or 3D plot of Q dynamics.  if 1, then plot agent responses
    print_shift_stay=True

    action_items=[(('Pport','6kHz'),'left'),(('Pport','6kHz'),'right')]
    ########### #For each run, randomize the order of this sequence #############
    prob_sets={'50:50':{'L':0.5, 'R': 0.5},'10:50':{'L':0.1,'R':0.5},
            '90:50':{'L':0.9, 'R': 0.5},'90:10':{'L':0.9, 'R': 0.1},
            '50:90':{'L':0.5, 'R': 0.9},'50:10':{'L':0.5, 'R': 0.1},'10:90':{'L':0.1,'R':0.9}}
    #           '91:49':{'L':0.91, 'R': 0.49},'49:91':{'L':0.49, 'R': 0.91},'49:11':{'L':0.49, 'R': 0.11},'11:49':{'L':0.11,'R':0.49}} 
    prob_sets=dict(sorted(prob_sets.items(),key=lambda item: float(item[0].split(':')[0])/float(item[0].split(':')[1]),reverse=True))

    #prob_sets={'20:80':{'L':0.2,'R':0.8},'80:20':{'L':0.8,'R':0.2}}
    learn_phases=list(prob_sets.keys())

    traject_items={phs:action_items+['rwd'] for phs in learn_phases}
    cues=[]
    trial_subset=int(0.1*numevents) #display mean reward and count actions over 1st and last of these number of trials 
    #update some parameters of the agent
    params['Q2other']=0.0
    params['numQ']=2
    params['beta_min']=0.5 #increased exploration when rewards are low
    params['beta']=1.5
    params['beta_GPi']=10
    params['gamma']=0.82
    params['moving_avg_window']=3  #This in units of trials, the actual window is this times the number of events per trial
    params['decision_rule']= None #'delta' #'mult' #
    params['split']=True
    non_rwd=rwd['base'] 
    state_thresh={'Q1':[1.0,0],'Q2':[0.75,0.625]} #For normalized Euclidean distance
    alpha={'Q1':[0.6,0],'Q2':[0.4,0.2]}    #For normalized Euclidean distance, 2x discrim values works with 100 trials
 
    params['state_thresh']=state_thresh['Q'+str(params['numQ'])] #for euclidean distance, no noise
    #lower means more states for Euclidean distance rule
    params['alpha']=alpha['Q'+str(params['numQ'])] #  

    epochs=['Beg','End']
    keys=rlu.construct_key(action_items +['rwd'],epochs)

    Qhx_states=[('Pport','6kHz')]
    Qhx_actions=['left','right']

    ### to plot performance vs trial block
    trials_per_block=10
    events_per_block=trials_per_block* events_per_trial
    num_blocks=int((numevents+1)/events_per_block)
    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset

    output_summary=[]
    key_params=['numQ','Q2other','beta_GPi','decision_rule','beta_min','beta','gamma']
    header=','.join(key_params)+',phase,P(L)_rwdmean,P(L)_rwdstd,noL_RMSm,noR_RMSs,Persev_Delta'
    output_summary.append(header)
    vary_param='beta' #'gamma' # 'numQ' #
    for new_val in [0.9, 1.5, 2, 3, 5]: #[0, 0.3,0.45,0.6,0.75,0.82,0.9,0.95,0.98]: #[1,2]: # 
        #params['beta_min']=new_val
        params[vary_param]=new_val
        resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
        traject_dict={phs:{ta:[] for ta in traject_items[phs]} for phs in learn_phases}
        results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}
        resultslist['params']={p:[] for p in params.keys()}
        random_order=[]
        key_list=list(prob_sets.keys())

        all_beta=[];all_lenQ=[];all_Qhx=[]; all_bounds=[]; all_ideals=[];all_RT=[]
        for r in range(runs):
            #randomize prob_sets
            acqQ={};acq={};beta=[];lenQ={q:[] for q in range(params['numQ'])};RT=[]
            random_order.append([k for k in key_list]) #keep track of order of probabilities
            print('*****************************************************\n************** run',r,'prob order',key_list)
            for phs in key_list:
                prob=prob_sets[phs]
                if not step1:
                    Tbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),prob['L']),((loc['Lport'],tone['error']),1-prob['L'])] #hear tone in poke port, go left, in left port/success
                    Tbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),prob['R']),((loc['Rport'],tone['error']),1-prob['R'])]
                Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],prob['L']),(non_rwd,1-prob['L'])]   #lick in left port - 90% reward   
                Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],prob['R']),(non_rwd,1-prob['R'])] 

                acq[phs] = RL(tone_discrim.completeT, QL.QL, states,act,Rbandit,Tbandit,params,env_params,printR=printR,oldQ=acqQ)
                results,acqQ=rlu.run_sims(acq[phs], phs,numevents,trial_subset,action_items,noise,Info,cues,-1,results,phist=plot_hist)
                traject_dict=acq[phs].trajectory(traject_dict, traject_items,events_per_block)
                beta.append(acq[phs].agent.learn_hist['beta'])
                RT.append([np.mean(acq[phs].agent.RT[x*events_per_trial:(x+1)*events_per_trial]) for x in range(trials)] )
                for q,qlen in acq[phs].agent.learn_hist['lenQ'].items():
                    lenQ[q].append(qlen)
            np.random.shuffle(key_list) #shuffle after run complete, so that first run does 50:50 first    
            #store beta, lenQ, Qhx, boundaries,ideal_states from the set of phases in a single trial/agent    
            all_beta.append([b for bb in beta for b in bb])
            all_RT.append([b for bb in RT for b in bb])
            all_lenQ.append({q:[b for bb in lenQ[q] for b in bb] for q in lenQ.keys()})
            agents=list(acq.values()) 
            Qhx, boundaries,ideal_states=Qhx_multiphase(Qhx_states,Qhx_actions,agents,params['numQ'])
        all_bounds.append(boundaries)
        all_Qhx.append(Qhx)
        all_ideals.append(ideal_states)
        all_ta=[];output_data={}
        for phs in traject_dict.keys():
            output_data[phs]={}
            for ta in traject_dict[phs].keys():
                all_ta.append(ta)
                output_data[phs][ta]={'mean':np.mean(traject_dict[phs][ta],axis=0),'sterr':np.std(traject_dict[phs][ta],axis=0)/np.sqrt(runs-1)}
        all_ta=list(set(all_ta))
        #move reward to front
        all_ta.insert(0, all_ta.pop(all_ta.index('rwd')))
        for p in resultslist['params'].keys():             #
            resultslist['params'][p].append(params[p])                #
        resultslist=rlu.save_results(results,keys,resultslist)

        fractionLeft,noL,noR,ratio=calc_fraction_left(traject_dict,runs)
        popt,pcov,delta,RMSmean,RMSstd,_=plot_prob_tracking(ratio,fractionLeft,runs,showplot=False)
        for phase in results.keys():
            newline=','.join([str(params[k]) for k in key_params])
            newline=newline+','+phase+','+str(round(ratio[phase],2))+','+str(round(np.nanmean(fractionLeft[phase]),2))+','+str(round(np.nanstd(fractionLeft[phase]),2))
            newline=newline+','+str(noL[phase])+','+str(noR[phase])+','+str(round((noL[phase]+noR[phase])/runs,3))
            output_summary.append(newline+',')
        tot_rwd=np.sum([np.mean(results[k]['rwd']['End']) for k in results.keys()])
        rwd_var=np.sum([np.var(results[k]['rwd']['End']) for k in results.keys()])
        newline=','.join([str(params[k]) for k in key_params])
        newline=newline+',TOTAL,'+str(round(tot_rwd,2))+','+str(round(np.sqrt(rwd_var),2))+','+str(round(RMSmean,3))+','+str(round(RMSstd,3))+','+str(round(delta,3))
        output_summary.append(newline)

        import datetime
        dt=datetime.datetime.today()
        date=str(dt).split()[0]
        fname_params=key_params+['split']
        fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])#+'_1step'
        np.savez(fname,par=params,results=resultslist,traject=output_data,traject_dict=traject_dict)
        np.savez('Qhx'+fname,all_Qhx=all_Qhx,all_bounds=all_bounds,params=params,phases=key_list,
                   all_ideals=all_ideals,random_order=random_order,num_blocks=num_blocks,all_beta=all_beta,all_lenQ=all_lenQ)
    fname_params.remove(vary_param)
    fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])+vary_param+'range'
    np.save(fname,output_summary)

##################################
'''
dir='NormEuclidPLoSsubmission2_Q2other0/'
data=np.load(dir+fname,allow_pickle=True)
res=data['results'].item()
means=[np.mean(res[phs]['rwd__End']) for phs in res.keys() if phs != 'params']   
vars=[np.var(res[phs]['rwd__End']) for phs in res.keys() if phs != 'params']    
tot_rwd=np.sum(means)
tot_std=np.sqrt(np.sum(vars))
#splitFalse: rwd=22.59 +/- 5.75
'''