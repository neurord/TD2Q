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
from TD2Q_Qhx_graphs import Qhx_multiphase,Qhx_multiphaseNewQ
import sys

from BanditTask import calc_fraction_left,plot_prob_tracking,accum_meanQ,calc_meanQ,shift_stay_list,opal_params

def parse_args(commandline,params):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('vary',type=str,help='parameter to vary, required')
    parser.add_argument('-bmin',type=float,help='value of beta_min',default=params['beta_min'])
    parser.add_argument('-OpAL',type=bool,help='whether to implement OpAL or not',default=False)
    parser.add_argument('-bconst',type=bool,help='whether to make beta min = beta max',default=False)
    parser.add_argument('-dr',type=str,help='decision rule', default=None)
    args = parser.parse_args(commandline) # maps arguments (commandline) to choices, and checks for validity of choices.
    return args

def update_params(pars,params):
    vary_param=pars.vary
    params['decision_rule']=pars.dr
    params['use_Opal']=pars.OpAL
    params['beta_min'] = pars.bmin
    return params,vary_param

def summary_data(output_summary,results,traject_dict,all_counts,runs,vary_param,params,key_params):
    fractionLeft,noL,noR,ratio=calc_fraction_left(traject_dict,runs)
    popt,pcov,delta,RMSmean,RMSstd,RMS=plot_prob_tracking(ratio,fractionLeft,runs,showplot=False)
        
    for phs in results.keys():
        newline=','.join([str(params[k]) for k in key_params])
        if vary_param == 'alpha':
            newline=','.join([str(p) for p in params['alpha']])
        newline=newline+','+phs+','+str(round(ratio[phs],2))+','+str(round(np.nanmean(fractionLeft[phs]),2))+','+str(round(np.nanstd(fractionLeft[phs]),2))
        newline=newline+','+str(noL[phs])+','+str(noR[phs])+','+str(round((noL[phs]+noR[phs])/runs,3))
        for key,counts in all_counts.items():
            ss_ratio=[stay/(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift']) if stay+shift>0 ]
            events=[(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift'])]
            newline=newline+','+key+','+str(round(np.mean(ss_ratio),3))+','+str(round(np.std(ss_ratio),3))+','+ str(np.mean(events))
        output_summary.append(newline+',')
    tot_rwd=np.sum([np.mean(results[k]['rwd']['End']) for k in results.keys()])
    rwd_var=np.sum([np.var(results[k]['rwd']['End']) for k in results.keys()])
    newline=','.join([str(params[k]) for k in key_params])
    newline=newline+',TOTAL,'+str(round(tot_rwd,2))+','+str(round(np.sqrt(rwd_var),2))+','+str(round(RMSmean,3))+','+str(round(RMSstd,3))+','+str(round(delta,3))
    output_summary.append(newline)
    return output_summary

if __name__ == "__main__":
    from BanditTaskParam import params, env_params, states,act, Rbandit, Tbandit
    from BanditTaskParam import loc, tone, rwd,include_wander

    args = sys.argv[1:]
    #for debugging
    #ARGS="beta"# -bmin 0.1"
    #args=ARGS.split()

    events_per_trial=params['events_per_trial']  #this is task specific
    trials=100 
    numevents= events_per_trial*trials
    runs=40 #Hamid et al uses 14 rats. 40 gives smooth trajectories
    noise=0.15 #0.15 if start from oldQ, 0.01-0.05 if start new each phase.make noise small enough or state_thresh small enough to minimize new states in acquisition.  
     #control output
    printR=False #print environment Reward matrix
    Info=False#print information for debugging
    plot_hist=0#1: plot final Q, 2: plot the time since last reward, etc.

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
    params['initQ']=-1 #split states, initialize Q values to best matching
    params['D2_rule']= None #'Bogacz' #'Ndelta' #'Opal'#'Bogacz' ### Opal: use Opal update without critic, Ndelta: calculate delta for N matrix from N values
    params['step1']=False
    use_oldQ=True
    params['use_Opal']=False
    non_rwd=rwd['base'] 
    params['Da_factor']=1
    state_thresh={'Q1':[1.0,0],'Q2':[0.75,0.625]} #For normalized Euclidean distance
    alpha={'Q1':[0.6,0],'Q2':[0.4,0.2]}    #For normalized Euclidean distance, 2x discrim values works with 100 trials
 
    params['state_thresh']=state_thresh['Q'+str(params['numQ'])] #for euclidean distance, no noise
    #lower means more states for Euclidean distance rule
    params['alpha']=alpha['Q'+str(params['numQ'])] #  

    pars=parse_args(args,params)
    params,vary_param=update_params(pars,params)
    ################# For OpAL ################
    if params['use_Opal']: #use critic instead of RPEs, and use Opal learning rule
        params=opal_params(params)
    ######################################  
    epochs=['Beg','End']
    keys=rlu.construct_key(action_items +['rwd'],epochs)

    Qhx_states=[('Pport','6kHz'), ('start','blip')]
    Qhx_actions=['left','right','center']

    ### to plot performance vs trial block
    trials_per_block=10
    events_per_block=trials_per_block* events_per_trial
    num_blocks=int((numevents+1)/events_per_block)
    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset

    extra_acts=['hold'] #to make task simpler and demonstrate results not due to complexity of task
    if include_wander:
        extra_acts=['hold', 'wander']
    
    key_params=['numQ','beta_GPi','decision_rule','beta_min','beta','gamma','use_Opal','step1','D2_rule','initQ']
    output_summary=[]
    header=','.join(key_params)+',phase,rwd_P(L),rwdstd_P(L),noL_RMS,noR_RMSstd,noLR_delta,ss_ratio,ss_std,events per run'
    output_summary.append(header)
    vary_param_values={'gamma':[0.1, 0.3,0.45,0.6,0.75,0.82,0.9,0.95,0.98],'beta':[0.9, 1.5, 2, 3, 5],'alpha':[0.05,0.1,0.2,0.3,0.5,0.8]}
    which_alpha=1 #not relevant unless vary_param='alpha'
    for new_val in vary_param_values[vary_param]: 
        if vary_param=='alpha':
            params['alpha'][alpha]=new_val
        else:
            params[vary_param]=new_val
        if pars.bconst:
            params['beta_min']=new_val
        resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
        traject_dict={phs:{ta:[] for ta in traject_items[phs]} for phs in learn_phases}
        results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}
        resultslist['params']={p:[] for p in params.keys()}
        random_order=[]
        key_list=list(prob_sets.keys())
        all_counts={'left_rwd':{},'left_none':{},'right_rwd':{},'right_none':{}}
        for key,counts in all_counts.items():
            for phase in learn_phases:
                counts[phase]={'stay':[0]*runs,'shift':[0]*runs}
        wrong_actions={aaa:[0]*runs for aaa in extra_acts} 
        all_beta=[];all_lenQ=[];all_Qhx=[]; all_bounds=[]; all_ideals=[];all_RT=[]
        mean_Qhx={q:{st:{aa:[] for aa in Qhx_actions } for st in Qhx_states} for q in range(params['numQ'])}
        for r in range(runs):
            #randomize prob_sets
            acqQ={};acq={};beta=[];lenQ={q:[] for q in range(params['numQ'])};RT=[]
            random_order.append([k for k in key_list]) #keep track of order of probabilities
            print('*****************************************************\n************** run',r,'prob order',key_list)
            for phs in key_list:
                prob=prob_sets[phs]
                Tbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),prob['L']),((loc['Lport'],tone['error']),1-prob['L'])] #hear tone in poke port, go left, in left port/success
                Tbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),prob['R']),((loc['Rport'],tone['error']),1-prob['R'])]
                Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],prob['L']),(non_rwd,1-prob['L'])]   #lick in left port - 90% reward   
                Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],prob['R']),(non_rwd,1-prob['R'])] 

                if use_oldQ:
                    acq[phs] = RL(tone_discrim.completeT, QL.QL, states,act,Rbandit,Tbandit,params,env_params,printR=printR,oldQ=acqQ)
                else:
                    acq[phs] = RL(tone_discrim.completeT, QL.QL, states,act,Rbandit,Tbandit,params,env_params,printR=printR) #start each epoch from init
                results,acqQ=rlu.run_sims(acq[phs], phs,numevents,trial_subset,action_items,noise,Info,cues,-1,results,phist=plot_hist)
                traject_dict=acq[phs].trajectory(traject_dict, traject_items,events_per_block)
                beta.append(acq[phs].agent.learn_hist['beta'])
                RT.append([np.mean(acq[phs].agent.RT[x*events_per_trial:(x+1)*events_per_trial]) for x in range(trials)] )
                for q,qlen in acq[phs].agent.learn_hist['lenQ'].items():
                    lenQ[q].append(qlen)
            np.random.shuffle(key_list) #shuffle after run complete, so that first run does 50:50 first    
            ###### Count stay vs shift         
            all_counts,responses=shift_stay_list(acq,all_counts,rwd,loc,tone,act,r)
            #store beta, lenQ, Qhx, boundaries,ideal_states from the set of phases in a single trial/agent    
            all_beta.append([b for bb in beta for b in bb])
            all_RT.append([b for bb in RT for b in bb])
            all_lenQ.append({q:[b for bb in lenQ[q] for b in bb] for q in lenQ.keys()})
            agents=list(acq.values()) 
            if use_oldQ:
                Qhx, boundaries,ideal_states=Qhx_multiphase(Qhx_states,Qhx_actions,agents,params['numQ'])
            else:  #### sort agents by name (prob), otherwise the mean will be meaningless
                Qhx, boundaries,ideal_states=Qhx_multiphaseNewQ(Qhx_states,Qhx_actions,agents,params['numQ'])
            all_bounds.append(boundaries)
            all_Qhx.append(Qhx)
            all_ideals.append(ideal_states)
            mean_Qhx=accum_meanQ(Qhx,mean_Qhx)
        if not use_oldQ: #do not average across Qvalues if a) starting from previous and b) random order
            mean_Qhx,all_Qhx=calc_meanQ(mean_Qhx,all_Qhx)
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

        output_summary=summary_data(output_summary,results,traject_dict,all_counts,runs,vary_param,params,key_params)

        import datetime
        dt=datetime.datetime.today()
        date=str(dt).split()[0]
        fname_params=[k for k in key_params]
        if vary_param=='alpha':
            fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])+'_alpha'+str(alpha)+str(params['alpha'][alpha])+'_rwd'+str(rwd['reward'])+'_'+str(rwd['none'])+'_wander'+str(include_wander)
        else:
            fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])+'_rwd'+str(rwd['reward'])+'_'+str(rwd['none'])+'_wander'+str(include_wander)
        np.savez(fname,par=params,results=resultslist,traject=output_data,traject_dict=traject_dict,shift_stay=all_counts,rwd=rwd)
        if runs>10 and not use_oldQ: 
            all_Qhx=[mean_Qhx]
            all_bounds=[all_bounds[0]] #all are the same, no need to save all of them
            ideals=[all_ideals[0]] #not all the same, but not used in Qhx graph
        np.savez('Qhx'+fname,all_Qhx=all_Qhx,all_bounds=all_bounds,events_per_trial=events_per_trial,phases=key_list,all_ideals=all_ideals,all_beta=all_beta,all_lenQ=all_lenQ)
    if vary_param == 'alpha':
        fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])+vary_param+str(which_alpha)+'range'
    else:
        fname_params.remove(vary_param) 
        if pars.bconst:
            fname_params.remove('beta_min')
        fname='Bandit'+date+'_'.join([k+str(params[k]) for k in fname_params])+vary_param+'range'
    np.save(fname,output_summary)
