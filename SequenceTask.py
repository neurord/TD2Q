# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:39:14 2020

@author: kblackw1
"""
import numpy as np

import sequence_env as task_env
import agent_twoQtwoSsplit as QL
import RL_utils as rlu
##########################################################
class RL:
    """Reinforcement learning by interaction of Environment and Agent"""

    def __init__(self, environment, agent, states,actions,R,T,Aparams,Eparams,oldQ={},printR=False):
        """Create the environment and the agent"""
        self.env = environment(states,actions,R,T,Eparams,printR)
        #self.agent = agent(self.env.T.keys(), self.env.actions,Aparams,oldQ)
        self.agent = agent(self.env.actions,Aparams,oldQ)
        self.vis = True  # visualization
        self.name=None
        self.results={'state': [], 'reward':[],'action':[]}
            
    def episode(self, tmax=50,noise=0,cues=[],info=False):
        state = self.env.start() #state tuple, (0,0) to start
        reward=0
        action = self.agent.start(state,cues) 
        self.append_results(action,reward)
        # Repeat interaction
        if info:
            print('start episode, from Q=', self.agent.Q,'\nresults',self.results)            
        for t in range(1, tmax+1):
            reward, state = self.env.step(action,prn_info=info) #determine new state and reward from env
            #print('t=',t,'state',state,'=',self.env.state_from_number(state),'reward=',reward)
            action = self.agent.step(reward, state, noise,cues=cues,prn_info=info) #determine next action from current state and reward
            self.append_results(action,reward)
        return 

    def append_results(self,action,reward):
        self.results['state'].append(self.env.state)
        self.results['reward'].append(reward)
        self.results['action'].append(action)
    
    def visual(self,title=None):
        """Visualize state,action,reward of an eipsode"""
        import matplotlib.pyplot as plt
        plt.ion()
        fig,ax=plt.subplots(nrows=3,ncols=1,sharex=True)
        if title is not None:
            fig.suptitle(title)
        xvals=np.arange(len(self.results['reward']))
        for i,key in enumerate(['reward','action']):
            ax[i].plot(xvals,self.results[key], label=key)
            ax[i].set_ylabel(key)
            ax[i].legend()
        ax[-1].set_xlabel('time')
        offset=0.1
        for i,((st,lbl),symbol) in enumerate(zip(self.env.state_types.items(),['k.','bx'])):
            yval=[s_tup[st]+i*offset for s_tup in self.results['state']]
            ax[2].plot(xvals,yval,marker=symbol[-1],color=symbol[0],label=lbl,linestyle='None')
        ax[2].set_ylabel('state')
        ax[2].legend()

    def state_to_words(self,nn,noise,hx_len):
        env_states=[];env_st_num=[]
        env_bits=len(self.env.states.keys())
        for st in self.agent.ideal_states[nn].values():
            env_st_num.append([np.round(si,1) for si in st])
            env_states.append([])
            for si in st:
                env_states[-1].append('--')
        for ii,st in enumerate(env_st_num):
            for jj,si in enumerate(st[0:env_bits]):
                key=list(self.env.states.keys())[jj]
                if np.abs(int(si))-np.abs(si)<=noise and int(np.round(si)) in self.env.states[key].values():
                    env_states[ii][jj]=list(self.env.states[key].keys())[list(self.env.states[key].values()).index(int(np.round(si)))][0:hx_len]
            for jj,si in enumerate(st[env_bits:]):
                env_states[ii][jj+env_bits]=str(si)
        return env_states
    
    def set_of_plots(self,numQ,noise,hx_len,title2='',hist=False):
        import matplotlib.pyplot as plt
        plt.ion()
        self.visual(numQ+'Q'+title2) #differs from RL_TD2Q in parameter numQ vs learn_phase , and using hx_len    
        for ii in range(len(self.agent.Q)):
            self.agent.visual(self.agent.Q[ii],labels=self.state_to_words(ii,noise,hx_len),
                         title=numQ+'Q, Q'+str(ii+1))
        if hist:
            self.agent.plot_learn_history(title=numQ+'Q, Q'+str(ii+1))
    
    def get_statenum(self,state): ####### Not part of RL_TD2Q
        if state[0] in self.env.states['loc']:
            state0num=self.env.states['loc'][state[0]]
        else:
            state0num=-1 #this occurs if wildcard specified as action
        #wildcard can be used to specify location or a characteri in press_hx
        #If wildcard is used, need to find all possible matching states
        matching_state1=[]
        if state[1] in self.env.states['hx']:
            state1num=self.env.states['hx'][state[1]]
        else:
            state1num=-1
            #star_index=state[1].find('*')#will only find first occurrence
            star_index=[i for i, letter in enumerate(state[1]) if letter =='*']
            #list of possible matching states to *LL
            for st in self.env.states['hx']:
                if np.all([state[1][i]==st[i] for i in range(len(st)) if i not in star_index]):
                    matching_state1.append(st)
        return state0num,state1num,matching_state1
            
    def count_actions(self,allresults,sa_combo,event_subset,accum_type='mean'): ####### Not part of RL_TD2Q
        #2021 jan 4: added multiply reward by events_per_trial to get mean reward per trial
        trial_subset=event_subset/self.agent.events_per_trial
        for sa in sa_combo:
            state=sa[0]
            anum=self.env.actions[sa[1]]
            state0num,state1num,matching_state1=self.get_statenum(state)
            #count how many times that state=state and action=action
            #print('sa',sa,'matching states',matching_state1)
            timeframe={'Beg':range(event_subset),'End':range(-event_subset,0)}
            actions=np.array(self.results['action'])
            for tf,trials in timeframe.items():
                sa_count=0
                action_indices=np.where(actions[trials]==anum)[0]+trials[0] #indices with correct actions
                #for tr in trials:
                for tr in action_indices:
                    #if self.results['action'][tr]==anum:
                    #count number of times that agent state is state0 and state1
                    if (state[0]=='*' or self.results['state'][tr][0]==state0num) and \
                        (self.results['state'][tr][1]==state1num or self.env.state_from_number(self.results['state'][tr])[1] in matching_state1):
                        sa_count+=1
                allresults[sa][tf].append(sa_count/trial_subset) #events per trial, fraction of responses in specified number of events
        if accum_type=='count':
            max_rwd=np.max(self.results['reward'])             
            allresults['rwd']['Beg'].append(self.results['reward'][0:event_subset].count(max_rwd))/trial_subset #number of rewards per trial
            allresults['rwd']['End'].append(self.results['reward'][-event_subset:].count(max_rwd))/trial_subset #maximum = 1
        else:
            allresults['rwd']['Beg'].append(np.mean(self.results['reward'][0:event_subset])*self.agent.events_per_trial) #mean reward per trial
            allresults['rwd']['End'].append(np.mean(self.results['reward'][-event_subset:])*self.agent.events_per_trial)            
        return allresults 

    def trajectory(self,traject,sa_combo, num_blocks,events_per_block,numQphs,accum_type='mean'):  #differs from RL_TD2Q 
        for sa in sa_combo:
            if sa=='rwd':
                if accum_type=='count':
                    max_rwd=np.max(self.results['reward'])  
                    traject[numQphs]['rwd'].append([self.results['reward'][block*events_per_block:(block+1)*events_per_block].count(max_rwd) for block in range(num_blocks)]) #rewards per block
                else:
                    traject[numQphs]['rwd'].append([self.agent.events_per_trial*np.mean(self.results['reward'][block*events_per_block:(block+1)*events_per_block]) for block in range(num_blocks)])
            else:    
                anum=self.env.actions[sa[1]]
                state=sa[0]
                state0num,state1num,matching_state1=self.get_statenum(state)
                block_count=[]
                for block in range(num_blocks):
                    sa_count=0
                    for tr in range(block*events_per_block,(block+1)*events_per_block):
                        if self.results['action'][tr]==anum:
                            #count number of times that agent state is state0 and state1
                            if (state[0]=='*' or self.results['state'][tr][0]==state0num) and \
                                (self.results['state'][tr][1]==state1num or self.env.state_from_number(self.results['state'][tr])[1] in matching_state1):
                                    sa_count+=1
                    block_count.append(sa_count)
                traject[numQphs][sa].append(block_count)
        return traject

def accum_Qhx(states,actions,rl,numQ,Qhx=None):
    #find the state number corresponding to states for each learning phase
    state_nums={state: {q: [] for q in range(numQ)} for state in states}
    for q in range(numQ):
        int_ideal_states=[(int(v[0]),int(v[1])) for v in rl.agent.ideal_states[q].values()]
        int_ideal_state1=[int(v[1]) for v in rl.agent.ideal_states[q].values()]
        for state in states:
            st0,st1,matching_states=rl.get_statenum(state)
            if len(matching_states)==0 and st1>-1:
                matching_states=[state[1]]
            for ms in matching_states:
                hx_num=rl.env.states['hx'][ms]
                if st0>-1:
                    if (st0,hx_num) in int_ideal_states:
                        qindex=int_ideal_states.index((st0,hx_num))
                        state_nums[state][q].append((state[0]+','+ms,qindex)) 
                        #print(state,',match:', ms,',num',st0,hx_num,'in Q:',qindex)
                    #else:
                        #print(state,',match:', ms,',num',st0,hx_num,'Not found')
                else:
                    qindices=np.where(np.array(int_ideal_state1)==hx_num)[0]                
                    #print(state,',match:', ms,',num',st0,hx_num,'in Q:',qindices)
                    for qnum in qindices:
                        state_pair=list(int_ideal_states[qnum])
                        state_words=rl.env.state_from_number(state_pair)
                        state_nums[state][q].append((state_words[0][0:3]+','+ms,qnum))
    if not Qhx:
        Qhx={st:{q:{ph[0]:{ac:[] for ac in actions} for ph in state_nums[st][q]} for q in state_nums[st].keys()} for st in state_nums.keys()} 
    for st in state_nums.keys(): 
         for qv in state_nums[st].keys():
            for (ph,qindex) in state_nums[st][qv]:
                if ph in Qhx[st][qv].keys(): #not all states are visited each run
                    for ac in actions.keys():
                        Qhx[st][qv][ph][ac].append(rl.agent.Qhx[qv][:,qindex,rl.agent.actions[ac]])
                else:
                    Qhx[st][qv][ph]={ac:[] for ac in actions}
                    for ac in actions.keys():
                        Qhx[st][qv][ph][ac].append(rl.agent.Qhx[qv][:,qindex,rl.agent.actions[ac]])
    #need to return state_nums? which may differ for each run
    return Qhx,state_nums

##########################################################
if __name__ == "__main__":
    from SequenceTaskParam import Hx_len,rwd
    from SequenceTaskParam import params,states,act
    from SequenceTaskParam import Tloc,R,env_params

    numtrials=600 # 450 #
    runs=15
    #If want to add reward and time since reward to cues, need to divide by ~100
    noise=0.01 #make noise small enough or state_thresh small enough to minimize new states in acquisition
    #control output
    printR=False #print environment Reward matrix
    Info=False #print information for debugging
    plot_hist=0#1: plot Q, 2: plot the time since last reward
    other_plots=True
    save_data=True #write output data in npz file
    Qvalues=[1,2] #simulate using these values for numQ, make this [2] to simulate inactivation
    inactivate=None #set to None to skip the inactivation test at the end 'D1', 'D2'
    inactivate_blocks=0# 1 or 3 for inactivate = 'D1' or 'D2', 0 otherwise

    ########## Plot Q values over time for these states and actions 
    plot_Qhx=True    
    actions_colors={'goL':'r','goR':'b','press':'k','goMag':'grey'}

    if Hx_len==3:
        #MINIMUM actions for reward = 6, so maximum rewards = 1 per 6 "trials"
        state_action_combos=[(('*','*LL'), 'goR'),(('Rlever','*LL'),'press'),(('Rlever','LLR'),'press')]
    elif Hx_len==4:
        #MINIMUM actions for reward = 7, so maximum rewards = 1 per 7 "trials"
        state_action_combos=[(('Llever','---L'), 'press'),(('Llever','**RL'), 'press'),(('Llever','**LL'), 'goR'),(('Rlever','**LL'),'press'),(('Rlever','*LLR'),'press'),(('Rlever','LLRR'),'goMag')]
        overstay=[(('Llever','**LL'), act) for act in ['goL','goMag','press','other']]+\
            [(('Rlever','LLRR'),act) for act in ['goL','goR','press','other']]   #'**LL' instead of --LL
        premature=[(('Llever','**RL'), act) for act in ['goL','goR','goMag','other']]+\
            [(('Rlever','**RL'), act) for act in ['goR','goMag','other','press']]+\
            [(('Llever','---L'), act) for act in ['goL','goR','goMag','other']]+\
            [(('Rlever','---L'), act) for act in ['goR','goMag','other','press']]+\
           [(('Rlever','*LLR'), act) for act in ['goL','goR','goMag','other']] #'*LLR' instead of -LLR
        start=[(('mag','----'), act) for act in ['goL','goR','goMag','press','other']]
        state_action_combos=state_action_combos+overstay+premature+start
        sa_errors={'stay':overstay,'switch':premature,'start':start}
    else:
        print('unrecognized press history length')
    
    plot_Qstates=[state[0] for state in state_action_combos]
    numevents=numtrials*params['events_per_trial'] #number of events/actions allowed for agent per run/trial
    trial_subset=int(0.05*numtrials)*params['events_per_trial']# display mean reward and count actions over 1st and last of these number of trials 
    epochs=['Beg','End']
   
    trials_per_block=10
    events_per_block=trials_per_block* params['events_per_trial']
    num_blocks=int((numevents+1)/events_per_block)
    #optionally add blocks of runs with D1 or D2 inactivated
    #update some parameters
    params['distance']='Euclidean'
    params['wt_learning']=False
    params['decision_rule']=None #'delta'#'combo', 'delta', 'sumQ2', None means use choose_winner
    params['Q2other']=0.0  #heterosynaptic syn plas of Q2 for other actions
    params['forgetting']=0#0.2 #heterosynaptic decrease Q1 for other actions
    params['beta_min']=0.5#params['beta'] #0.1 is only slightly worse#
    params['beta']=3
    params['gamma']=0.95
    params['beta_GPi']=10
    params['moving_avg_window']=3 
    params['initQ']=-1 #-1 means do state splitting.  If initQ=0, 1 or 10, it means initialize Q to that value and don't split
    params['D2_rule']= None #'Ndelta' #'Bogacz' #'Opal'### Opal: use Opal update without critic, Ndelta: calculate delta for N matrix from N values
    params['rwd']=rwd['reward']
    #lower means more states
    state_thresh={'Q1':[0.75,0],'Q2':[0.75,0.875]} #without normalized ED, with heterosynaptic LTP 
    state_thresh={'Q1':[0.75,0],'Q2':[0.75,0.625]} #or st2= 0.875 with normalized ED, with heterosynaptic LTD
    alpha={'Q1':[0.2,0],'Q2':[0.2,0.35]}    
    #params['state_thresh']=[0.25,0.275]#[0.15,0.2] #threshold on prob for creating new state using Gaussian mixture
    # higher means more states. Adjusted so that in new context, Q2 creates new states, but not Q1
    #params['alpha']=[0.3,0.15] # [0.2,0.14] # double learning to learn in half the trials, slower for Q2 - D2 neurons
    output_data={q:{} for q in Qvalues}
    all_Qhx={q:[] for q in Qvalues}
    all_beta={q:[] for q in Qvalues}
    all_lenQ={q:{qq:[] for qq in range(1,q+1)} for q in Qvalues}
    numchars=6 
    state_subset=['RRLL','RLLR']

    params['events_per_block']=events_per_block
    params['trials_per_block']=trials_per_block
    params['trial_subset']=trial_subset
    params['inact']=inactivate 
    params['inact_blocks']=inactivate_blocks

    sa_keys=rlu.construct_key(state_action_combos +['rwd'],epochs)
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
            if runs==1:
                print('&&&&&&&&&&&&&&&&&&&& STATES',states,'\n     ****  R:',R.keys(),'\n   ****   T:',Tloc.keys())
            ######### acquisition trials, context A, only 6 Khz + L turn allowed #########
            acq = RL(task_env.separable_T, QL.QL, states,act,R,Tloc,params,env_params,printR=printR)
            acq.episode(numevents,noise=noise,info=Info)
            if params['inact'] and numQ==2:
                if params['inact']=='D2':
                    acq.agent.Q[1]=np.zeros(np.shape(acq.agent.Q[1]))
                    acq.agent.alpha[1]=0
                    #acq.agent.numQ=1
                    params['Da_factor']=acq.agent.Da_factor=0.5
                elif params['inact']=='D1':
                    acq.agent.Q[0]=np.zeros(np.shape(acq.agent.Q[0]))
                    acq.agent.alpha[0]=0
                    params['Da_factor']=acq.agent.Da_factor=2                    
                acq.episode(events_per_block*params['inact_blocks'],noise=noise,info=Info)
                #acq.set_of_plots('LLRR, numQ='+str(params['numQ']),noise,Hx_len,hist=plot_hist)
            results[numQ]=acq.count_actions(results[numQ],state_action_combos,trial_subset,accum_type='mean')#,accum_type='count')
            traject_dict=acq.trajectory(traject_dict, sa_keys,num_blocks+params['inact_blocks'],events_per_block,numQ,accum_type='mean')#,accum_type='count')
            if r<1 and other_plots:
                acq.set_of_plots(str(numQ),noise,Hx_len,title2='',hist=plot_hist)
                #acq.visual()
            if plot_Qhx: #need to return state_nums, which may differ for each run
                Qhx,state_nums=accum_Qhx(plot_Qstates,actions_colors,acq,params['numQ'],Qhx)
            #del acq #to free up memory
            print('numQ=',numQ,', run',r,'Q0 mat states=',len(acq.agent.Q[0]),'alpha',acq.agent.alpha)
            all_beta[numQ].append(acq.agent.learn_hist['beta'])
            for qq in all_lenQ[numQ].keys():
                if qq-1 in acq.agent.learn_hist['lenQ'].keys():
                    all_lenQ[numQ][qq].append(acq.agent.learn_hist['lenQ'][qq-1])
            all_Qhx[numQ]=Qhx
        resultslist=rlu.save_results(results,sa_keys,resultslist)      
        for p in resultslist[numQ]['params'].keys():               #
            resultslist[numQ]['params'][p].append(params[p])                #
        for ta in traject_dict[numQ].keys():
            output_data[numQ][ta]={'mean':np.mean(traject_dict[numQ][ta],axis=0),'sterr':np.std(traject_dict[numQ][ta],axis=0)/np.sqrt(runs-1)}
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(' Using',params['numQ'], 'Q, alpha=',params['alpha'],'thresh',params['state_thresh'], 'runs',runs,'of total events',numevents)
        print(' weights:',[k+':'+str(params[k]) for k in params.keys() if k.startswith('wt')])
        print('Q2 hetero=',params['Q2other'],'decision rule=',params['decision_rule'],'beta=',params['beta_min'],params['beta'])
        print('counts from ',trial_subset,' events: BEGIN    END    std over ',runs,'runs. Hx_len=',Hx_len)
        norm={sac:100 for sac in results[numQ].keys()}
        norm['rwd']=1
        for sa_combo,counts in results[numQ].items():
            print(sa_combo,':::',np.round(np.mean(counts['Beg'])*norm[sa_combo],2),'% +/-',np.round(np.std(counts['Beg'])*norm[sa_combo],2),
                    ',', np.round(np.mean(counts['End'])*norm[sa_combo],2),'% +/-',np.round(np.std(counts['End'])*norm[sa_combo],2))
        if other_plots:
            for i in range(params['numQ']):
                acq.agent.visual(acq.agent.Q[i],labels=acq.state_to_words(i,noise,numchars),title='numQ='+str(numQ)+',Q'+str(i),state_subset=state_subset)
        
        if save_data:
            import datetime
            dt=datetime.datetime.today()
            date=str(dt).split()[0]
            key_params=['numQ','Q2other','beta_GPi','decision_rule','beta_min','beta','gamma','rwd']
            fname_params=key_params+['initQ']
            fname='Sequence'+date+'_'.join([k+str(params[k]) for k in fname_params])
            #fname='Sequence'+date+'_HxLen'+str(Hx_len)+'_alpha'+'_'.join([str(a) for a in params['alpha']])+'_st'+'_'.join([str(st) for st in params['state_thresh']])+\

            if params['inact']:
                fname=fname +'_inactive'+params['inact']+'_'+str(params['Da_factor'])
            np.savez(fname,par=params,results=resultslist[numQ],traject=output_data[numQ],Qhx=all_Qhx[numQ],all_beta=all_beta[numQ],all_lenQ=all_lenQ[numQ],sa_errors=sa_errors)
            allQ={i:acq.agent.Q[i] for i in range(params['numQ'])}
            all_labels={i:acq.state_to_words(i,noise,numchars) for i in range(params['numQ'])}
            actions=acq.agent.actions
    print('\nsummary for beta_min=',params['beta_min'],'beta_max=',params['beta'],'beta_GPi=', params['beta_GPi'],'gamma=',params['gamma'])
    for numQ in Qvalues:
        halfrwd=(np.max(output_data[numQ]['rwd']['mean'])+np.min(output_data[numQ]['rwd']['mean']))/2
        ####### replace this with 90% of maximal to measure effect of beta? #############
        block=np.min(np.where(output_data[numQ]['rwd']['mean']>halfrwd))
        print('rwd End',':::',round(np.mean(results[numQ]['rwd']['End'])*norm['rwd'],2), \
                'per trial, +/-',round(np.std(results[numQ]['rwd']['End'])*norm['rwd'],2), \
                ', blocks to half reward=',block, 'for nQ=', numQ)
    #
    title='History Length '+str(Hx_len)+'\nminimum '+str(params['events_per_trial'])+' actions per reward'
    if other_plots:
        rlu.plot_trajectory(output_data,title,[Qvalues])
    if plot_Qhx:
        plot_states=[('Llever','--LL'),('Rlever','-LLR'),('Rlever','LLRR')]#[('Llever','RRLL'),('Rlever','LLLL'),('Rlever','RLLR'),('Rlever','RRLL')]
        actions_lines={a:'solid' for a in actions_colors.keys()}
        from TD2Q_Qhx_graphs import plot_Qhx_sequence, plot_Qhx_sequence_1fig
        figs=plot_Qhx_sequence_1fig (all_Qhx,plot_states,actions_colors,params['events_per_trial'],actions_lines)
        #for numQ,Qhx in all_Qhx.items():
            #figs=plot_Qhx_sequence(Qhx,actions_colors,params['events_per_trial'],numQ)
            #figs=plot_Qhx_sequence_1fig (Qhx,plot_states,actions_colors,params['events_per_trial'])


