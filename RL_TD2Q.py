# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:01:56 2021
1. Restructured:
    a. Move RL into separate file
    b. condense select agent gaus and select agent Euclid into single selection action with two distance calculations
2. Use Boltzman to choose action among different Q matrix actions
    Need to use a higher beta - similar to max - to reproduce the previous good results
@author: kblackw1
"""
import numpy as np

##########################################################
class RL:
    """Reinforcement learning by interaction of Environment and Agent"""

    def __init__(self, environment, agent, states,actions,R,T,Aparams,Eparams,oldQ={},printR=False):
        """Create the environment and the agent"""
        self.env = environment(states,actions,R,T,Eparams,printR)
        #self.agent = agent(self.env.T.keys(), self.env.actions,Aparams,oldQ)
        self.agent = agent(self.env.actions,Aparams,oldQ)
        self.vis = True  # visualization
        self.name=None #will be named later
            
    def episode(self, tmax=50,noise=0,cues=[],info=False,name=None,block_DA=False):
        self.name=name
        state = self.env.start() #state tuple, (0,0) to start
        self.results={'state': [state], 'reward':[0],'action':[0]} 
        action = self.agent.start(state,cues) #1st action is random 
        # Repeat interaction
        if info:
            print('start episode, from Q=', self.agent.Q,'\nresults',self.results)
        for t in range(1, tmax+1):
            reward, state = self.env.step(action,prn_info=info) #determine new state and reward from env
            #possibly do steps in blocks of trials?  Estimate reward prob for each block?
            action = self.agent.step(reward, state, noise,cues=cues,prn_info=info,block_DA=block_DA) #determine next action from current state and reward
            self.append_results(action,reward)
        return 

    def append_results(self,action,reward):
        self.results['state'].append(self.env.state)
        self.results['reward'].append(reward)
        self.results['action'].append(action)
    
    def visual(self,title=None): #The state graph will only show two types of states, not three
        """Visualize state,action,reward of an eipsode"""
        import matplotlib.pyplot as plt
        plt.ion()
        fig,ax=plt.subplots(nrows=3,ncols=1,sharex=True)
        if title is not None:
            fig.suptitle(title)
        xvals=np.arange(len(self.results['reward']))
        for i,key in enumerate(['reward','action']):
            ax[i].plot(xvals,self.results[key], marker='*',label=key)
            ax[i].set_ylabel(key)
            ax[i].legend()
        ax[-1].set_xlabel('events')
        offset=0.1
        for i,((st,lbl),symbol) in enumerate(zip(self.env.state_types.items(),['k.','b*'])):
            yval=[s_tup[st]+i*offset for s_tup in self.results['state']]
            ax[2].plot(xvals,yval,marker=symbol[-1],color=symbol[0],label=lbl,linestyle='None')
            #ax[2].plot(xvals,yval,label=lbl)
        ax[2].set_ylabel('state')
        ax[2].legend()
        plt.show()

    def state_to_words(self,nn,noise,chars=3):
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
                    env_states[ii][jj]=list(self.env.states[key].keys())[list(self.env.states[key].values()).index(int(np.round(si)))][0:chars]
            for jj,si in enumerate(st[env_bits:]):
                env_states[ii][jj+env_bits]=str(si)
        return env_states
    
    def set_of_plots(self,learn_phase,noise,title2='',hist=0):
        self.visual(learn_phase+title2) #states vs time and actions vs time     
        if hist:
            for ii in range(len(self.agent.Q)): #Q matrix
                self.agent.visual(self.agent.Q[ii],labels=self.state_to_words(ii,noise),
                             title=learn_phase+' Q'+str(ii))
            if hist>1:
                self.agent.plot_learn_history(title=learn_phase+', numQ='+str(ii+1)) 
    
    def count_state_action(self,allresults,sa_combo,event_subset):
        #2021 jan 4: added multiply rewared by self.agent.events_per_trial to get mean reward per trial
        #2021 mar: make events_per_Trial an agent parameter
        learn_phase=self.name
        actions=[]
        act_results={}
        trial_subset=event_subset/self.agent.events_per_trial
        for sa in sa_combo:
            anum=self.env.actions[sa[1]]
            actions.append(sa[1])
            #for figure title, count actions
            #normalize - convert to actions per trial by dividing by trials
            act_results[sa[1]]={}
            act_results[sa[1]]['Beg']=self.results['action'][0:event_subset].count(anum)/trial_subset
            act_results[sa[1]]['End']=self.results['action'][-event_subset:].count(anum)/trial_subset
            #Now, count how many times that state=state and action=action
            state=sa[0]
            state0num=self.env.states[self.env.state_types[0]][state[0]]
            state1num=self.env.states[self.env.state_types[1]][state[1]]
            timeframe={'Beg':range(event_subset),'End':range(-event_subset,0)}
            for tf,trials in timeframe.items():
                sa_count=0
                for tr in trials:
                    #count number of times that agent state is state0 and state1
                    if self.results['action'][tr]==anum and \
                        self.results['state'][tr]==(state0num,state1num):
                            #print(sa,tf,self.results['action'][tr],self.results['state'][tr],sa_count)
                            sa_count+=1
                allresults[learn_phase][sa][tf].append(sa_count/trial_subset)
                #print(learn_phase,sa,tf,trials,sa_count)
        result_str=' '.join([','+a+'= B:'+str(np.round(act_results[a]['Beg'],3))+
                             ',E:'+ str(np.round(act_results[a]['End'],3))
                             for a in np.unique(actions)])
        allresults[learn_phase]['rwd']['Beg'].append(np.mean(self.results['reward'][0:event_subset])*self.agent.events_per_trial)             
        allresults[learn_phase]['rwd']['End'].append(np.mean(self.results['reward'][-event_subset:])*self.agent.events_per_trial)             
        return allresults,result_str  

    def trajectory(self,traject,sa_combo, events_per_block,saphase=None):
        if saphase is None:
            saphase=self.name
        phase=self.name
        num_blocks=int(len(self.results['reward'])/events_per_block)
        for sa in sa_combo[saphase]:
            if sa=='rwd':
                traject[phase]['rwd'].append([self.agent.events_per_trial*np.mean(self.results['reward'][block*events_per_block:(block+1)*events_per_block]) for block in range(num_blocks)])
            else:    
                anum=self.env.actions[sa[1]]
                state=sa[0]
                state0num=self.env.states[self.env.state_types[0]][state[0]]
                state1num=self.env.states[self.env.state_types[1]][state[1]]
                block_count=[]
                for block in range(num_blocks):
                    sa_count=0
                    for tr in range(block*events_per_block,(block+1)*events_per_block):
                        if self.results['action'][tr]==anum and\
                            self.results['state'][tr]==(state0num,state1num):
                                sa_count+=1
                    block_count.append(sa_count)
                traject[phase][sa].append(block_count)
        return traject

