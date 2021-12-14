# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:26:59 2020

@author: kblackw1
"""

import numpy as np
from RL_class import Environment
############ In fact, this can be used with any task where states are tuples
class separable_T(Environment): 
    """Specific Environment:
        """    
    def __init__(self, states,actions,R,T,params,printR=False):
        self.state_types={v:k for v,k in enumerate(states.keys())} #GOOD 
        self.states=states #GOOD - not really enumerating all states
        self.num_states={st:len(v) for st,v in states.items()}  #GOOD - not reall enumerating all states
        self.Ns=self.num_states[self.state_types[0]]*self.num_states[self.state_types[1]]
        print('   ## env init, num states:',self.Ns, 'states types:',self.state_types,'\n   states',self.states,self.num_states)
        self.actions=actions #Fine
        self.Na = len(self.actions)  #Fine
        super().__init__(self.Ns,self.Na)        
        self.R=R #reward matrix
        self.T=T #transition matrix
        self.Hx_len=params['hx_len']
        self.start_state=params['start']
        self.hx_act=params['hx_act']
        self.env_state_bits=0 #which parts of agent state are also env states, make this parameter.
        if printR:
            reward_thresh_for_printing=0
            print('########## R ############')
            for s in self.R.keys():
                s_words=self.state_from_number(s)
                print('state:',s, '=',s_words)
                for a in self.R[s].keys():
                    if np.any([rw[0]>reward_thresh_for_printing for rw in self.R[s][a]]):
                      print('reward of ',self.R[s][a], 'for state,action pair:',
                            s_words,self.action_from_number(a))
            for a in self.R[s].keys():
                print('action:',a, '=', list(self.actions.keys())[list(self.actions.values()).index(a)])
                
    def state_from_number(self,s):
        return [list(self.states[self.state_types[i]].keys())[list(self.states[self.state_types[i]].values()).index(t)] for i,t in enumerate(s)]
    def action_from_number(self,a):
        return list(self.actions.keys())[list(self.actions.values()).index(a)]
   
    def T_for_action_hx(self,lever):
        #shift
        presses=self.pressHx[1:]
        #add new press
        self.pressHx=presses+lever
        return self.pressHx
    
    def step(self, action,prn_info=False):
        """step by an action"""
        #reward from taking action in state
        #R[s][a] and T[s][a] are list of tuples, 
        #in each tuple 1st value, e.g. [0] is reward/new state, 2nd, e.g. p[1] is prob
        act=self.action_from_number(action)
        num_choices=len(self.R[self.state][action])
        weights=[p[1] for p in self.R[self.state][action]]
        choice = np.random.choice(num_choices,p=weights) #probabalistic reward 
        self.reward=self.R[self.state][action][choice][0]
        if prn_info and np.abs(self.reward)>2:
            print('******* env step, reward=', self.reward,',state=',self.state,self.state_from_number(self.state),',action=',action,act)
        #Determine new state (location or other external state) from taking action in state
        num_choices=len(self.T[self.state[self.env_state_bits]][action]) #How to select action if T doesn't include pressHx
        weights=[p[1] for p in self.T[self.state[self.env_state_bits]][action]]
        choice=np.random.choice(num_choices,p=weights) #new state
        state_loc=self.T[self.state[self.env_state_bits]][action][choice][0]
        #Now determine press history state
        env_state=self.state_from_number(self.state)[self.env_state_bits]
        if act==self.hx_act and env_state.endswith('lever'):
            new_presshx= self.T_for_action_hx(env_state[0])  #1st character of location part of state         
            state_press=self.states['hx'][new_presshx] 
        else:
            state_press=self.state[1] #press_hx part of state
        newstate=(state_loc,state_press)
        #once reward is received, must scramble the press_hx to prevent agent from getting numerous rewards
        #this is kluge, to avoid enumerating all possible transitions
        if self.reward>0:
            newstate=self.start_state
        self.state=newstate
        #print('new state',self.state,self.state_from_number(self.state))
        return self.reward, self.state
    
    def start(self):
        """start an episode"""
        self.state = self.start_state
        #print('start trial from ',self.state_from_number(self.state))
        self.pressHx=self.state_from_number(self.state)[1]
        return self.state

    def encode(self, state):
        i=list(self.R.keys()).index(state)
        return i
    #
    def decode(self, i):
        st=list(self.R.keys())[i]
        return st
