# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:26:59 2020

@author: kblackw1
"""

import numpy as np
from RL_class import Environment
import copy

############ In fact, this can be used with any task where states are tuples
class completeT(Environment): 
    """Specific Environment:
        """    
    def __init__(self, states,actions,R,T,params,printR=False):
        self.state_types={v:k for v,k in enumerate(states.keys())}
        self.states=copy.copy(states)
        self.num_states={st:len(v) for st,v in states.items()}
        self.Ns=len(T)
        print('   ## env init, num states:',self.Ns, 'states types:',self.state_types,'\n   states',self.states,self.num_states)
        self.actions=copy.copy(actions)
        self.Na = len(self.actions)
        super().__init__(self.Ns,self.Na)        
        self.R=copy.deepcopy(R) #reward matrix
        self.T=copy.deepcopy(T) #transition matrix
        self.start_state=copy.copy(params['start'])
        if printR:
            reward_thresh_for_printing=0
            print('########## R ############')
            for s in self.R.keys():
                s_words=self.state_from_number(s)
                print('state:',s, '=',s_words)
                for a in self.R[s].keys():
                    if np.any([rw[0]>reward_thresh_for_printing and rw[1]>0 for rw in self.R[s][a]]):
                      print('reward of ',self.R[s][a], 'for state,action pair:',
                            s_words,self.action_from_number(a))
            for a in self.R[s].keys():
                print('action:',a, '=', list(self.actions.keys())[list(self.actions.values()).index(a)])
       
    def state_from_number(self,s):
        return [list(self.states[self.state_types[i]].keys())[list(self.states[self.state_types[i]].values()).index(t)] for i,t in enumerate(s)]
    def action_from_number(self,a):
        return list(self.actions.keys())[list(self.actions.values()).index(a)]
   
    def step(self, action,prn_info=False):
        """step by an action"""
        #reward from taking action in state
        #R[s][a] and T[s][a] are list of tuples, 
        #in each tuple 1st value, e.g. [0] is reward/new state, 2nd, e.g. p[1] is prob
        num_choices=len(self.R[self.state][action])
        weights=[p[1] for p in self.R[self.state][action]]
        if np.sum(weights)!=1.0:
            print('Reward probs do not sum to 1',self.state,self.state_from_number(self.state),action,self.action_from_number(action))
        choice = np.random.choice(num_choices,p=weights) #probabalistic reward 
        self.reward=self.R[self.state][action][choice][0] #0 contains reward
        if prn_info and np.abs(self.reward)>2:
            print('******* env reward', self.reward,'state,action',self.state,action)
        #Determine new state from taking action in state
        if len(self.T[self.state][action])!=len(self.R[self.state][action]):
            num_choices=len(self.T[self.state][action])
            #print('***********env',self.state,action,num_choices)
            Tweights=[p[1] for p in self.T[self.state][action]]
            #if len(Tweights)>1:
            #    print('W',weights,'TW',Tweights, 'choice',choice,'state',self.state,'ACTION',action,'\nR',self.R[self.state][action],self.T[self.state][action])
            if np.sum(Tweights)!=1.0:
                print('transition probs do not sum to 1',self.state,self.state_from_number(self.state),action,self.action_from_number(action))
            if Tweights!=weights:
                choice=np.random.choice(num_choices,p=Tweights) #transition selection is separate from reward selection
            #if self.reward>9:
            #    print ('prior state',self.state, 'new state',self.T[self.state][action][choice][0])
        self.state=self.T[self.state][action][choice][0]
        return self.reward, self.state
    
    def start(self):
        """start an episode"""
        self.state = self.start_state
        return self.state

    def encode(self, state):
        i=list(self.R.keys()).index(state)
        return i
    #
    def decode(self, i):
        st=list(self.R.keys())[i]
        return st
    
    
