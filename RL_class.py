# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:28:21 2020

@author: kblackw1
"""
import numpy as np

class Environment:
    """Class for a reinforcement learning environment"""
    
    def __init__(self, nstate, naction):
        """Create a new environment"""
        self.Ns = nstate   # number of states
        self.Na = naction  # number of actions
        
    def start(self):
        """start an episode"""
        # randomly pick a state
        self.state = np.random.randint(self.Ns)
        return self.state
    
    def step(self, action):
        """step forward given an action"""
        # random reward
        self.reward = np.random.random()  # between 0 and 1
        # random state transition
        self.state = np.random.randint(self.Ns)
        return self.reward, self.state
    
    def visual(self):
        """visualize envirnment variables"""
        print("r =", self.reward, "; s = ", self.state)
       
class Agent:
    """Class for a reinforcement learning agent"""
    
    def __init__(self, nstate, naction):
        """Create a new agent"""
        self.Ns = nstate   # number of states
        self.Na = naction  # number of actions
        
    def start(self, state):
        """first action, without reward feedback"""
        # randomly pick an action
        self.action = np.random.randint(self.Na)
        return self.action
    
    def step(self, reward, state):
        """learn by reward and take an action"""
        # do nothing for reward
        # randomly pick an action, not constrained by current state
        self.action = np.random.randint(self.Na)
        return self.action
    
    def visual(self):
        """visualize agent variables"""
        print("a =", self.action)

