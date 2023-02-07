# -*- coding: utf-8 -*-
"""
Created on 2022 Dec 5

@author: kblackw1
"""

############ reward   ################
from BanditTaskParam import params,rwd,validate_T,validate_R
rwd['reward']=rwd['reward']+2*rwd['base']
act={'left':0,'right':1} 
states={'loc':{'Pport':1},
        'tone':{'6kHz':6}} 
params['state_units']={'loc':False,'tone':False} #Try false/true
start=(states['loc']['Pport'],states['tone']['6kHz']) #used many times
env_params={'start':start}
loc=states['loc'] #used only to define R and T
tone=states['tone'] #used only to define R and T
params['events_per_trial']=1

Rbandit={};Tbandit={}  #dictionaries to improve readability/prevent mistakes
prwdR=0.8; prwdL=0.5 #initial values.  These change with each block of trials

Tbandit={start:{act['left']:[(start,1)],act['right']:[(start,1)]}}
Rbandit={start:{act['left']:[(rwd['reward'],prwdL),(rwd['base'],1-prwdL)], \
                act['right']: [(rwd['reward'],prwdR),(rwd['base'],1-prwdR)]}}

if __name__== '__main__':
    ######## Make sure all needed transitions have been created
    validate_T(Tbandit,msg='validate bandit T')
    validate_R(Tbandit,Rbandit,msg='validate bandit R')
