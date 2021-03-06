# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:12:28 2020

@author: kblackw1
"""

############ reward   ################
rwd={'error':-5,'reward':10,'base':-1,'none':-1} 

######### Parameters for the agent ##################################
params={}
params['wt_learning']=True
params['wt_noise']=False #whether to multiply noise by learning_rate - not helpful
params['numQ']=2
params['alpha']=[0.3,0.06]  # learning rate 0.3 and 0.06 produce learning in 400 trials,#slower for Q2 - D2 neurons 
params['beta']=0.9  # inverse temperature, controls exploration
params['beta_min']=0.1
params['gamma']=0.9  #discount factor 
params['hist_len']=40
params['state_thresh']=[0.12,0.2] #similarity of noisy state to ideal state
#if lower state_creation_threshold for Q[0] compared to Q[1], then Q[0] will have fewer states
#possibly multiply state_thresh by learn_rate? to change dynamically?
params['sigma']=0.25 #similarity of noisey state to ideal state,std used in Mahalanobis distance.
params['time_inc']=0.1 #increment time since reward by this much in no reward
params['moving_avg_window']=5  #This in units of trials, the actual window is this times the number of events per trial
params['decision_rule']=None #'combo', 'delta', 'sumQ2', None ## None means use direct negative of D1 rule
params['Q2other']=0.1
params['forgetting']=0
params['reward_cues']=None ##options: 'TSR', 'RewHx3', 'reward', None
params['distance']='Euclidean'
params['split']=True

############### Make sure you have all the state transitions needed ##########
def validate_T(T,msg=''):
    print(msg)
    for st in T.keys():
        for newst_list in T[st].values():
           for newst in newst_list:
               if newst[0] not in T.keys():
                   print('new state', newst[0],'not in Tacq')

def validate_R(T,R,msg=''):
    print(msg)
    for st in T.keys():
        if st not in R.keys():
            print('state', st,'in T,but not in R')
        else:
            for a in T[st].keys():
                if a not in R[st].keys():
                    print('state/action:', st,a,'in T, but not in R')
   
######### Parameters for the environment ##################################
act={'center':0,'left':1,'return':2,'right':3,'wander':4,'hold':5} 

states={'loc':{'start':0,'Pport':2,'Lport':1,'other':-1,'Rport':3},
        'tone':{'blip':0,'6kHz':6,'success':2,'error':-2,'10kHz':10}} 
params['state_units']={'loc':False,'tone':True} #Try false/true

#some convenient variables
start=(states['loc']['start'],states['tone']['blip']) #used many times
env_params={'start':start}
loc=states['loc'] #used only to define R and T
tone=states['tone'] #used only to define R and T
move=['left','right','wander','center']
stay=['hold']


################# Two arm bandit task of Josh Berke (Hamid et al. Nat Neuro V19)
Rbandit={};Tbandit={}  #dictionaries to improve readability/prevent mistakes
prwdR=0.8; prwdL=0.5 #initial values.  These change with each block of trials

####value of T dict is the new state
#from start - best response is poke
Tbandit[start]={a:[(start,1)] for a in act.values()} # stay at start
for a in ['left','right']:
    Tbandit[start][act[a]]=[((loc['other'],tone['error']),1)] #error if go to left or right port from start box
Tbandit[start][act['wander']]=[((loc['other'],tone['blip']),1)] #meandering, not yet at poke port
Tbandit[start][act['center']]=[((loc['Pport'],tone['6kHz']),1)] #poke at start tone, go to poke port
#What happens if agent doesn't go to poke port immediately
Tbandit[(loc['other'],tone['blip'])]={a:[((loc['other'],tone['blip']),1)] for a in act.values()} #default: remain in other unless
Tbandit[(loc['other'],tone['blip'])][act['center']]=[((loc['Pport'],tone['6kHz']),1)] #go to center port, after wandering
Tbandit[(loc['other'],tone['blip'])][act['return']]=[(start,1)] #return to start

#from poke port - correct response is 'left'
Tbandit[(loc['Pport'],tone['6kHz'])]={a:[((loc['Pport'],tone['6kHz']),1)] for a in act.values()} #default - stay in poke port
for a in ['wander','return']: #incorrect movements
    Tbandit[(loc['Pport'],tone['6kHz'])][act[a]]=[((loc['other'],tone['error']),1)]  #incorrect movements
Tbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),prwdR),((loc['Rport'],tone['error']),1-prwdR)]
Tbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),prwdL),((loc['Lport'],tone['error']),1-prwdL)] #hear tone in poke port, go left, in left port/success

Tbandit[(loc['other'],tone['error'])]={a:[((loc['other'],tone['error']),1)] for a in act.values()}#remain in other unless
Tbandit[(loc['other'],tone['error'])][act['return']]=[(start,1)] #return to start

#from left port or right port - best response is return
Tbandit[(loc['Lport'],tone['success'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} #default, wandering around
Tbandit[(loc['Lport'],tone['success'])][act['hold']]=[((loc['Lport'],tone['error']),1)] #staying at Lport, but not continued success (reward)
Tbandit[(loc['Lport'],tone['success'])][act['return']]=[(start,1)] #go back to start to begin again

Tbandit[(loc['Rport'],tone['success'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} #default, wandering around
Tbandit[(loc['Rport'],tone['success'])][act['hold']]=[((loc['Rport'],tone['error']),1)] #staying at Lport, but not continued success (reward)
Tbandit[(loc['Rport'],tone['success'])][act['return']]=[(start,1)] #go back to start to begin again

#from right or left port with error tone - best response is return
Tbandit[(loc['Rport'],tone['error'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} 
Tbandit[(loc['Rport'],tone['error'])][act['hold']]=[((loc['Rport'],tone['error']),1)] #remain in Rport if no movement,
Tbandit[(loc['Rport'],tone['error'])][act['return']]=[(start,1)] #return to start

Tbandit[(loc['Lport'],tone['error'])] = {act[a]:[((loc['other'],tone['error']),1)] for a in move}
Tbandit[(loc['Lport'],tone['error'])][act['hold']]=[((loc['Lport'],tone['error']),1)] #remain in Lport if no movement,
Tbandit[(loc['Lport'],tone['error'])][act['return']]=[(start,1)] #return to start


#error tone is not associated with penalty except with incorrect response from Pport
for k in Tbandit.keys(): #Tbandit determines what states pairs need reward values
    Rbandit[k]={a:[(rwd['base'],1)] for a in act.values()} #default: cost of basic action
    Rbandit[k][act['hold']]=[(rwd['none'],1)] #not moving - no cost

#reward for correct response
Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],prwdR),(rwd['base'],1-prwdR)]  
Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],prwdL),(rwd['base'],1-prwdL)]

#Error if go anywhere but Left or Right port after tone
for a in ['wander','return']:
    Rbandit[(loc['Pport'],tone['6kHz'])][act[a]]=[(rwd['error'],1)] 
#Error if go straight to left or right port from start box - same error for discrimination
for a in['right','left']:
    Rbandit[(loc['start'],tone['blip'])][act[a]]=[(rwd['error'],1)] 
############## End bandit parameters ###################

if __name__== '__main__':
    ######## Make sure all needed transitions have been created
    validate_T(Tbandit,msg='validate bandit T')
    validate_R(Tbandit,Rbandit,msg='validate bandit R')
