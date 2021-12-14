# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:12:28 2020

@author: kblackw1
"""

############ reward   ################
rwd={'error':-1,'reward':10,'base':-1,'none':0} 

######### Parameters for the agent ##################################
params={}
params['wt_learning']=False
params['wt_noise']=False #whether to multiply noise by learning_rate - not helpful
params['numQ']=1
params['alpha']=[0.3,0.06]  # learning rate 0.3 and 0.06 produce learning in 400 trials,slower for Q2 - D2 neurons
params['beta']=0.9  # inverse temperature, controls exploration
params['beta_min']=0.5
params['gamma']=0.9  #discount factor 
params['hist_len']=40
params['state_thresh']=[0.12,0.2] #similarity of noisy state to ideal state
#if lower state_creation_threshold for Q[0] compared to Q[1], then Q[0] will have fewer states
#possibly multiply state_thresh by learn_rate? to change dynamically?
params['sigma']=0.25 #similarity of noisey state to ideal state,std used in Mahalanobis distance.
params['time_inc']=0.1 #increment time since reward by this much in no reward
params['moving_avg_window']=5 ##This in units of trials, the actual window is this times the number of events per trial
params['decision_rule']=None #'combo', 'delta', 'sumQ2'
params['Q2other']=0.05
params['forgetting']=0
params['reward_cues']=None #options: 'TSR', 'RewHx3', 'reward'
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
act={'goMag':0,'goL':1,'goR':2,'press':3,'other':4}  #other includes grooming, not moving
Hx_len=4 # this specifies the number of for loops #Hx_len=3 works better than 4

hx_values=['L','R'] #possible characters in the history
start_hx='-'*Hx_len #starting press history is 'empty'
'''
WOrse performance if no '-'
hx_values=['L','R'] #possible characters in the history
import numpy
#starting press history is random
start_presses=numpy.random.randint(len(hx_values),size=Hx_len)
start_hx=''.join([hx_values[s] for s in start_presses]) 
'''
### Enumerate all possible 3-way combinations of press history
sequences={}
value=0
if Hx_len==3:
    for c1 in hx_values:
        for c2 in hx_values:
            for c3 in hx_values:
               sequences[c1+c2+c3]=value
               value+=1 ############ Why are states being numbered?  - only for the states dictionary.  
    for c3 in hx_values:
        sequences['--'+c3]=value
        value+=1
        for c2 in hx_values:
            sequences['-'+c2+c3]=value
            value+=1
    sequences['---']=value
elif Hx_len==4:
    for c1 in hx_values:
        for c2 in hx_values:
            for c3 in hx_values:
                for c4 in hx_values:
                    sequences[c1+c2+c3+c4]=value
                    value+=1 ############ Why are states being numbered?  - only for the states dictionary.  
    for c4 in hx_values:
        sequences['---'+c4]=value
        value+=1
        for c3 in hx_values:
            sequences['--'+c3+c4]=value
            value+=1
            for c2 in hx_values:
                sequences['-'+c2+c3+c4]=value
                value+=1
    sequences['----']=value
else:
    print('unanticipated Hx_len in press history')

            
#create state dictionary
states={'loc':{'mag':0,'Llever':1,'Rlever':2,'other':3},
        'hx': sequences} 
params['state_units']={'loc':False,'hx':False}
#some convenient variables
loc=states['loc'] #used only to define R and T
hx=states['hx'] #used only to define R and T

Tloc={loc[location]:{} for location in loc}  #dictionaries to improve readability/prevent mistakes

#two Transition matrices - NOTE, this is the transition for locations
#The transition for lever presses is a function specified in the environment

for location in ['Llever','Rlever','other','mag']:
    Tloc[loc[location]][act['goL']]=[(loc['Llever'],1)]
    Tloc[loc[location]][act['goR']]=[(loc['Rlever'],1)]
    Tloc[loc[location]][act['goMag']]=[(loc['mag'],1)]
    
for location in ['Llever','Rlever','other','mag']:
    for action in ['press','other']:
        Tloc[loc[location]][act[action]]=[(loc[location],1)]


#where to start episodes, and also re-start trial after reward  
start=(states['loc']['mag'],states['hx'][start_hx]) 
#put some environment values into dictionary for ease of param passing
env_params={'start':start,'hx_len':Hx_len,'hx_act':'press'}

#Reward matrix: enumerates all states.  
#Would be nice to avoid such enumeration and create function similar to T
R={}
for k in Tloc.keys(): #T determines what states pairs need reward values
    for st in states['hx'].values():
        R[(k,st)]={a:[(rwd['base'],1)] for a in act.values()} #default: cost of basic action
if Hx_len==3:
    R[(loc['Rlever'],states['hx']['LLR'])][act['press']]=[(rwd['reward'],0.95),(rwd['base'],0.05)]  #95% reward for correct press sequence
elif Hx_len==4:
    for location in loc:
        R[(loc[location],states['hx']['LLRR'])][act['goMag']]=[(rwd['reward'],0.95),(rwd['base'],0.05)]   #95% reward for correct press sequence
else:
    print('unanticipated Hx_len in reward assignment')

if __name__== '__main__':
    ######## Make sure all needed transitions have been created
    validate_T(Tloc,msg='validate Tloc')
    validate_R(Tloc,R,msg='validate R')
    print('press history length=',Hx_len,', start press',start_hx,', hx values',hx_values)
