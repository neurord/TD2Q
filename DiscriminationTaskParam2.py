# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:12:28 2020

@author: kblackw1
"""

############ reward   ################
rwd={'error':-5,'reward':10,'base':-1,'none':0} 

######### Parameters for the agent ##################################
params={}
params['wt_learning']=False
params['wt_noise']=False #whether to multiply noise by learning_rate - not helpful
params['numQ']=1
params['alpha']=[0.3,0.06]  # learning rate 0.3 and 0.06 produce learning in 400 trials,#slower for Q2 - D2 neurons 
params['beta']=0.9  # inverse temperature, controls exploration
params['beta_min']=0.5
params['beta_GPi']=10 #Should be similar to using max
params['gamma']=0.9  #discount factor 
params['hist_len']=40
params['state_thresh']=[0.12,0.2] #similarity of noisy state to ideal state
#if lower state_creation_threshold for Q[0] compared to Q[1], then Q[0] will have fewer states
#possibly multiply state_thresh by learn_rate? to change dynamically?
params['sigma']=0.25 #similarity of noisey state to ideal state,std used in Mahalanobis distance.
params['time_inc']=0.1 #increment time since reward by this much in no reward
params['moving_avg_window']=3  #This in units of trials, the actual window is this times the number of events per trial
params['decision_rule']=None #'combo', 'delta', 'sumQ2', None ## None means use direct negative of D1 rule
params['Q2other']=0.1
params['forgetting']=0
params['reward_cues']=None ##options: 'TSR', 'RewHx3', 'reward', None
params['distance']='Euclidean'
params['split']=True
params['events_per_trial']=3

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
act={'center':0,'left':1,'return':2,'right':3,'wander':4,'hold':5,'groom':6,'other':7} 

states={'loc':{'start':0,'Pport':2,'Lport':1,'other':4,'Rport':3},
        'tone':{'blip':0,'6kHz':6,'success':2,'error':-2,'10kHz':10}} #These values need units
params['state_units']={'loc':False,'tone':True}

#some convenient variables
start=(states['loc']['start'],states['tone']['blip']) #used many times
env_params={'start':start}
loc=states['loc'] #used only to define R and T
tone=states['tone'] #used only to define R and T
move=['left','right','wander','center']
stay=['groom','other','hold']# ['hold'] - if eliminate groom and other#

Racq={};Tacq={}  #dictionaries to improve readability/prevent mistakes
####value of T dict is the new state
#from start - best response is poke
Tacq[start]={a:[(start,1)] for a in act.values()} # stay at start
for a in ['left','right']:
    Tacq[start][act[a]]=[((loc['other'],tone['error']),1)] #error if go to left or right port from start box
Tacq[start][act['wander']]=[((loc['other'],tone['blip']),1)] #meandering, not yet at poke port
Tacq[start][act['center']]=[((loc['Pport'],tone['6kHz']),1)] #poke at start tone, go to poke port
#What happens if agent doesn't go to poke port immediately
Tacq[(loc['other'],tone['blip'])]={a:[((loc['other'],tone['blip']),1)] for a in act.values()} #default: remain in other unless
Tacq[(loc['other'],tone['blip'])][act['center']]=[((loc['Pport'],tone['6kHz']),1)] #go to center port, after wandering
Tacq[(loc['other'],tone['blip'])][act['return']]=[(start,1)] #return to start

#from poke port - correct response is 'left'
Tacq[(loc['Pport'],tone['6kHz'])]={a:[((loc['Pport'],tone['6kHz']),1)] for a in act.values()} #default - stay in poke port
for a in ['wander','return']: #incorrect movements
    Tacq[(loc['Pport'],tone['6kHz'])][act[a]]=[((loc['other'],tone['error']),1)]  #incorrect movements
Tacq[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['error']),1)]
Tacq[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),1)] #hear tone in poke port, go left, in left port/success

Tacq[(loc['other'],tone['error'])]={a:[((loc['other'],tone['error']),1)] for a in act.values()}#remain in other unless
Tacq[(loc['other'],tone['error'])][act['return']]=[(start,1)] #return to start

#from left port - best response is return
Tacq[(loc['Lport'],tone['success'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} #default, wandering around
for a in stay:
    Tacq[(loc['Lport'],tone['success'])][act[a]]=[((loc['Lport'],tone['error']),1)] #staying at Lport, but not continued success (reward)
Tacq[(loc['Lport'],tone['success'])][act['return']]=[(start,1)] #go back to start to begin again

#from right port - best response is return
Tacq[(loc['Rport'],tone['error'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} 
for a in  stay:
    Tacq[(loc['Rport'],tone['error'])][act[a]]=[((loc['Rport'],tone['error']),1)] #remain in Rport if no movement,
Tacq[(loc['Rport'],tone['error'])][act['return']]=[(start,1)] #return to start

Tacq[(loc['Lport'],tone['error'])] = {act[a]:[((loc['other'],tone['error']),1)] for a in move}
for a in  stay:
    Tacq[(loc['Lport'],tone['error'])][act[a]]=[((loc['Lport'],tone['error']),1)] #remain in Lport if no movement,
Tacq[(loc['Lport'],tone['error'])][act['return']]=[(start,1)] #return to start

#error tone is not associated with penalty except with incorrect response from Pport
for k in Tacq.keys(): #Tacq determines what states pairs need reward values
    Racq[k]={a:[(rwd['base'],1)] for a in act.values()} #default: cost of basic action
    Racq[k][act['hold']]=[(rwd['none'],1)] #not moving - nocost
#reward for correct response
Racq[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],0.9),(rwd['base'],0.1)]   #lick in left port - 90% reward
#Error if go anywhere but correct port after tone
for a in ['right','wander','return']:
    Racq[(loc['Pport'],tone['6kHz'])][act[a]]=[(rwd['error'],1)] 
#Error if go straight to left or right port from start box - same error for discrimination
for a in['right','left']:
    Racq[(loc['start'],tone['blip'])][act[a]]=[(rwd['error'],1)] 
############## End acquisition parameters ###################
### initialize extinction as same as acquistion
## But, no errors or rewards
import copy #needed to make copies of dict of dicts
#Liu J Neurosci 40-6409 - during extinction, syringe pump (success sound) still triggered
#THUS, transitions the same, but no reward (nor penalty?)
Rext=copy.deepcopy(Racq)
Text=copy.deepcopy(Tacq) 
for a in ['right','wander','return','left']:
    Rext[(loc['Pport'],tone['6kHz'])][act[a]]=[(rwd['base'],1)] #base cost 

###### initialize discrimination as same as acquistion
Rdis=copy.deepcopy(Racq)
Tdis=copy.deepcopy(Tacq)
### add in some states
#Change transitions from start so that 50% of time 6kHz and 50% 10kHz
Tdis[start][act['center']]=[((loc['Pport'],tone['6kHz']),0.5),((loc['Pport'],tone['10kHz']),0.5)] #after poking, each tone presented 50% of time
Tdis[(loc['other'],tone['blip'])][act['center']]=[((loc['Pport'],tone['6kHz']),0.5),((loc['Pport'],tone['10kHz']),0.5)] #go to poke port late
#add in transitions to (Pport,10kHz)
Tdis[(loc['Pport'],tone['10kHz'])]={a:[((loc['Pport'],tone['10kHz']),1)]  for a in act.values()} #default - stay in poke port
for a in ['wander','return']:
    Tdis[(loc['Pport'],tone['10kHz'])][act[a]]=[((loc['other'],tone['error']),1)]  #incorrect movements
Tdis[(loc['Pport'],tone['10kHz'])][act['right']]=[((loc['Rport'],tone['success']),1)] #hear tone in poke port, go left, in left port/success
Tdis[(loc['Pport'],tone['10kHz'])][act['left']]=[((loc['Lport'],tone['error']),1)] #hear tone in poke port, go left, in left port/success

Tdis[(loc['Rport'],tone['success'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} ##wandering around
for a in stay:
    Tdis[(loc['Rport'],tone['success'])][act[a]]=[((loc['Rport'],tone['error']),1)] #staying at Rport, but not continued success (reward)
Tdis[(loc['Rport'],tone['success'])][act['return']]=[(start,1)] #go back to start to begin again

#add in rewards to (Pport,10kHz)
Rdis[(loc['Pport'],tone['10kHz'])]={a:[(rwd['base'],1)] for a in act.values()}
Rdis[(loc['Pport'],tone['10kHz'])][act['right']]=[(rwd['reward'],0.9),(rwd['base'],0.1)]   #lick in left port - 90% reward
Rdis[(loc['Pport'],tone['10kHz'])][act['hold']]=[(rwd['none'],1)]
for a in ['left','wander','return']:
    Rdis[(loc['Pport'],tone['10kHz'])][act[a]]=[(rwd['error'],1)]
#add in rewards to (Rport,success)
Rdis[(loc['Rport'],tone['success'])]={a:[(rwd['base'],1)] for a in act.values()}
Rdis[(loc['Rport'],tone['success'])][act['hold']]=[(rwd['none'],1)]

############## End discrimination parameters ###################
#complex reversal - change which behavior rewarded for which tone
#i.e., now 6 Khz means go right for reward (not left)
# 10 Khz means go left for rewerd (not right)
Rrev=copy.deepcopy(Rdis)
Trev=copy.deepcopy(Tdis)
Trev[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),1)]
Trev[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['error']),1)] #hear tone in poke port, go left, in left port/success
Trev[(loc['Pport'],tone['10kHz'])][act['right']]=[((loc['Rport'],tone['error']),1)] #hear tone in poke port, go left, in left port/success
Trev[(loc['Pport'],tone['10kHz'])][act['left']]=[((loc['Rport'],tone['success']),1)] #hear tone in poke port, go left, in left port/success

Rrev[(loc['Pport'],tone['10kHz'])][act['left']]=[(rwd['reward'],0.9),(rwd['base'],0.1)]   #lick in left port - 90% reward
Rrev[(loc['Pport'],tone['10kHz'])][act['right']]=[(rwd['error'],1)]   #lick in left port - 90% reward
Rrev[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['error'],1)]   #lick in left port - 90% reward
Rrev[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],0.9),(rwd['base'],0.1)]   #lick in left port - 90% reward

################# Two arm bandit task of Josh Berke (Hamid et al. Nat Neuro V19)
Rbandit=copy.deepcopy(Racq)
Tbandit=copy.deepcopy(Tacq) 
prwdR=0.8; prwdL=0.5 #initial values.  These change with each block of trials
Tbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),prwdR),((loc['Rport'],tone['error']),1-prwdR)]
Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],prwdR),(rwd['base'],1-prwdR)]  
Tbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),prwdL),((loc['Lport'],tone['error']),1-prwdL)] #hear tone in poke port, go left, in left port/success
Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],prwdL),(rwd['base'],1-prwdL)]

Tbandit[(loc['Rport'],tone['success'])]={act[a]:[((loc['other'],tone['error']),1)] for a in move} #default, wandering around
for a in stay:
    Tbandit[(loc['Rport'],tone['success'])][act[a]]=[((loc['Rport'],tone['error']),1)] #staying at Lport, but not continued success (reward)
Tbandit[(loc['Rport'],tone['success'])][act['return']]=[(start,1)] #go back to start to begin again
Rbandit[(loc['Rport'],tone['success'])]={a:[(rwd['base'],1)] for a in act.values()} #default: cost of basic action
Rbandit[(loc['Rport'],tone['success'])][act['hold']]=[(rwd['none'],1)]

if __name__== '__main__':
    ######## Make sure all needed transitions have been created
    validate_T(Tacq,msg='validate Tacq')
    validate_R(Tacq,Racq,msg='validate Racq')
    validate_T(Tdis,msg='validate discrim T')
    validate_R(Tdis,Rdis,msg='validate discrim R')
    validate_T(Trev,msg='validate reversal T')
    validate_R(Trev,Rrev,msg='validate reversal R')
    validate_T(Tbandit,msg='validate bandit T')
    validate_R(Tbandit,Rbandit,msg='validate bandit R')
