# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:44:10 2020

@author: kblackw1
Q learning agent with two sets of states, Q matrices
Q[0]: equivalent to the traditional Q matrix, representing D1-SPNs
Q[1]: a second Q matrix representing D2-SPNs
The number of states for each matrix can be different, allowing 
D2s to be specialists
D1s to generalize
action determined by comparing probabilities associated with selected action
Currently, learning rule is same for both Q, but Q[1] learns more slowly
"""
import numpy as np
from scipy import linalg
from scipy import spatial
from RL_class import Agent
from scipy.ndimage.filters import uniform_filter1d
import copy

def flatten(isiarray):
    return [item for sublist in isiarray for item in sublist]

class QL(Agent):
    """Class for a Q-learning agent"""
    
    def __init__(self, actions, params, oldQ={}):
        self.numQ=params['numQ'] #Q matrix for D1 and D2
        naction=len(actions)
        self.actions=actions
        #self.n_cues=len(cues)
        #each ideal_state = env state and zeros for other cues, 
        if len(oldQ):
            print('********** starting from previous learning phase:',oldQ['name'])
            if 'rwd_prob' in oldQ.keys():
                print('rwd_prob',round(oldQ['rwd_prob'],4))#,'learn_weight',round(oldQ['learn_weight'],4))
            self.ideal_states=copy.deepcopy(oldQ['ideal_states'])
            print('init ideal states',[len(self.ideal_states[kk]) for kk in range(self.numQ)]
                  ,'::', [list(np.round(st,2))  for kk in range(self.numQ) for st in self.ideal_states[kk].values()])
        else:
            #start with only 1 ideal_state - the starting point
            print('*** NO OLD states, initalize empty ideal_states')
            self.ideal_states={kk:{} for kk in range(self.numQ)}
        nstate={kk:len(self.ideal_states[kk]) for kk in range(self.numQ)}
        super().__init__(nstate, naction)
        ############# default parameters
        self.alpha = copy.copy(params['alpha'])  # learning rate, no forgetting if alpha[1]=0, or using old rule 
        self.beta_max = copy.copy(params['beta'])  # inverse temperature
        if 'beta_min' in params.keys():
            self.beta_min = copy.copy(params['beta_min'])  #make this parameter.  If the same as params['beta'], beta doesn't vary
        else:
            self.beta_min=copy.copy(params['beta'])
        self.gamma = copy.copy(params['gamma'])  # discount factor - future anticipated rewards worth 10% less per time unit
        self.state_thresh=copy.copy(params['state_thresh'])
        self.time_increment=copy.copy(params['time_inc'])
        self.window=params['moving_avg_window']*params['events_per_trial']
        self.sigma=copy.copy(params['sigma'])
        self.wt_noise=copy.copy(params['wt_noise'])
        self.wt_learning=copy.copy(params['wt_learning'])
        self.learn_weight=1 #initialize learning rate to 1
        self.prior_rwd_prob=None
        self.min_learn_weight=0.1
        self.events_per_trial=copy.copy(params['events_per_trial'])
        self.distance=params['distance']
        if 'Q2other' in params.keys():
            self.wt_Q2other=copy.copy(params['Q2other'])
        else:
            self.wt_Q2other=1
        if 'decision_rule' in params.keys():
            self.decision_rule=copy.copy(params['decision_rule'])
        else:
            self.decision_rule=None
        if 'split' in params.keys():
            self.split=copy.copy(params['split'])
        else:
            self.split=True
        if 'forgetting' in params.keys():
            self.forgetting=copy.copy(params['forgetting'])
        else:
            self.forgetting=0
        if 'reward_cues' in params.keys():
            self.reward_cues=copy.copy(params['reward_cues'])
        else:
            self.rewared_cues=None
        if 'state_units' in params:
            self.state_units=[x for x in params['state_units'].values()]
        self.winner_count={k:0 for k in range(self.numQ)}
        #################### allocate Q table
        self.Q={}
        if len(oldQ):
            print('oldQ shapes',[np.shape(oldQ['Q'][kk]) for kk in range(self.numQ)])
            for kk in range(self.numQ):
                self.Q[kk]=copy.deepcopy(oldQ['Q'][kk])
            #if 'learn_weight' in oldQ.keys():
            #    self.learn_weight=copy.copy(oldQ['learn_weight'])
            if 'rwd_prob' in oldQ.keys():
                self.prior_rwd_prob=copy.copy(oldQ['rwd_prob'])                
        ############# initalize memory of prior "non-ideal" states, covariance matrix
        self.history_length=params['hist_len'] #maximum number of items in state_history
        self.state_history={kk:{i:[] for i in range(self.Ns[kk])} for kk in range(self.numQ)}
        #### the following are unused in by the Euclidean distance agent
        self.cov={kk: {} for kk in range(self.numQ)}
        self.inv_cov_mat={kk: {} for kk in range(self.numQ)}
        self.det_cov={kk: {} for kk in range(self.numQ)}
        print('&&&&&&& END INIT &&&&&&&, alpha=', self.alpha, 'state_thresh=',self.state_thresh)

    def decode(self,nn,state):
        return list(self.ideal_states[nn].values()).index(state) 
            
    def update_cov_inv(self,cov):
        inv_cov_mat=linalg.inv(cov)
        det_cov=linalg.det(cov)
        return inv_cov_mat,det_cov

    def init_cov(self,state_num,nn):
        self.cov[nn][state_num]=np.diag([self.sigma for s in range(self.nstate_types)])
        self.inv_cov_mat[nn][state_num],self.det_cov[nn][state_num]=self.update_cov_inv(self.cov[nn][state_num])
        
    def update_cov(self,state_num,nn):
        self.cov[nn][state_num]=np.cov(self.state_history[nn][state_num],rowvar=False)
        self.inv_cov_mat[nn][state_num],self.det_cov[nn][state_num]=self.update_cov_inv(self.cov[nn][state_num])
        #print('update_cov for Q',nn,',state',state_num,'cov',self.cov[nn][state_num])
        
    def extendQ(self,nn,numrows):
        self.Q[nn]=np.vstack((self.Q[nn],np.zeros((numrows,self.Na))))
        #print('Q extended. new shape',np.shape(self.Q[nn]))

    def splitQ(self,nn,Qrow):
        self.Q[nn]=np.vstack((self.Q[nn],self.Q[nn][Qrow]))
        #print('Q split for Q',nn,' new row',self.Q[nn][Qrow])
        
    def update_Qhx(self):
        for k in self.Qhx.keys():
            if np.shape(self.Qhx[k])[1]<np.shape(self.Q[k])[0]:
                #first, axtend rows of Qhx so it matches Q
                dim0=np.shape(self.Qhx[k])[0]
                dim1=np.shape(self.Qhx[k])[1]
                dim2=np.shape(self.Qhx[k])[2]
                delta=np.shape(self.Q[k])[0]-dim1
                self.Qhx[k]=np.concatenate((self.Qhx[k],np.zeros((dim0,delta,dim2))),axis=1)                
            self.Qhx[k]=np.vstack((self.Qhx[k],self.Q[k][None]))

    def start(self,state,cues=[],state_num=0):
        """first action, without reward feedback"""
        #Note, state_num input must be within 0 and len(self.Q)
        # Boltzmann action selection - randomly choose one of the actions possible from the specified state
        #set up a few more variables
        if self.reward_cues is not None:
            if self.reward_cues.startswith('RewHx'):
                self.numbins=int(self.reward_cues[-1]) #no more than 9 bins
                self.bin_edges=np.array([i*1/self.numbins for i in range(self.numbins+1)])
            allcues=self.add_reward_cue(cues,0)
        else:
            allcues=cues
        self.state_num=[state_num for kk in range(self.numQ)]
        #### initalize ideal_states (to 1st state) and Q if oldQ not provided
        if not len(self.Q):
            for kk in range(self.numQ):
                self.ideal_states[kk]={state_num:[si for si in state]+allcues }
                self.Ns[kk]+=1
            self.Q={kk:np.zeros((self.Ns[kk], self.Na)) for kk in range(self.numQ)}
        #nstate_types is len of list used to represent states
        self.nstate_types=len(self.ideal_states[0][state_num]) 
        #initalize weighting on noise as 1, update according to range of states values
        self.wt={kk:[1 for ii in range(self.nstate_types)] for kk in range(self.numQ)}
        self.std={kk:[1 for ii in range(self.nstate_types)] for kk in range(self.numQ)}
        if not np.any(list(self.state_history.values())):
            self.state_history={kk:{self.state_num[kk]:[self.ideal_states[kk][self.state_num[kk]]]} for kk in range(self.numQ)}
        #
        if self.distance=='Gaussian':
            for kk in range(self.numQ): 
                for i in range(self.Ns[kk]):
                    self.init_cov(i,kk)
        hist_items=['LR','TSR','rwd_hist','rwd_prob','delta','beta']
        self.learn_hist={k:[] for k in hist_items}
        if self.prior_rwd_prob:
            self.beta=self.prior_rwd_prob*(self.beta_max-self.beta_min)+self.beta_min
        else:
            self.beta=self.beta_min
        self.time_since_reward=0
        #initalize dictionary to hold history of Q values, stack to initialize 3D array
        self.Qhx={k: np.vstack((self.Q[k][None],self.Q[k][None])) for k in self.Q.keys()} 
        #start by selecting state and action from Q[0]
        #Note, state is list - one state for each Q
        self.state=[self.ideal_states[kk][self.state_num[kk]] for kk in range(self.numQ)]
        self.action,actp = self.boltzmann( self.Q[0][self.state_num[0],:])
        print('>>>>> agent start: state=',self.state,self.state_num,', action=',self.action)
        # remember the state
        return self.action
                
    def add_reward_cue(self,cues,reward):
        if self.reward_cues.startswith('RewHx'):
            #newcues=cues+[self.bin_edges[np.min(np.where(self.rwd_hist<=self.bin_edges))]/self.numbins/2]
            #may not need to quantize.
            newcues=cues+[self.rwd_hist]
        elif self.reward_cues=='reward':
            newcues=cues+[np.heaviside(reward,0)]
        elif self.reward_cues=='TSR':
            newcues=cues+[self.time_since_reward] #specifying 'TSR' means use time_since_reward
        else:
            print('agent, add_reward_cue: unknown reward type')
            exit()
        #if reward>5:
        #    print('>>>>>>>>>>>>> add_reward_cue',reward, cues, newcues)
        return newcues
        
    def limit_list(self,item, old_list):
        old_list.insert(0, item)    
        return old_list[:self.history_length]

    def mahalanobis(self,noisestate,nn,Acues,prn_info):
        dist=np.zeros(len(self.ideal_states[nn]))
        prob=np.zeros(len(self.ideal_states[nn]))
        nc=len(Acues)/2
        for index,ideal in self.ideal_states[nn].items(): #calc distance to each possible state
            dist[index]=spatial.distance.mahalanobis(noisestate,ideal,self.inv_cov_mat[nn][index])
            #could try spatial.distance.euclidean(noisestate,ideal) - then skip the prob step
            prob[index] = np.exp(-0.5 * dist[index]) / (2*np.pi)**nc /np.sqrt(self.det_cov[nn][index])
            if prob[index]>1:
                if prn_info:
                    print('>>> PROBLEM, prob>1 for ',index,',p:',prob[index],'dist:',dist[index],'s:',self.det_cov[nn][index])
                prob[index]=1
        #state threshold (for Q0 only) is optionally weighted by learn_weight - same as weight for alpha[0]
        #if prob high enough, then good match to existing state.  Update that
        best_match=np.argmax(prob)
        if np.max(prob)>self.state_thresh[nn] or self.alpha[nn]==0: #no state splitting if learning set to 0
            newstate_num=np.argmax(prob)
            if len(self.state_history[nn][newstate_num])==self.history_length:
                self.update_cov(newstate_num,nn)
                #update ideal_state as mean of history, if history long enough. 
                self.ideal_states[nn][newstate_num]=list(np.mean(self.state_history[nn][newstate_num],axis=0))
        else:
            newstate_num=self.Ns[nn]
            self.init_cov(newstate_num,nn)
            if prn_info:
                print('dist,prob',dist,prob,'cov',self.det_cov[nn][np.argmax(prob)],self.cov[nn][np.argmax(prob)])
                print('@@@ add new state for Q',nn,'st=',[round(ns,2) for ns in noisestate], 'num',newstate_num,'maxprob=',round(np.max(prob),5))
        return newstate_num

    def Euclid(self,noisestate,nn,prn_info):
        dist=np.zeros(len(self.ideal_states[nn]))
        for index,ideal in self.ideal_states[nn].items(): #calc distance to each possible state
            dist[index]=spatial.distance.euclidean(noisestate,ideal,self.std[nn])
        #state threshold (for Q0 only) is optionally weighted by learn_weight - same as weight for alpha[0]
        #if prob high enough, then good match to existing state.  Update that state
        best_match=np.argmin(dist)
        if np.min(dist)<self.state_thresh[nn] or self.alpha[nn]==0:
            newstate_num=np.argmin(dist)
            #update ideal_state as mean of history, if history long enough. 
            if len(self.state_history[nn][newstate_num])==self.history_length:
                self.ideal_states[nn][newstate_num]=list(np.mean(self.state_history[nn][newstate_num],axis=0))
            all_state_values=[idst for st_hx in self.state_history[nn].values() for idst in st_hx]
            if len(all_state_values)>self.window: 
                for ii in range(self.nstate_types):
                    if self.state_units[ii]:
                        statevals=[idst[ii] for idst in all_state_values]
                        self.std[nn][ii]=1/np.std(statevals) #update std only if state uses units 
        else:
            newstate_num=self.Ns[nn]
            if prn_info:
                print('@@@ add new state for Q',nn,'st=',[round(ns,2) for ns in noisestate], 'num',newstate_num,'\ndist',dist,'argmax',np.argmin(dist))
        return newstate_num, best_match
    
    def select_agent_state(self,Acues,noise,nn,prn_info,ncues):
        if prn_info:
            print('###### BEG SELECT: Q',nn)
        #noise (agent's perception of noise) is optionally weighted by learn_weight - same as weight for alpha[0]
        #lower noise as animal learns to distinguish cues
        if self.wt_noise:
            noisestate=Acues+noise*self.learn_weight*np.random.randn(len(Acues))*np.array(self.wt[nn])
        else:
            noisestate=Acues+noise*np.random.randn(len(Acues))
        if self.distance=='Euclidean':
            newstate_num,best_match=self.Euclid(noisestate,nn,prn_info)
        elif self.distance=='Gaussian':
            newstate_num,best_match=self.Mahalanobis(noisestate,nn,Acues,prn_info)
        if newstate_num==self.Ns[nn]:
            #Above might show need for 2Q with discrim if it was only reset with new environmental cues (not new tasks)
            self.Ns[nn]+=1 #increment number of states
            self.state_history[nn][newstate_num]=[]
            self.ideal_states[nn][newstate_num]=list(noisestate) #add new state 
            #add row to Q matrix.  
            if prn_info:
                print('>>>>> new state for Q',nn, 'num',newstate_num,'ideal',self.ideal_states[0])
            if self.split==True: #add row to Q matrix.  
                #Initialize it the same as best matching state
                self.splitQ(nn,best_match)
            else:
                #add row of zeros to Q matrix.  Ideally, add in parameter that determines whether to initialize to 0 or to existing state
                self.extendQ(nn,1)
            #update weights on the noise - according to range of state values (differences between adjacent values)
            for ii in range(self.nstate_types):
                statevals=[np.round(idst[ii],1) for idst in self.ideal_states[nn].values()]
                self.wt[nn][ii]=(np.max(statevals)-np.min(statevals)+1)/len(np.unique(statevals))
        self.state_history[nn][newstate_num]=self.limit_list(list(noisestate),self.state_history[nn][newstate_num]) #add noisy state to memory
        #update ideal_state as mean of history, if history is long enough
        if len(self.state_history[nn][newstate_num])==self.history_length:
            self.ideal_states[nn][newstate_num]=list(np.mean(self.state_history[nn][newstate_num],axis=0))
        newstate=[int(np.round(ns)) for ns in self.ideal_states[nn][newstate_num]]
        if prn_info:
            print('###### END SELECT: Q',nn,'Acues,teststate, new state:',Acues,[round(ns,2) for ns in noisestate],newstate,', time since reward',self.time_since_reward)
        return newstate,newstate_num
    
    def boltzmann(self, qval): #q must be an array or list of actions that can be taken
        """Boltzmann selection"""
        pa = np.exp( self.beta*qval)   # unnormalized probability
        pa = pa/sum(pa)    # normalize
        act=np.random.choice(self.Na, p=pa)
        actp=pa[act]
        return  act,actp#choose action randomly, with prob distr given by pa
    
    def boltzman2Q(self,q1,q2):
        #(soft) maximize over action: Q1(action) - Q2 (action)
        if self.decision_rule=='delta' :
            deltaQ=q1-q2
        elif self.decision_rule=='sumQ2':
            deltaQ=q1-q2+np.sum(q2) #add Q2 val of all OTHER actions
        elif self.decision_rule=='combo':
            # Q1(action) - Q2 (action) + Q2[other actions)]
            deltaQ=q1-2*q2+np.sum(q2) 
        pa=np.exp( self.beta*deltaQ)
        act=np.random.choice(self.Na, p=pa/sum(pa))
        return act
    
    def moving_average(self,x, w):
        return uniform_filter1d(x, w,origin=int(w/2)) #follows mean great, origin makes it causal
    
    def D1_learn(self,reward,q,last_state_num,prn_info=False):      
        delta = reward + self.gamma*max(self.Q[q][self.state_num[q],:]) - self.Q[q][last_state_num[q],self.action]
        #if reward>0 or delta<5:
        #    print('delta',round(delta,2),'rwd',round(reward,2),'prob',self.rwd_prob,'act',self.action,'new_state',self.state_num[q],'gamma*max',self.Q[q][self.state_num[q]],'-this Q',self.Q[0][last_state_num[0],self.action])
        if prn_info:
            print('>>>>>>>> before learn, Q:',q,'=',self.Q[q][last_state_num[q]],'act=',self.action)
        if self.wt_learning:
            self.Q[q][last_state_num[q],self.action] += self.learn_weight*self.alpha[q]*delta #if you take best action but no reward, decrease value
        else:
            #reward<= 0 means no Da, #rule for D2 Q matrix if decision_rule = 'none'
            self.Q[q][last_state_num[q],self.action] += self.alpha[q]*delta #if you take best action but no reward, decrease value
        if prn_info:
            print('>>>>>>>> after learn, Q:',q,'=',self.Q[q][last_state_num[q]],'delta',delta)

        #if delta<0, then Q decreases, either if reward<0 or lower future rewards in current state
        # ABOVE: Q[laststate,action]=Q[laststate,action]+alpha*(reward-Q[laststate,action])+alpha*gamma*max(Q[newstate]) 
        #         =  alpha*reward - (1-alpha)*Q[laststate,action] + alpha*gamma*max(Q[newstate])
        #         =  alpha*(reward +gamma*max(Q[newstate])) - (1-alpha)*Q[laststate,action]
        #Doya (Funamizu): if reward=0, change alpha*reward to -alpha*k2.  
        #If reward=+k1 and "non-reward" = -k2, then above learning rule impelments Doya & Funamizu rule
        #Except, need to implement the forgetting part: decrement Q values for actions not taken
		#maybe should only do this on reward trials?  maybe this only works in trial based algorithms?
        for act in self.actions.values(): 
            if act != self.action:
                self.Q[q][last_state_num[q],act]= (1-self.forgetting*self.alpha[q])*self.Q[q][last_state_num[q],act]
        return delta
    
    def D2_learn(self,delta,q,last_state_num,prn_info=False,block_DA=False):
        ### DIFFERENT RULE for Q2 ####
        '''
        #1. Invert the reward or the delta (implemented)
        #2. Da dips have smaller dynamic range than increases
        dip=reward**(1/3) ##3d root compresses the range
        delta2 = dip + (self.gamma*max(self.Q[q][self.state_num[q],:]) - self.Q[q][last_state_num,self.action])
        #3. compress the RPE instead of the reward?
        delta2=delta**(1/3)
        '''
        #Change the Q value according to delta (calculated in D1_learn), but DECREASE Q2 when delta is positive
        #i.e., LTD if delta positive, LTP if delta negative (DA dip)
        #I.e., a Da dip increases Qvalue.  
        if self.wt_learning:
            alpha=self.learn_weight*self.alpha[q]
        else:
            alpha=self.alpha[q]
        #use above alpha for both action taken and other actions
        if delta>0 or (delta<0 and not block_DA): #possibly change to if not block_DA
            self.Q[q][last_state_num[q],self.action] -= alpha*delta #if you take best action but no reward, decrease value
		#. change values for actions not taken - i.e., increase value of other actions
        #  If wt_Q2other=0, then this part does not happen
        #if (block_DA=='no_dip' and delta>0) or (block_DA=='AIP' and delta<0) or block_DA==False: #both types of heterosynaptic depression
        if delta<0 and block_DA != 'no_dip': # heterosynaptic depression only.  Justified by 2ag diffusion
            for act in self.actions.values(): 
                if act != self.action:
                    self.Q[q][last_state_num[q],act]+= self.wt_Q2other*alpha*delta
                #self.Q[q][last_state_num[q],act]= (1-self.forgetting*self.alpha[q])*self.Q[q][last_state_num[q],act]
        #if delta< 0 - as in low dopamine condition of Gurney, then LTP occurs for 'other' actions
        ## alternative - to maintain sumQ constant, make self.wt_Q2other = self.alpha/num_actions
		## ther actions represent state input to neuron with no action input - reduced (weaker) Ctx inputs.
		##       Consider LTD regardless of Da dip (delta)    
        return

    def choose_act(self,D1rule=False,prn_info=False):
        self.action,act0prob = self.boltzmann( self.Q[0][self.state_num[0],:])
        winner='Q0'
        self.winner_count[0]+=1
        if self.numQ>1:
            if D1rule:
                self.action2,act1prob = self.boltzmann( self.Q[1][self.state_num[1],:])
            else:
                self.action2,act1prob = self.boltzmann( -self.Q[1][self.state_num[1],:])               
            if self.action != self.action2:
                if prn_info:
                    print('$$$$$$$ step, Q0 action,prob', self.action,round(act0prob,5), 'Q1 action,prob',self.action2,round(act1prob,5))
                if act1prob>act0prob:
                    #print('$$$$$$$ step, Q1 action', self.action,round(act0prob,5), 'Q2 action',self.action2,round(act1prob,5))
                    self.action=self.action2 
                    winner='Q1'
                    self.winner_count[1]+=1
                    self.winner_count[0]-=1
            else:
                winner='Q0act=Q1act'
        return winner

    def step(self, reward,state,noise,cues=[],prn_info=False,block_DA=False):
        last_state=self.state
        last_state_num=self.state_num
        if reward>0: #track time since reward - use as cue or learning rate
            self.time_since_reward=0
        else:
            self.time_since_reward+=self.time_increment
        #vector of reward history - used to estimate reward probability
        self.learn_hist['rwd_hist'].append(np.heaviside(reward,0))
        #vector of time since reward
        self.learn_hist['TSR'].append(self.time_since_reward)
        if len(self.learn_hist['rwd_prob'])>self.window: #test moving average if enough samples and agent has finally received a reward
            #learning rate depends on difference between prior reward prob and current reward prob
            #learning rate should not be lower than arbitrary minimum, no higher than 1.0
            self.rwd_prob=self.events_per_trial*self.moving_average(self.learn_hist['rwd_hist'],self.window)[-1]
            if self.prior_rwd_prob:
                prob_diff=max(self.min_learn_weight,np.abs(self.prior_rwd_prob-self.rwd_prob))
                self.learn_weight=min(1,prob_diff)
            #update exploration parameter, beta based on rwd_prob
            #if rwd_prob is low, make beta low, if rwd_prob is high, make beta high
            self.beta=self.rwd_prob*(self.beta_max-self.beta_min)+self.beta_min
        else:
            self.rwd_prob=0   
            #do not change self.beta
        self.learn_hist['LR'].append(self.learn_weight)
        self.learn_hist['rwd_prob'].append(self.rwd_prob)
        self.learn_hist['beta'].append(self.beta)
        #
        if self.reward_cues is not None:
            allcues=self.add_reward_cue(cues,reward)
        else:
            allcues=cues
        Acues=list(state)+allcues
        #if state[0:2]==(2,2):
        #    print('reward state', state,last_state[0],Acues,'reward',reward)
        ####################### determine new agent state
        newstate=[]
        newstate_num=[]
        for nn in range(self.numQ):
            ns,ns_num=self.select_agent_state(Acues,noise,nn,prn_info,len(cues))
            newstate.append(ns)
            newstate_num.append(ns_num)
        self.state_num=newstate_num
        #
        ############################# based on reward, update Q matrix ###############################
        delta=self.D1_learn(reward,0,last_state_num,prn_info=False) #0 means update Q[0]
        self.learn_hist['delta'].append(delta)
        if self.numQ>1:
            #self.D1_learn(reward,1,last_state_num,prn_info=False) #use same learning rule for Q0 and Q1 (D2 and D1)
            #delta is the RPE = Da signal.  Cannot estimate it from Q2
            #if block_DA: delta=0
            self.D2_learn(delta,1,last_state_num,prn_info=False,block_DA=block_DA) #1 means update Q[1]
        self.update_Qhx()
        ####################### given the state, what is the best action?
        winner=self.choose_act() #use this rule if Q2 updated using D1_learn, or numQ==1
        #Three other possible action selection rules
        if self.decision_rule!=None and self.numQ==2:            
            self.action=self.boltzman2Q( self.Q[0][self.state_num[0],:],self.Q[1][self.state_num[1],:]) #this will replace action determined using choose_act
        if prn_info:
            act_words=list(self.actions.keys())[list(self.actions.values()).index(self.action)]
            print('  END STEP: winner:',winner,', action',self.action,'=',act_words,', reward',reward)
        # remember the state
        self.state = newstate
        return self.action
    
    def visual(self,Q,title='',labels=None,state_subset=None):
        """Visualize the Q table by bar plot"""
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure()
        plt.suptitle(title)
        colors=plt.get_cmap('inferno') #plasma, viridis, inferno or magma
        color_increment=int((len(colors.colors)-40)/(len(self.actions)-1)) #40 to avoid to light colors
        newQ=[];statenums=[];xlabels=[]
        for s,row in enumerate(Q):
            if np.any(row):
                newQ.append(row)
                statenums.append(s)
                if labels is not None:
                    xlabels.append(labels[s])
        if len(xlabels) and state_subset is not None:
            Qsubset=[]
            keep_state=[(i,lbl)  for i,lbl in enumerate(xlabels) for ss in state_subset if ss in lbl]
            for (i,lbl) in keep_state:
                Qsubset.append(newQ[i])
            plotQ=np.array(Qsubset)
            xlabels=[ks[1] for ks in keep_state]
            statenums=[ks[0] for ks in keep_state]
        else:
            plotQ=np.array(newQ)
        if len(plotQ):
            w = 1./(self.Na+0.5) # bar width
            for a in range(self.Na):
                cnum=a*color_increment
                plt.bar(np.arange(len(statenums))+(a-(self.Na-1)/2)*w, plotQ[:,a], w,color=colors.colors[cnum])  
            plt.xticks(range(len(plotQ)),statenums)
            plt.xlabel("state"); plt.ylabel("Q")
            plt.legend(list(self.actions.keys()))
            #make vertical grid - between groups of bars
            for ll in range(len(plotQ)-1):
                plt.vlines(ll+0.5,np.min(plotQ),np.max(plotQ),'grey',linestyles='dashed')
            if labels is not None:
                for ii,l in enumerate(xlabels):
                    plt.annotate(' '.join(l), (ii-0.5, np.min(plotQ)-0.05*(ii%2)*(np.max(plotQ)-np.min(plotQ))))
            plt.show()
    
    def plot_learn_history(self,title=''):
        import matplotlib.pyplot as plt
        plt.ion()
        fig,ax=plt.subplots(len(self.learn_hist),1,sharex=True)
        fig.suptitle(title)
        for i,k in enumerate(self.learn_hist.keys()):
            ax[i].plot(self.learn_hist[k],'k',label=k)
            mean_hist=self.moving_average(self.learn_hist[k],self.window)
            ax[i].plot(mean_hist,label=k+' avg')
            ax[i].legend()
        plt.show()
    
    def plot_Qdynamics(self,actions,plot_type='',title=''):
        import matplotlib.pyplot as plt
        from numpy import matlib #part of numpy, but not imported by default
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        for k in self.Qhx.keys():
            tr=matlib.repmat(np.arange(np.shape(self.Qhx[k])[0]),np.shape(self.Qhx[k])[1],1).T
            st=matlib.repmat(np.arange(np.shape(self.Qhx[k])[1]),np.shape(self.Qhx[k])[0],1)
            fig = plt.figure()
            fig.suptitle(title+' Q'+str(k))
            for inum,act in enumerate(actions):
                actnum=self.actions[act]
                ax = fig.add_subplot(1, len(actions)+1, inum+1, projection='3d')
                Z=self.Qhx[k][:,:,actnum]
                ax = fig.gca(projection='3d')
                if plot_type=='wire':
                    ax.plot_wireframe(tr, st, Z, rcount=np.shape(Z)[0],ccount=np.shape(Z)[1],alpha=0.5)
                else:
                    norm = plt.Normalize(self.Qhx[k].min(), self.Qhx[k].max())
                    colors=cm.plasma(norm(Z[k]))
                    #rcount, ccount, _ = colors.shape
                    ax.plot_surface(tr, st, Z, rstride=1,cstride=1,cmap=cm.seismic,shade=False)
                ax.view_init(elev=18., azim=-40)
                ax.set_ylabel('state',fontsize=12)
                ax.set_xlabel('Event',fontsize=12)
                ax.set_zlabel('Q[state,'+act+']',fontsize=12)
        plt.show()
   
