====
TD2Q
====
Python code for new reinforcement learning model
Q learning model with Q matrices representing dSPN and iSPN

**A. Several reinforcement learning tasks have been implemented**
- Discrim2stateProbT_twoQtwoS.py:

  * Variations on Discrimination and extinction.	
  * parameters, including reward and transition matrix, in DiscriminationTaskparam2.py.
  * 5 different learning phases have been implemented:
    
    * acquisition of 1 choice task (tone A, left)
    * extinction of task (in different context)
    * renewal - retesting extinction in the same context
    * discrimination - adding in 2nd choice (toneB, right)
    * reversal - switching the direction that is rewarded
         
  * Can also test
    * savings after extinction
    * acquisition in new context
    * AAB, ABA and ABB extinction and renewal, where A and B are contexts
		
- BanditTask.py
	* parameters, including reward and transition matrix, in BanditTaskparam.py
	* The 2-arm bandit task, from (start location, tone blip) the agent must go to the center poke port. 
	* At the poke port, the agent hears a single tone (go cue) which contains no information about which port is rewarded. 
	* To receive a reward, the agent has to select either left port or right port.  
	* Both left and right choices are rewarded with probabilities assigned independently, and which change periodically.  
	
- SequenceTask.py
	* parameters, including reward and transition matrix, in SequenceTaskparam.py
	* the task is that reported in Geddes et al. Cell 2018
	* the agent must press the left lever twice, and then the right lever twice to obtain a reward. 
	* There are no external cues to indicate when the left lever or right lever need to be pressed.

**B. Other files**
- agent_twoQtwoSsplit.py
	* agent which has one or two Q matrices.
	* the agents states can include a context cue - one that does not influence the reward or transitions
	* agent states (and Q matrix rows) are added as the agent encounters new states
	
- completeT_env.py
	environment in which every state is explicated
	
- sequence_env.py
	environment used for large numbers of states, in which one type of state (e.g. press history) is independent of another type of state (e.g. location).  I.e., an agents action alters either press history or location, but not both.
	This simplifies specification of the transition matrix
	
- RL_class.py
	base classes for the environment and the agent
  
- RL_utils.py
  some additional functions used for graphs and accumulate output data
	
- Qlearn_multifile_param_anal.py
	To analyze a large set of parameter sweep simulations

- TD2Q_manuscript_graphs.py and TD2Q_Qhx_graphs.py
	Used to create publication quality figures.

**C. Parameters**
- params['numQ']=1 #number of Q matrices.  numQ=2 is improves the 2-arm bandit task and sequence task.  No effect on discirmination/extinction
- params['alpha']=[0.3,0.06]  # learning rate for Q1 and (optionally Q2) matrices.  Task dependent
- params['beta']=0.9  # maximum value of inverse temperature, controls exploration-exploitation
- params['beta_min']=0.5 # minimum value of inverse temperature, controls exploration
- params['gamma']=0.9  #discount factor
- params['hist_len']=40 #update covariance matrix and ideal states of agents as average of this many events
- params['state_thresh']=[0.12,0.2] #threshold distance of input state to ideal state. Task and distance measure dependent 
- params['sigma']=0.25 #std used in Mahalanobis distance.
- params['moving_avg_window']=5 #This in units of trials, the actual window is this times the number of events per trial.  It is used to calculate reward probability
- params['decision_rule']
  * None: choose action based on Q1 and Q2, then resolve difference
  * 'delta': choose action based on difference between Q1 and Q2 matrix
- params['Q2other']=0.1 #fractional learning rate (multiplied by alpha) for Q2 values for NON-selected actions
- params['distance']='Euclidean' #determine best matching state based on Euclidean distance, alternative: "Gaussian": mahalanobis distance
- params['split']=True #initialize new row of Q matrix as values of best matching state or as zeros.  



