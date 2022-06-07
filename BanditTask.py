# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:14:25 2021

@author: kblackw1
"""
import numpy as np
import completeT_env as tone_discrim
import agent_twoQtwoSsplit as QL
from RL_TD2Q import RL
from BanditTaskParam import params, env_params, states,act, Rbandit, Tbandit
import RL_utils as rlu
from BanditTaskParam import loc, tone, rwd
from TD2Q_Qhx_graphs import Qhx_multiphase

def plot_prob_traject(data,params):
    p_choose_L={}
    for k in data.keys():
        p_choose_L[k]=data[k][(('Pport', '6kHz'), 'left')]['mean']/(data[k][(('Pport', '6kHz'), 'left')]['mean']+data[k][(('Pport', '6kHz'), 'right')]['mean'])
    p_choose_k_sorted=dict(sorted(p_choose_L.items(),key=lambda item: float(item[0].split(':')[0])-float(item[0].split(':')[1]),reverse=True))
    
    from matplotlib import pyplot as plt   
    plt.figure()
    plt.suptitle(traject_title+' wt learn:'+str(params['wt_learning']))
    
    colors=plt.get_cmap('inferno') #plasma, viridis, inferno or magma possible
    color_increment=int((len(colors.colors)-40)/(len(data.keys())-1)) #40 to avoid to light colors
    for k,key in enumerate(p_choose_k_sorted.keys()):
        cnum=k*color_increment
        plt.plot(p_choose_k_sorted[key],color=colors.colors[cnum],label=key)
    plt.legend()
    plt.ylabel('prob(choose L)')
    plt.xlabel('block')

def count_shift_stay(rwd_indices,same,different,counts,r):
    for phase in rwd_indices.keys(): 
        for index in rwd_indices[phase]:
            #count how many times next trial was left versus right
            if index+1 in same[phase]:
                counts[phase]['stay'][r]+=1
            elif index+1 in different[phase]:
                counts[phase]['shift'][r]+=1
    return counts
def shift_stay_list(acq,all_counts):
    responses={};total={}
    actions=['left','right']
    rwd={act:{} for act in actions}
    no_rwd={act:{} for act in actions}
    for phase,rl in acq.items():
        res=rl.results
        responses[phase]=[list(res['state'][i])+[(res['action'][i])]+[(res['reward'][i+1])] for i in range(len(res['reward'])-1) if res['state'][i]==(loc['Pport'],tone['6kHz'])]    
        for action in actions:
            rwd[action][phase]=[i for i,lst in enumerate(responses[phase]) if lst==[loc['Pport'],tone['6kHz'],act[action],10]]
            no_rwd[action][phase]=[i for i,lst in enumerate(responses[phase])if lst==[loc['Pport'],tone['6kHz'],act[action],-1]]
    for action in actions:
        total[action]={phase:sorted(rwd[action][phase]+no_rwd[action][phase]) for phase in acq.keys()}

    all_counts['left_rwd']=count_shift_stay(rwd['left'],total['left'],total['right'],all_counts['left_rwd'],r)
    all_counts['left_none']=count_shift_stay(no_rwd['left'],total['left'],total['right'],all_counts['left_none'],r)
    all_counts['right_rwd']=count_shift_stay(rwd['right'],total['right'],total['left'],all_counts['right_rwd'],r)
    all_counts['right_none']=count_shift_stay(no_rwd['right'],total['right'],total['left'],all_counts['right_none'],r)
    return all_counts,responses

def combined_bandit_Qhx_response(random_order,num_blocks,traject_dict,Qhx,boundaries,ept,phases):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    from TD2Q_Qhx_graphs import agent_response,plot_Qhx_2D
    
    fig=plt.figure()
    gs=GridSpec(2,2) # 2 rows, 2 columns
    ax1=fig.add_subplot(gs[0,:]) # First row, span all columns
    ax2=fig.add_subplot(gs[1,0]) # 2nd row, 1st column
    ax3=fig.add_subplot(gs[1,1]) # 2nd row, 2nd column
    agent_response([-1],random_order,num_blocks,traject_dict,fig,ax1)
    fig=plot_Qhx_2D(Qhx,boundaries,ept,phases,fig=fig,ax=[ax2,ax3])  
    
    #add subplot labels 
    letters=['A','B']
    fsize=14
    blank=0.03
    label_inc=(1-2*blank)/len(letters)
    for row in range(len(letters)):
        y=(1-blank)-(row*label_inc) #subtract because 0 is at bottom
        fig.text(0.02,y,letters[row], fontsize=fsize)
    fig.tight_layout()
    return fig

######################################### "main" ###########################################################################
events_per_trial=3  #this is task specific
trials=100 
numevents= events_per_trial*trials
runs=40 #Hamid et al uses 14 rats. 40 gives smooth trajectories
noise=0.15 #make noise small enough or state_thresh small enough to minimize new states in acquisition
#control output
printR=False #print environment Reward matrix
Info=False#print information for debugging
plot_hist=0#1: plot final Q, 2: plot the time since last reward, etc.
plot_Qhx=2 #2D or 3D plot of Q dynamics.  if 1, then plot agent responses
save_data=True

action_items=[(('Pport','6kHz'),'left'),(('Pport','6kHz'),'right')]
########### #For each run, randomize the order of this sequence #############
prob_sets={'50:50':{'L':0.5, 'R': 0.5},'10:50':{'L':0.1,'R':0.5}
           ,'90:50':{'L':0.9, 'R': 0.5},'90:10':{'L':0.9, 'R': 0.1},
           '50:90':{'L':0.5, 'R': 0.9},'50:10':{'L':0.5, 'R': 0.1},'10:90':{'L':0.1,'R':0.9}}  
#prob_sets={'20:80':{'L':0.2,'R':0.8},'80:20':{'L':0.8,'R':0.2}}
learn_phases=list(prob_sets.keys())
figure_sets=[list(prob_sets.keys())]
traject_items={phs:action_items+['rwd'] for phs in learn_phases}

cues=[]
trial_subset=int(0.1*numevents) #display mean reward and count actions over 1st and last of these number of trials 
#update some parameters of the agent
params['Q2other']=0.1 
params['numQ']=2
params['events_per_trial']=events_per_trial
params['wt_learning']=False
params['distance']='Euclidean'
params['beta_min']=0.1#params['beta'] #increased exploration after a mistake
params['moving_avg_window']=3  #This in units of trials, the actual window is this times the number of events per trial
params['decision_rule']=None#'delta'
params['split']=True
divide_rwd_by_prob=False
non_rwd=rwd['base'] #rwd['base'] or rwd['error'] #### base is better

if params['distance']=='Euclidean':
    #state_thresh={'Q1':[0.875,0],'Q2':[0.875,0.75]} #For Euclidean distance
    #alpha={'Q1':[0.6,0],'Q2':[0.6,0.3]}    #For Euclidean distance
    state_thresh={'Q1':[1.0,0],'Q2':[0.75,0.625]} #For normalized Euclidean distance
    alpha={'Q1':[0.6,0],'Q2':[0.4,0.2]}    #For normalized Euclidean distance, 2x discrim values works with 100 trials

else:
    state_thresh={'Q1':[0.22, 0.22],'Q2':[0.20, 0.22]} #For Gaussian Mixture?, 
    alpha={'Q1':[0.4,0],'Q2':[0.4,0.2]}    #For Gaussian Mixture? [0.62,0.19] for beta=0.6, 1Q or 2Q;'

params['state_thresh']=state_thresh['Q'+str(params['numQ'])] #for euclidean distance, no noise
#lower means more states for Euclidean distance rule
params['alpha']=alpha['Q'+str(params['numQ'])] #  

traject_title='num Q: '+str(params['numQ'])+', beta:'+str(params['beta_min'])+':'+str(params['beta'])+', non_rwd:'+str(non_rwd)+',rwd/p:'+str(divide_rwd_by_prob)
epochs=['Beg','End']

keys=rlu.construct_key(action_items +['rwd'],epochs)
resultslist={phs:{k+'_'+ep:[] for k in keys.values() for ep in epochs} for phs in learn_phases}
traject_dict={phs:{ta:[] for ta in traject_items[phs]} for phs in learn_phases}

#count number of responses to the following actions:
results={phs:{a:{'Beg':[],'End':[]} for a in action_items+['rwd']} for phs in learn_phases}

### to plot performance vs trial block
trials_per_block=10
events_per_block=trials_per_block* events_per_trial
num_blocks=int((numevents+1)/events_per_block)
params['events_per_block']=events_per_block
params['trials_per_block']=trials_per_block
params['trial_subset']=trial_subset
resultslist['params']={p:[] for p in params.keys()}

random_order=[]
key_list=list(prob_sets.keys())
######## Initiate dictionaries storing stay shift counts
all_counts={'left_rwd':{},'left_none':{},'right_rwd':{},'right_none':{}}
for key,counts in all_counts.items():
    for phase in learn_phases:
        counts[phase]={'stay':[0]*runs,'shift':[0]*runs}
wrong_actions={aaa:[0]*runs for aaa in ['wander','hold']}
all_beta=[];all_lenQ=[];all_Qhx=[]; all_bounds=[]; all_ideals=[]
Qhx_states=[('Pport','6kHz')]
Qhx_actions=['left','right']
for r in range(runs):
    #randomize prob_sets
    acqQ={};acq={};beta=[];lenQ={q:[] for q in range(params['numQ'])}
    random_order.append([k for k in key_list]) #keep track of order of probabilities
    print('*****************************************************\n************** run',r,'prob order',key_list)
    for phs in key_list:
        prob=prob_sets[phs]
        print('$$$$$$$$$$$$$$$$$$$$$ run',r,'prob',phs,prob)
        #do not scale these rewards by prob since the experiments did not
        Tbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[((loc['Lport'],tone['success']),prob['L']),((loc['Lport'],tone['error']),1-prob['L'])] #hear tone in poke port, go left, in left port/success
        Tbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[((loc['Rport'],tone['success']),prob['R']),((loc['Rport'],tone['error']),1-prob['R'])]
        if divide_rwd_by_prob:
            Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward']/prob['L'],prob['L']),(non_rwd,1-prob['L'])]   #lick in left port - 90% reward   
            Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward']/prob['R'],prob['R']),(non_rwd,1-prob['R'])] 
        else:
            Rbandit[(loc['Pport'],tone['6kHz'])][act['left']]=[(rwd['reward'],prob['L']),(non_rwd,1-prob['L'])]   #lick in left port - 90% reward   
            Rbandit[(loc['Pport'],tone['6kHz'])][act['right']]=[(rwd['reward'],prob['R']),(non_rwd,1-prob['R'])] 

        acq[phs] = RL(tone_discrim.completeT, QL.QL, states,act,Rbandit,Tbandit,params,env_params,printR=printR,oldQ=acqQ)
        results,acqQ=rlu.run_sims(acq[phs], phs,numevents,trial_subset,action_items,noise,Info,cues,r,results,phist=plot_hist)
        for aaa in ['wander','hold']:
            wrong_actions[aaa][r]+=acq[phs].results['action'].count(act[aaa])
        traject_dict=acq[phs].trajectory(traject_dict, traject_items,events_per_block)
        #print ('prob complete',acq.keys(), 'results',results[phs],'traject',traject_dict[phs])
        beta.append(acq[phs].agent.learn_hist['beta'])
        for q,qlen in acq[phs].agent.learn_hist['lenQ'].items():
            lenQ[q].append(qlen)
    np.random.shuffle(key_list) #shuffle after run complete, so that first run does 50:50 first    
    ###### Count stay vs shift 
    all_counts,responses=shift_stay_list(acq,all_counts)
    #store beta, lenQ, Qhx, boundaries,ideal_states from the set of phases in a single trial/agent    
    all_beta.append([b for bb in beta for b in bb])
    all_lenQ.append({q:[b for bb in lenQ[q] for b in bb] for q in lenQ.keys()})
    agents=list(acq.values()) 
    Qhx, boundaries,ideal_states=Qhx_multiphase(Qhx_states,Qhx_actions,agents,params['numQ'])
all_bounds.append(boundaries)
all_Qhx.append(Qhx)
all_ideals.append(ideal_states)

all_ta=[];output_data={}
for phs in traject_dict.keys():
    output_data[phs]={}
    for ta in traject_dict[phs].keys():
        all_ta.append(ta)
        output_data[phs][ta]={'mean':np.mean(traject_dict[phs][ta],axis=0),'sterr':np.std(traject_dict[phs][ta],axis=0)/np.sqrt(runs-1)}
all_ta=list(set(all_ta))
#move reward to front
all_ta.insert(0, all_ta.pop(all_ta.index('rwd')))
for p in resultslist['params'].keys():             #
    resultslist['params'][p].append(params[p])                #
resultslist=rlu.save_results(results,keys,resultslist)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print(' Using',params['numQ'], 'Q, alpha=',params['alpha'],'thresh',params['state_thresh'], 'beta=',params['beta'],'runs',runs,'of total events',numevents)
print(' apply learning_weights:',[k+':'+str(params[k]) for k in params.keys() if k.startswith('wt')])
print('Q2 hetero=',params['Q2other'],'decision rule=',params['decision_rule'])
print('counts from ',trial_subset,' events (',events_per_trial,' events per trial)          BEGIN    END    std over ',runs,'runs')
for phase in results.keys():
    for sa,counts in results[phase].items():
        print(phase,prob_sets[phase], sa,':::',np.round(np.mean(counts['Beg']),1),'+/-',np.round(np.std(counts['Beg']),2),
              ',', np.round(np.mean(counts['End']),1),'+/-',np.round(np.std(counts['End']),2))
        if sa in resultslist[phase]:
            print( '            ',sa,':::',[round(val,3) for lst in resultslist[phase][sa] for val in lst] )
print('$$$$$$$$$$$$$ total End reward=',np.sum([np.mean(results[k]['rwd']['End']) for k in results.keys()]))
print('divide by reward prob=',divide_rwd_by_prob,',non reward value', non_rwd)
fractionLeft={k:[] for k in results.keys()}
for k in results.keys():
    #print(k,'Left',results[k][(('Pport', '6kHz'),'left')]['End'])
    #print(k,'right',results[k][(('Pport', '6kHz'),'right')]['End'])
    for a,b in zip(results[k][(('Pport', '6kHz'),'left')]['End'],results[k][(('Pport', '6kHz'),'right')]['End']):
        if (a+b)>0:
            fractionLeft[k].append(round(a/(a+b),4))
        else:
            fractionLeft[k].append(np.nan)
    print('fraction left for',k,fractionLeft[k])
for k in fractionLeft.keys():
    print(k,'mean Left',round(np.nanmean(fractionLeft[k]),2), 
      ', std',round(np.nanstd(fractionLeft[k]),2), '::: trials with: no response', fractionLeft[k].count(np.nan), 
      ', no L',results[k][(('Pport', '6kHz'),'left')]['End'].count(0),', no R',results[k][(('Pport', '6kHz'),'right')]['End'].count(0))
            
for phs in all_counts['left_rwd'].keys():
    print('\n*******',phs,'******')
    for key,counts in all_counts.items():
        print(key,':::\n   stay',counts[phs]['stay'],'\n   shift',counts[phs]['shift'])
        ratio=[stay/(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift']) if stay+shift>0 ]
        events=[(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift'])]
        print(key,'mean stay=',round(np.mean(ratio),3),'+/-',round(np.std(ratio),3), 'out of', np.mean(events), 'events per trial')
print('wrong actions',[(aaa,np.mean(wrong_actions[aaa])) for aaa in wrong_actions.keys()])
if save_data:
    import datetime
    dt=datetime.datetime.today()
    date=str(dt).split()[0]
    #fname='Bandit'+date+'DecisionRule'+str(params['decision_rule'])+'_numQ'+str(params['numQ'])\
    fname='Bandit'+date+'_numQ'+str(params['numQ'])+'_alpha'+'_'.join([str(a) for a in params['alpha']])\
    +'_q2o'+str(params['Q2other'])+'_beta'+str(params['beta_min'])+'_split'+str(params['split'])+'_window'+str(params['moving_avg_window'])
    #'_st'+'_'.join([str(st) for st in params['state_thresh']])
    np.savez(fname,par=params,results=resultslist,traject=output_data,traject_dict=traject_dict,shift_stay=all_counts)

rlu.plot_trajectory(output_data,traject_title,figure_sets)
plot_prob_traject(output_data,params)

if plot_Qhx:
    ############## Ideally put this in TD2Q_manuscript_graphs.py
    #need to save traject_dict to do that, OR 
    #instead of responses per block, do moving average?
    from TD2Q_Qhx_graphs import agent_response
    display_runs=range(min(3,runs))
    figs=agent_response(display_runs,random_order,num_blocks,traject_dict)
    phases=list(acq.keys())
    if save_data:
        np.savez('Qhx'+fname,all_Qhx=all_Qhx,all_bounds=all_bounds,events_per_trial=events_per_trial,phases=phases,
        all_ideals=all_ideals,random_order=random_order,num_blocks=num_blocks,all_beta=all_beta,all_lenQ=all_lenQ)
    
if plot_Qhx==2:
    from TD2Q_Qhx_graphs import plot_Qhx_2D 
    #plot Qhx and response from last agent/trial      
    fig=plot_Qhx_2D(Qhx,boundaries,params['events_per_trial'],phases)   
    ########## combined figure ##############   
    fig=combined_bandit_Qhx_response(random_order,num_blocks,traject_dict,Qhx,boundaries,params['events_per_trial'],phases)
elif plot_Qhx==3: 
    ### B. 3D plot Q history for selected actions, for all states, one graph per phase
    for rl in acq.values():
        rl.agent.plot_Qdynamics(['left','right'],'surf',title=rl.name)

select_states=['suc','Ppo']
for i in range(params['numQ']):
    acq['50:50'].agent.visual(acq['50:50'].agent.Q[i],labels=acq['50:50'].state_to_words(i,noise),title='50:50 Q'+str(i),state_subset=select_states)
  
