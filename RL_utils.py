# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 2021

@author: kblackw1
"""
import numpy as np

def plot_trajectory(output_data,title,figure_sets):
    ############# Plots for publication ##########            
    from matplotlib import pyplot as plt
    plt.ion()
    colors=plt.get_cmap('inferno') #plasma, viridis, inferno or magma possible
    #colors=['k','r','b','gray']
    for phases in figure_sets:
        if len(phases)>1:
            color_increment=int((len(colors.colors)-40)/(len(phases)-1)) #40 to avoid to light colors
        else:
            color_increment=127
        all_items=list(reversed(list(set([item for phs in  phases for item in output_data[phs].keys()]))))
        panels=[len(output_data[phs].keys()) for phs in  phases]
        fig,axis=plt.subplots(nrows=max(panels),ncols=1,sharex=True)
        fig.suptitle(title)
        ymin=0;ymax=0
        for phs in phases:
            ymax= max(ymax,np.max([np.max(vals['mean']+vals['sterr']) for vals in output_data[phs].values()]))
            ymin= min(ymin,np.min([np.min(vals['mean']+vals['sterr']) for vals in output_data[phs].values()]) )
            cnum=phases.index(phs)*color_increment
            for ta,data in output_data[phs].items():
                ax=all_items.index(ta)
                #print(phs,ta,type(data['mean']))
                if isinstance(data['mean'],np.ndarray) or isinstance(data['mean'],list):
                    if np.all(np.isnan(data['sterr'])):
                        axis[ax].plot(range(len(data['mean'])),data['mean'],label=phs,c=colors.colors[cnum])
                    else:
                        axis[ax].errorbar(range(len(data['mean'])),data['mean'],yerr=data['sterr'],label=phs,capsize=5,c=colors.colors[cnum])
                if ta=='rwd':
                    axis[ax].set_ylabel('reward')
                    axis[ax].set_ylim([np.floor(ymin),np.ceil(ymax)])
                else:
                    if len(ta[0])>1:
                        ylabel=ta[0][0][0:4]+','+ta[0][1]+' '+ta[1][0:3]
                    else:
                        ylabel=ta[0][0]+' '+ta[1]
                    axis[ax].set_ylabel(ylabel)
                    axis[ax].set_ylim([0,np.ceil(ymax)*1.05]) 
                    #if phs == 'discrim' or phs == 'reverse':
                    #    axis[ax].set_ylim([0,10])
                    #else:
                    #    axis[ax].set_ylim([0,11])
                axis[ax].legend()
        axis[-1].set_xlabel('block')
    #plt.show()
    return fig

def save_results(results,key_dict,resultslist):  
    for phase in results.keys():
        if phase in resultslist.keys():
            for sacombo in results[phase].keys():
                for ep,counts in results[phase][sacombo].items():
                    resultslist[phase][key_dict[sacombo]+'_'+ep].append(counts)
    return resultslist
'''
def save_results(results,epochs,allresults,resultslist):
    for phase in results.keys():
        for ac,counts in results[phase].items():
            for ep in epochs:
                allresults[phase+'_'+ac+'_'+ep].append(np.round(np.mean(counts[ep]),3))
                print(phase,ac,ep,counts[ep])
                resultslist[phase+'_'+ac+'_'+ep].append(counts[ep])
    return allresults,resultslist
'''
def construct_key(state_actions,epochs=None):
    keys={}
    for sacombo in state_actions:
        if sacombo =='rwd':
            env=['rwd']
            ac=''
        else:
            env=sacombo[0]
            ac=sacombo[1]
        keys[sacombo]='_'.join(env)+'_'+ac
    return keys

def run_sims(RL,phase,events,n_subset,action_items,noise,info,cues,rr,summary,phist=0,block_DA=False): 
    #Need to add block_DA as input to episode and run_sims
    RL.episode(events,noise=noise,info=info,cues=cues,name=phase,block_DA=block_DA)
    rwd_prob=np.mean(RL.agent.learn_hist['rwd_prob'][-n_subset:])
    #summary,t2=RL.count_actions(summary,action_items,trial_subset)
    summary,t2=RL.count_state_action(summary,action_items,n_subset)
    Q={'Q':RL.agent.Q,'ideal_states':RL.agent.ideal_states,'learn_weight':RL.agent.learn_weight,'rwd_prob':rwd_prob,'name':phase}
    if hasattr(RL.agent,'V'):
        Q['V']=RL.agent.V
    if rr==0:
        if np.max(RL.results['reward'])>0:
            t2=',mean reward='+str(np.round(np.mean(RL.results['reward'][-n_subset:]),2))
        RL.set_of_plots(phase,noise,t2,hist=phist)
    return summary,Q

def beta_lenQ(all_agents,all_phases,all_beta,all_lenQ,numQ):
    for phase_set in all_phases:
        beta=[];lenQ={q:[] for q in range(numQ)}
        key='_'.join(phase_set)
        for phs in phase_set:
            beta.append(all_agents[phs].agent.learn_hist['beta'])
            for q,qlen in all_agents[phs].agent.learn_hist['lenQ'].items():
                lenQ[q].append(qlen)
        all_beta[key].append([b for bb in beta for b in bb])
        for q in lenQ.keys():
            all_lenQ[key][q].append([b for bb in lenQ[q] for b in bb]) 
 
    return all_beta,all_lenQ
