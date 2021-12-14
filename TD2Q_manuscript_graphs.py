# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:55:57 2021

@author: kblackw1
"""
import numpy as np
import glob
'''
Update figures to show mean responses per trial?  Currently showing mean responses per 10 trials
Q matrix dynamics
'''

import matplotlib.pyplot as plt
plt.ion()
colors=[plt.get_cmap('Blues'),plt.get_cmap('Reds'),plt.get_cmap('Greys')] #plasma, viridis, inferno or magma possible
color_offset=0
letters=['A','B','C','D','E']
fsize=12

    
def create_traject_fig(traject,phases,actions,action_text,leg_panel=0, leg_loc='best',leg_fs=fsize,sequential=False):
    fig,ax=plt.subplots(len(actions),1,sharex=True)
    ax=fig.axes
    color_inc=int((255-color_offset)/(len(traject.keys())))
    blank=0.03
    ymin={act:0 for act in action_text}
    ymax={act:0 for act in action_text}
    for nQ,(numQ,data) in enumerate(traject.items()): #some figures look better if for pnum,phase comes first
        for pnum,phase in enumerate(phases):
            if len(phases)<=len(colors):
                cmap=pnum
                cnum=int(nQ+1)*color_inc+color_offset
                leg_cols=len(phases)
            else:
                color_inc=int((255-color_offset)/(len(data.keys())))
                cmap=nQ
                cnum=int(color_offset+(pnum+1)*color_inc)
                leg_cols=len(traject.keys())
            print(phase,'Q=',numQ,nQ,'c=',cnum)
            color=colors[cmap].__call__(cnum)
            label_inc=(1-3*blank)/len(actions) #used for putting subplot labels
            ########## Text for legend ###########
            if len(phases)>len(colors):
               trace_label=phase
               legend_title=''.join([numQ+'Q               ' for numQ in traject.keys()])
               #leg_fs=10
            elif len(phases)>1:
                trace_label=phase
                if numQ.isdigit(): #don't add 'numQ=' if traject.keys() are not numQ
                    legend_title='      '.join(['numQ='+str(numQ) for numQ in traject.keys()])
                else:
                    legend_title=''
                    trace_label=phase+', '+numQ
                #if numQ.isdigit(): #don't add 'Q' if traject.keys() are not numQ
                #    trace_label=trace_label+'Q'
                #
            else: #acquisition only
                trace_label=numQ
                legend_title='num Q'
            for anum,act in enumerate(actions):
                if act in data[phase].keys():
                    num_blocks=len(data[phase][act]['mean'])
                    if sequential:
                        block=np.arange(num_blocks)+(pnum*num_blocks)
                    else:
                        block=np.arange(num_blocks)
                    ymax[action_text[anum]]=max(np.max(data[phase][act]['mean']+data[phase][act]['sterr']),ymax[action_text[anum]])
                    ymin[action_text[anum]]=min(ymin[action_text[anum]],np.min(data[phase][act]['mean']-data[phase][act]['sterr']))
                    ax[anum].errorbar(block,data[phase][act]['mean'],yerr=data[phase][act]['sterr'],label= trace_label,color=color)
                    if act=='rwd':
                        ax[anum].hlines(0,xmin=0,xmax=np.max(block),linestyle='dotted')  
        for anum in range(len(ax)):
            ax[anum].set_ylabel(action_text[anum], fontsize=fsize+1)
            ax[anum].set_ylim([ymin[action_text[anum]],ymax[action_text[anum]]])
            ylim=ax[anum].get_ylim()
            ax[anum].set_ylim([np.floor(ylim[0]),np.ceil(ylim[1])])
            y=(1-blank)-(anum*label_inc) #subtract because 0 is at bottom
            if len(actions)>1:
                fig.text(0.02,y,letters[anum], fontsize=fsize+2)
            ax[anum].tick_params(axis='x', labelsize=fsize)
            ax[anum].tick_params(axis='y', labelsize=fsize)                
            if sequential:
                ax[anum].set_xlim([0,num_blocks*(pnum+1)])
            else:
                ax[anum].set_xlim([0,num_blocks])
            if leg_fs>0:
                leg=ax[leg_panel].legend(frameon=True,title=legend_title,ncol=leg_cols,loc=leg_loc,fontsize=leg_fs-1,title_fontsize=leg_fs-1,handletextpad=0.2,labelspacing=0.3,columnspacing=1)# markerfirst=True
    ax[anum].set_xlabel('Block', fontsize=fsize+1)
    plt.show()
    return fig #so you can adjust size and then do fig.tight_layout()

def create_sequence_traject_fig(traject,actions,action_text):
    fig,ax=plt.subplots(len(actions),1,sharex=True)
    ax=fig.axes
    blank=0.03
    for numQ,data in traject.items():
        color=colors[0].__call__(int(numQ)*127)
        label_inc=(1-2*blank)/len(actions) #used for putting subplot labels
        ########## Text for legend ###########
        for anum,act in enumerate(actions):
            num_blocks=len(data[act]['mean'])
            block=np.arange(num_blocks)
            ax[anum].errorbar(block,data[act]['mean'],yerr=data[act]['sterr'],label= numQ,color=color)
            ax[anum].set_ylabel(action_text[anum], fontsize=fsize)
            if act=='rwd':
                ax[anum].hlines(0,xmin=0,xmax=num_blocks,linestyle='dotted')                    
            y=(1-blank)-(anum*label_inc) #subtract because 0 is at bottom
            if len(actions)>1:
                fig.text(0.02,y,letters[anum], fontsize=fsize)
        ax[anum].set_xlim([0,num_blocks])
        ax[anum].set_xlabel('block', fontsize=fsize)
        ax[0].legend(frameon=True,title='num Q')
    plt.show()
    return fig #so you can adjust size and then do fig.tight_layout()

def create_bandit_fig(traject, numpanels=2):
    fig,ax=plt.subplots(numpanels,1,sharex=True)
    ax=fig.axes
    blank=0.03
    label_inc=(1-2*blank)/len(ax) #used for putting subplot labels
    leg_text=''
    for nQ, (numQ,data) in enumerate(traject.items()):
        leg_text=leg_text+numQ+'Q               '
        color_inc=int((255-color_offset)/(len(data.keys())))
        print(color_inc)
        if numpanels==1:
            axnum=0
        else:
            axnum=nQ
        ax[axnum].tick_params(axis='x', labelsize=fsize)
        ax[axnum].tick_params(axis='y', labelsize=fsize)
        for pnum,phs in enumerate(data.keys()):
            color=colors[nQ].__call__(int(color_offset+(pnum+1)*color_inc))
            num_blocks=len(data[phs])
            block=np.arange(num_blocks)
            ax[axnum].plot(block,data[phs],label= phs,color=color)
            ax[axnum].set_ylabel('Prob (L)', fontsize=fsize)
            ax[axnum].hlines(0.5,xmin=0,xmax=num_blocks,linestyle='dotted')                    
        if numpanels>1:
            ax[axnum].legend(frameon=True, title=numQ+'Q', fontsize=fsize)
        if len(ax)>1:
            y=(1-blank)-(nQ*label_inc) #subtract because 0 is at bottom
            fig.text(0.02,y,letters[nQ], fontsize=fsize)
    if numpanels==1:
        ax[0].legend(frameon=True,title=leg_text,ncol=len(traject.keys()), fontsize=fsize-1,title_fontsize=fsize-1,handletextpad=0.2,labelspacing=0.3,columnspacing=1)
    ax[axnum].set_xlim([0,num_blocks])
    ax[axnum].set_xlabel('Block', fontsize=fsize)
    plt.show()
    return fig #so you can adjust size and then do fig.tight_layout()

def read_data(pattern, files=None, keys=None):
    if not files:
        files=glob.glob(pattern)
    print('pattern',pattern,'files',files)
    traject={}
    all_counts={}
    for k,f in enumerate(files):
        data=np.load(f,allow_pickle=True)
        if keys is None:
            numQ=f.split('numQ')[-1][0]
        else:
            numQ=keys[k]
        traject[numQ]=data['traject'].item()
        if 'shift_stay' in data.keys():
            all_counts[numQ]=data['shift_stay'].item()
        del data   
    return traject,all_counts
  
################## Stat Analysis  #########################
def create_df(pattern,del_variables=[],params=['numQ']):
    files=glob.glob(pattern)
    print('pattern',pattern,'files',files)
    df=[]
    for f in files:
        par={}
        data=np.load(f,allow_pickle=True)
        print(f,list(data.keys()))
        results=data['results'].item()
        for p in params:
            par[p]=results['params'][p][0]
            #numQ=str(results['params']['numQ'][0])
        del results['params']
        key_combos=list(results.keys())
        if type(results[key_combos[0]])==dict:
            #if results is dictionary of dictonaries, need to flatten it
            new_results={}
            for phase in key_combos:
                for key,vals in results[phase].items():
                    new_results[phase+'_'+key]=vals[0] #shape is 1 row by blocks columns
            for zr in del_variables:
                del new_results[zr]
            results=new_results 
        elif len(np.shape(results[key_combos[0]]))==2:
            new_results={}
            for phs in key_combos:
                new_results[phs]=results[phs][0]
            for zr in del_variables:
                del new_results[zr]
            results=new_results
        dfsubset=pd.DataFrame.from_dict(results,orient='index').transpose()
        for p in params:
            pval=[par[p] for i  in range(len(dfsubset))]
            dfsubset[p]=pval
        #nQ=[numQ for i in range(len(dfsubset))]
        #dfsubset['numQ']=nQ
        df.append(dfsubset)
    alldf=pd.concat(df).reset_index() #concatentate everything into big dictionary
    return alldf

if __name__ == "__main__":
    discrim=1
    sequence=0
    block_da=0
    bandit=0
    ######################### DISCRIM #########################
    if discrim:
        pattern='Discrim2021-12-13_numQ?_alpha*.npz'
        traject,_=read_data(pattern) 
        ##### acquisition ######
        phase=['acquire']
        actions=['rwd',(('Pport', '6kHz'),'left')]
        action_text=['reward','6 kHz Left']
        figA=create_traject_fig(traject,phase,actions,action_text) #Fig 2A,B
                
        ##### extinction, renewal #####
        phase=['extinc','renew']
        actions=[(('Pport', '6kHz'),'left')]
        action_text=['6 kHz Left']
        figE=create_traject_fig(traject,phase,actions,action_text) #Fig 2C
        
        ##### discrimination, reversal #####
        phase=['discrim','reverse']
        actions=['rwd',(('Pport', '6kHz'),'left'), (('Pport', '10kHz'),'right')]#(('Pport', '10kHz'),'left'),
        action_text=['reward','6 kHz Left', '10 kHz Right'] #
        figD=create_traject_fig(traject,phase,actions,action_text,leg_panel=1, sequential=True) #Fig 4
    
    ######################### block Dopamine #########################
    if block_da:
        pattern='DiscrimD2AIP*.npz'
        files=['DiscrimD2AIP2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625.npz','Discrim2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625.npz']
        keys=['block', 'ctrl']
        traject,_=read_data(pattern, files, keys) 
        phase=['acquire','discrim']
        #actions=['rwd',(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        #action_text=['reward','6 kHz Left', '10 kHz Right', '10 kHz Left']
        actions=[(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        action_text=['6 kHz Left', '10 kHz Right', '10 kHz Left']
        figB=create_traject_fig(traject,phase,actions,action_text,leg_panel=0,leg_loc='lower right', leg_fs=12) #Fig 6
        ax=figB.axes
        ax[0].set_ylim([0,11])
        ax[1].set_ylim([-0.3,8]) 
        ax[2].set_ylim([-0.3,8]) 
    #################### Sequence trajectory ##################
    if sequence:
        pattern='Sequence2021-07*.npz'
        traject,_=read_data(pattern) 
        actions=['rwd',(('Llever', '**LL'), 'goR'), (('Rlever', '*LLR'), 'press')]
        action_text=['reward', '**LL, go Right','*LLR press R']
        figS=create_sequence_traject_fig(traject,actions,action_text) #fig 7
    
    #################### Bandit Task Probabilities ##################
    if bandit:
        pattern='Bandit2021-12-14_numQ*_alpha*beta*.npz'#'Bandit2021-05-28*beta0.1.npz'#'Bandit2021-05-28_numQ2_alpha0.6_0.3_beta0.7.npz'#
        traject,shift_stay=read_data(pattern)
        p_choose_L={q:{} for q in traject.keys()}
        for numQ, data in traject.items():
            tmp_prob={}
            for prob in data.keys():
                tmp_prob[prob]=data[prob][(('Pport', '6kHz'), 'left')]['mean']/(data[prob][(('Pport', '6kHz'), 'left')]['mean']+data[prob][(('Pport', '6kHz'), 'right')]['mean'])
            p_choose_L[numQ]=dict(sorted(tmp_prob.items(),key=lambda item: float(item[0].split(':')[0])-float(item[0].split(':')[1]),reverse=True))
        figB=create_bandit_fig(p_choose_L,numpanels=2) #Fig 10A,B
        actions=[(('Pport', '6kHz'),'left'), (('Pport', '6kHz'),'right')]
        action_text=['6 kHz Left','6 kHz Right']
        tmp_phs=list(traject['2'].keys())
        phases=sorted(tmp_phs,key=lambda tmp_phs: float(tmp_phs.split(':')[0])-float(tmp_phs.split(':')[1]),reverse=True)
        figBT=create_traject_fig(traject,phases,actions,action_text,leg_fs=0) #Fig 10C,D
        for numQ,all_counts in shift_stay.items():
            print('\n ############################ numQ=',numQ)
            for phs in all_counts['left_rwd'].keys():
                print('\n*******',phs,'******')
                #print('left_rwd=',all_counts['left_rwd'][phs],'left_none=',all_counts['left_none'][phs],
                #      'right_rwd=',all_counts['right_rwd'][phs],'right_none=',all_counts['right_none'][phs])
                for key,counts in all_counts.items():
                    ratio=[stay/(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift']) if stay+shift>0 ]
                    events=[(stay+shift) for stay,shift in zip(counts[phs]['stay'],counts[phs]['shift'])]
                    print(key,round(np.mean(ratio),3),round(np.std(ratio),3), 'out of', np.mean(events), 'responses')
    
    import pandas as pd
    from scipy.stats import ttest_ind
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    ########### Discrim ###########
    if discrim or block_da:
        zero_responses=['extinc_Pport_10kHz_left_Beg','extinc_Pport_10kHz_left_End', 'extinc_Pport_10kHz_right_Beg', 'extinc_Pport_10kHz_right_End',
                        'renew_Pport_10kHz_left_Beg', 'renew_Pport_10kHz_left_End', 'renew_Pport_10kHz_right_Beg', 'renew_Pport_10kHz_right_End',
                        'acquire_Pport_10kHz_left_Beg', 'acquire_Pport_10kHz_left_End', 'acquire_Pport_10kHz_right_Beg', 'acquire_Pport_10kHz_right_End']
        
        test_variables=['acquire_rwd__End','discrim_rwd__End','reverse_rwd__End', 
                        'discrim_Pport_6kHz_left_End','discrim_Pport_10kHz_right_End',
                        'reverse_Pport_6kHz_right_End','reverse_Pport_10kHz_left_End',]
        ########## Sequence ##########
    if sequence:
        #pattern='Sequence*.npz'
        test_variables=['rwd__End','*_**LL_goR_End', 'Rlever_*LLR_press_End']
   
    ########## bandit ##########
    if bandit:
        test_variables=[k+'_rwd__End' for k in ['50:50','10:50','10:90','50:90','90:10','90:50','50:10']]

    dep_var=['numQ','beta_min']
    df=create_df(pattern,params=dep_var)
    if pattern.startswith('Bandit'):
        df['total_reward']=df.loc[:,test_variables].sum(axis=1)
        test_variables=['total_reward']
    
    print(df.groupby(dep_var)[test_variables].aggregate(['mean','std','count']))
    mean=df.groupby(dep_var)[test_variables].mean()
    std=df.groupby(dep_var)[test_variables].std()
    cnt=df.groupby(dep_var)[test_variables].count()
    textname=pattern.replace('?','x').replace('*','all_').split('.npz')[0]+'summary.txt'
    columns=[tv[0:-4] for tv in test_variables]
    header=' '.join(cnt.index.names)+' '+'  '.join(columns)
    rows=list(cnt.index.values)
    #np.savetxt(textname,np.column_stack((rows*3,np.vstack([cnt,mean,std]))),fmt='%4.3f',header=header,comments='') #Ttest tables/Fig 5 - Igor
    
    for tv in test_variables:
        if len(np.unique(df[dep_var]))==2:
            unique_vals=np.unique(df[dep_var[0]])
            tt=ttest_ind(df[df[dep_var[0]]==unique_vals[0]][tv], df[df[dep_var[0]]==unique_vals[1]][tv], equal_var=False)
            print('\n*******',tv,'\n',tt)      
        else:
            dependents=['C('+dv+')' for dv in dep_var]
            model_statement=' ~ '+'+'.join(dependents)
            print(model_statement)
            model=ols(tv+model_statement,data=df).fit()
            print (tv, sm.stats.anova_lm (model, typ=2), '\n',model.summary())
    