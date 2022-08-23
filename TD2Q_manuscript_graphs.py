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
fsize=11
import RL_utils as rlu
    
def create_traject_fig(traject,phases,actions,action_text,leg_panel=0, leg_loc='best',leg_fs=fsize,sequential=False,leg_text={},color_dict={}):
    fig,ax=plt.subplots(len(actions),1,sharex=True)
    ax=fig.axes
    color_inc=int((255-color_offset)/(len(traject.keys())))
    blank=0.03
    ymin={act:0 for act in action_text}
    ymax={act:0 for act in action_text}
    for nQ,(numQ,data) in enumerate(traject.items()): #some figures look better if for pnum,phase comes first
        for pnum,phase in enumerate(phases):
            if len(phases)<=len(colors):
                if sequential:
                    cmap=0
                else:
                    cmap=pnum
                cnum=int(nQ+1)*color_inc+color_offset
                leg_cols=len(phases)
            else:
                color_inc=int((255-color_offset)/(len(data.keys())))
                cmap=nQ
                cnum=int(color_offset+(pnum+1)*color_inc)
                leg_cols=len(traject.keys())
            print('traject_fit',phase,'Q=',numQ,nQ,'c=',cnum)
            if len(color_dict):
                color=color_dict[numQ][pnum]
                print('col_dict, color=',color)
            else:
                color=colors[cmap].__call__(cnum)
            label_inc=(1-3*blank)/len(actions) #used for putting subplot labels
            ########## Text for legend ###########
            if len(phases)>len(colors):
                trace_label=phase
                if len(leg_text):
                   trace_label=leg_text[phase]
                legend_title=''.join([numQ+'Q               ' for numQ in traject.keys()])
                #leg_fs=10
            elif len(phases)>1:
                trace_label=phase
                if len(leg_text):
                   trace_label=leg_text[phase]
                if sequential:
                    legend_title=''
                    trace_label='numQ='+numQ
                if numQ.isdigit(): #don't add 'numQ=' if traject.keys() are not numQ
                    legend_title=''#TEMPORARY'      '.join(['numQ='+str(numQ) for numQ in traject.keys()])
                    #trace_label=trace_label+'Q'
                else:
                    legend_title=''
                    trace_label=phase+', '+numQ                    
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
                    if pnum>0: #TEMPORARY - ELIMINATE EXTRA LEGEND
                        trace_label='_'
                    ax[anum].errorbar(block,data[phase][act]['mean'],yerr=data[phase][act]['sterr'],label= trace_label,color=color)
                    if act=='rwd':
                        ax[anum].hlines(0,xmin=0,xmax=np.max(block),linestyle='dotted') 
        maxy=np.max([axis.get_ylim()[1] for axis in ax[1:]] )
        miny=np.min([axis.get_ylim()[0] for axis in ax[1:]] )
        for anum in range(len(ax)):
            ax[anum].set_ylabel(action_text[anum], fontsize=fsize+1)
            ax[anum].set_ylim([ymin[action_text[anum]],ymax[action_text[anum]]])
            ylim=ax[anum].get_ylim()
            if action_text[anum]=='reward':
                ax[anum].set_ylim([np.floor(ylim[0]),np.ceil(ylim[1])*1.05]) 
                yticks=np.linspace(np.floor(ylim[0]),np.ceil(ylim[1]),4)
                ylabels=[str(round(f)) for f in yticks]
                ax[anum].set_yticks(yticks,ylabels)
            else:
                ax[anum].set_ylim([np.floor(miny),np.ceil(maxy)]) 
            y=(1-blank)-(anum*label_inc) #subtract because 0 is at bottom
            #if len(actions)>1:
            #    fig.text(0.02,y,letters[anum], fontsize=fsize+2)
            ax[anum].tick_params(axis='x', labelsize=fsize)
            ax[anum].tick_params(axis='y', labelsize=fsize)                
            if sequential:
                ax[anum].set_xlim([0,num_blocks*(pnum+1)])
            else:
                ax[anum].set_xlim([0,num_blocks])
            #if leg_fs>0:#TEMPORARY - ELIMINATE EXTRA LEGEND
            #    leg=ax[leg_panel].legend(frameon=True,title=legend_title,ncol=leg_cols,loc=leg_loc,fontsize=leg_fs-1,title_fontsize=leg_fs-1,handletextpad=0.2,labelspacing=0.3,columnspacing=1)# markerfirst=True
    ax[anum].set_xlabel('Block', fontsize=fsize+1)
    plt.show()
    return fig #so you can adjust size and then do fig.tight_layout()

def create_sequence_traject_fig(traject,actions,action_text,lenQ={},ept=7):
    if len(lenQ):
        numrows=len(actions)+1
    else:
        numrows=len(actions)
    fig,ax=plt.subplots(numrows,1,sharex=False)
    ax=fig.axes
    blank=0.03
    for ijk,(numQ,data) in enumerate(traject.items()):
        incr=int(255/len(traject.keys()))
        color=colors[0].__call__((ijk+1)*incr)
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
            #if len(actions)>1:
            #    fig.text(0.02,y,letters[anum], fontsize=fsize)
            ax[anum].set_xlim([0,num_blocks])

        ax[anum].set_xlabel('Block', fontsize=fsize)
        ax[0].legend(frameon=True,title='num Q')
    if len(lenQ):
        #lenQ[1]=lenQ[2] for numQ=2, so plot lenQ for both numQ=1 and 2
        for q,data in lenQ.items():
            color=colors[0].__call__(int(q)*incr)
            Xvals=np.arange(np.shape(data[int(q)])[1])/ept
            ax[anum+1].plot(Xvals,np.mean(data[int(q)],axis=0),label='numQ='+str(q),color=color)
        #ax[anum+1].legend()
        ax[anum+1].set_ylabel('States')
        ax[-1].set_xlabel('Trial')
    for anum in range(1,numrows):
        if anum==numrows-1:
            yticks=[0,20,40,60]
        else:
            yticks=[0,5,10]
        ylabels=[str(round(f)) for f in yticks]
        ax[anum].set_yticks(yticks,ylabels)
    ax[0].set_ylim(-7.5,5.0)

    plt.show()
    return fig #so you can adjust size and then do fig.tight_layout()

def create_bandit_fig(traject, numpanels=2,color_dict={}):
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
            if len(color_dict):
                color=color_dict[numQ][pnum]
            else:
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
    ax[axnum].set_xlim([0,num_blocks+1])
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
        elif 'all_beta' in data.keys(): #files beginning on 3 June 2022
           all_counts[numQ]=data['all_lenQ'].item()
        if 'sa_errors' in data.keys():
            switch=[tuple(sa) for sa in data['sa_errors'].item()['switch']]
            switch_keys=rlu.construct_key(switch)
            overstay=[tuple(sa) for sa in data['sa_errors'].item()['stay']]
            overstay_keys=rlu.construct_key(overstay)
            start=[tuple(sa) for sa in data['sa_errors'].item()['start']]
            start_keys=rlu.construct_key(start)
            sa_error_keys={'switch':switch_keys,'overstay':overstay_keys,'start':start_keys}
        else:
            sa_error_keys={}
        del data   
    return traject,all_counts,sa_error_keys
  
################## Stat Analysis  #########################
def create_df(pattern,files=None,del_variables=[],params=['numQ']):
    if not files:
        files=glob.glob(pattern)
    print('pattern',pattern,'files',files)
    df=[]
    for f in files:
        par={}
        data=np.load(f,allow_pickle=True)
        print(f,list(data.keys()))
        results=data['results'].item()
        print('params',results['params'])
        for p in params:
            if p in results['params'].keys():
                par[p]=results['params'][p][0]
            else:
                par[p]=-1
            #numQ=str(results['params']['numQ'][0])
        del results['params']
        key_combos=list(results.keys())
        new_results={}
        if type(results[key_combos[0]])==dict:
            #if results is dictionary of dictonaries, need to flatten it
            for phase in key_combos:
                for key,vals in results[phase].items():
                    new_results[phase+'_'+key]=vals[0] #shape is 1 row by blocks columns
            for zr in del_variables:
                del new_results[zr]
        elif len(np.shape(results[key_combos[0]]))==2:
            for phs in key_combos:
                if '--' in phs:
                    new_key=phs.replace('-','s')
                else:
                    new_key=phs.replace('*','x')
                new_results[new_key]=results[phs][0]
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

def barplot(mean,sterr,variables,varname,ylabel='Reward',transpose=False):
    figS,ax=plt.subplots(figsize=(2,2.5)) #width, height
    if transpose:
        mean=mean.T
        sterr=sterr.T
        variables=mean.columns
    rows=list(mean.index.values)
    xlabels=['Ctrl' if x==-1 else varname+' '+str(x) for x in rows]
    if transpose:
        xlabels=[x[7:16] for x in rows]
    xvalues=np.arange(len(rows))
    w = 1./(len(variables)+0.5)
    for a,tv in enumerate(variables):
        #print(tv, type(tv))
        yvalues=mean[tv].values
        yerr=sterr[tv].values
        lbl='Ctrl' if tv==-1 else tv
        ax.bar(xvalues+(a-(len(variables)-1)/2)*w, yvalues,width=w,yerr=yerr,label=str(lbl)[0:8]) 
    ax.set_ylabel(ylabel)
    ax.hlines(0,np.min(xvalues),np.max(xvalues),linestyles='dashed',colors='gray')
    ax.set_xlabel('Condition')
    ax.set_xticks(xvalues,xlabels)
    if not tv.startswith('rwd'):
        ax.set_ylim([0,1.2])
        #ax.legend(loc='best',fontsize=9,frameon=False)
    figS.suptitle('+'.join([str(v) for v in variables]))
    return figS

def barplot_means(df,dep_var,test_variables):
    print(df.groupby(dep_var)[test_variables].aggregate(['mean','std','count']))
    mean=df.groupby(dep_var)[test_variables].mean()
    cnt=df.groupby(dep_var)[test_variables].count()
    sterr=df.groupby(dep_var)[test_variables].std()/np.sqrt(cnt-1)
    return mean,cnt,sterr

if __name__ == "__main__":
    discrim=0
    sequence=1
    block_da=0
    bandit=0
    files=None
    ######################### DISCRIM #########################
    if discrim:
        #2022 jun 14 sims used for Figs 2&3 - mean trajectory, example Q values and beta
        #also used for statistics comparing numQ=1 vs 2
        #2021 Dec sims used to assess the effect of beta, Q2other, State-splitting
        pattern='discrim2022-06-14_numQ?_alpha*.npz'
        dep_var=['numQ', 'split','beta_min']#,'Q2other'] #'decision_rule']##'trial_subset']# 
        files=['Discrim2021-12-17_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.9_splitTrue.npz',
                'Discrim2021-12-14_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitFalse.npz',
                'Discrim2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue.npz'] 
                #['Discrim2022-06-14_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.5_splitTrue.npz',
                #'Discrim2022-06-14_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue.npz']
                #             #'Discrim2021-12-13_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.5_splitTrue.npz']#,
                #'Discrim2021-12-17_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.9_splitTrue.npz',#Test split, beta_min
                #
                #
                #'Discrim2021-12-17_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.5_splitFalse.npz',
                #'Discrim2021-12-16_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0_beta0.5_splitTrue.npz', #test Q2other
                #'Discrim2021-12-17_numQ1_alpha0.3_0_st1.0_0_q2o0_beta0.5_splitTrue.npz']
                #'Discrim2021-12-17_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue_ruleDelta.npz', #test decision rule
                #'Discrim2021-12-17_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.5_splitTrue_ruleDelta.npz']
        traject,_,_=read_data(pattern,files=files) 
        ##### acquisition ######
        phase=['acquire']
        actions=['rwd',(('Pport', '6kHz'),'left')]
        action_text=['reward','6 kHz Left']
        #figA=create_traject_fig(traject,phase,actions,action_text) #Fig 2A,B
                
        ##### extinction, renewal #####
        phase=['acquire','extinc','renew'] #phase=['extinc','renew']
        actions=['rwd',(('Pport', '6kHz'),'left')]#[(('Pport', '6kHz'),'left')]
        action_text=['reward','6 kHz Left']
        #phase_text={'extinc':'Context B','renew':'Context A'}
        phase_text={'acquire':'','extinc':'','renew':''}
        #figE=create_traject_fig(traject,phase,actions,action_text,leg_text=phase_text) #Fig 2C
        #figE=create_traject_fig(traject,phase,actions,action_text,sequential=True) #Fig 2C
        
        ##### discrimination, reversal #####
        phase=['acquire','discrim','reverse']
        actions=['rwd', (('Pport', '10kHz'),'right'),(('Pport', '10kHz'),'left')]#,(('Pport', '6kHz'),'left')]
        action_text=['reward', '10 kHz Right', '10 kHz Left']#,'6 kHz Left'] #
        figD=create_traject_fig(traject,phase,actions,action_text,leg_panel=1, sequential=True) #Fig 4
        xlim=[19,61]
        ax=figD.axes
        for axis in ax:
            axis.set_xlim(xlim)

    ######################### block Dopamine #########################
    if block_da:
        pattern='DiscrimD2AIP*.npz'
        files=['DiscrimD2AIP2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625.npz','Discrim2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue.npz']
        keys=['block', 'ctrl']
        traject,_,_=read_data(pattern, files, keys) 
        phase=['acquire','discrim']
        #actions=['rwd',(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        #action_text=['reward','6 kHz Left', '10 kHz Right', '10 kHz Left']
        actions=[(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        action_text=['6 kHz Left', '10 kHz Right', '10 kHz Left']
        figB=create_traject_fig(traject,phase,actions,action_text,leg_panel=1,leg_loc='upper left', leg_fs=12, sequential=True) #Fig 6
        ax=figB.axes
        ax[0].set_ylim([0,11])
        ax[1].set_ylim([-0.3,8]) 
        ax[2].set_ylim([-0.3,8]) 
    #################### Sequence trajectory ##################
    if sequence:
        pattern='Sequence2022-08-16_*.npz'
        add_barplot=1
        dep_var=['numQ']
        test_var=[]
        #files=['Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz','Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz']
        files=[ 'Sequence2022-08-10_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz','Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz']
        if add_barplot:
            files=[ 'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz',
                    'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue_inactiveD1.npz',
                    'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue_inactiveD2.npz']

            keys=['Ctrl','inactiveD1','inactiveD2']#
            traject,all_lenQ,error_keys=read_data(pattern,files,keys)   
            dep_var=['inact']#'split','beta_min']#, 'Q2other','split','beta_min'] #'trial_subset']# 'decision_rule']#

            '''#'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz',
                    #'Sequence2021-12-17DecisionRuledelta_numQ2.npz', #test delta rule
                    #'Sequence2021-12-17DecisionRuledelta_numQ1.npz']
            #Additional files for testing effect of state_splitting, beta_min, Q2other
            files=['Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitFalse.npz',
                'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz',
                'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.5splitTrue.npz']
            files=['Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.0beta0.5splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.0beta0.9splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.5splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.25_st0.75_0.75_q2o0.0beta0.5splitTrue.npz',    
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.5splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitFalse.npz',
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitFalse.npz',]
            files=['Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitFalse.npz',
                    'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz',
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitFalse.npz',
                    'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz']
            '''       
        else:
            traject,all_lenQ,error_keys=read_data(pattern,files)#,keys) 

        switch=[kk.replace('*','x')+'_End' for kk in error_keys['switch'].values() if 'press' in kk ]
        overstay=[kk.replace('*','x')+'_End' for kk in error_keys['overstay'].values() if 'press' in kk]
        start=[kk.replace('*','x')+'_End' for kk in error_keys['start'].values()]
        actions=['rwd',(('Llever', '**LL'), 'goR'), (('Rlever', '*LLR'), 'press')]
        action_text=['reward', '**LL, go Right','*LLR press R']
        #figS=create_sequence_traject_fig(traject,actions,action_text,lenQ=all_lenQ) #fig 7
        #figS=create_sequence_traject_fig(traject,actions,action_text) #fig 7

    #################### Bandit Task Probabilities ##################
    if bandit:
        #bandit sims from 2021 dec 14-16 used for statistics and trajectory plots
        pattern='Bandit2021-12-14_numQ*_alpha*beta0.1.npz'#'Bandit2021-05-28*beta0.1.npz'#'Bandit2021-05-28_numQ2_alpha0.6_0.3_beta0.7.npz'#
        dep_var=['split','beta_min'] #,'numQ']#'Q2other', 'decision_rule']#'trial_subset']# 
        files=[ 'Bandit2021-12-14_numQ2_alpha0.4_0.2_q2o0.1_beta0.1splitTrue.npz',
                'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.1_beta0.9_splitTrue.npz',
                'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.1_beta0.1_splitFalse.npz']
                #'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.0_beta0.1_splitTrue.npz']
                #'Bandit2021-12-21_numQ2_alpha0.4_0.2_q2o0.1_beta0.1_splitTrue_window1.npz']
                #'Bandit2021-12-14_numQ1_alpha0.6_0_q2o0.1_beta0.1splitTrue.npz']
                 # #next four to test beta and split
                #'Bandit2021-12-16_numQ1_alpha0.6_0_q2o0.1_beta0.9_splitTrue.npz']
                #'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.1_beta0.9_splitTrue.npz', #next four to test beta and split
                #'Bandit2021-12-16_numQ1_alpha0.6_0_q2o0.1_beta0.9_splitTrue.npz',
                #'Bandit2021-12-16_numQ1_alpha0.6_0_q2o0.1_beta0.1_splitFalse.npz',
                #'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.1_beta0.1_splitFalse.npz']
                #'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.0_beta0.1_splitTrue.npz',  #to test q2o (Q2other)
                #'Bandit2021-12-16_numQ1_alpha0.6_0_q2o0.0_beta0.1_splitTrue.npz']
                #'Bandit2021-12-16_numQ1_alpha0.3_0_q2o0.1_beta0.1_splitTrue.npz', #to test number of trials (trial_subset)
                #'Bandit2021-12-16_numQ2_alpha0.2_0.1_q2o0.1_beta0.1_splitTrue.npz'] 
                #'Bandit2021-12-17DecisionRuledelta_numQ2_q2o0.1_beta0.1_splitTrue.npz', #to test decision rule
                #'Bandit2021-12-17DecisionRuledelta_numQ1_q2o0.1_beta0.1_splitTrue.npz']
        colors2=[plt.get_cmap('seismic').__call__(c) for c in [18,50]]+[plt.get_cmap('hsv').__call__(c) for c in [188,204,225]]+[plt.get_cmap('seismic').__call__(c) for c in [192,242]]
        newcol={'2':colors2,'1':colors2}
        ########## !!!!!!!!!!!!!! newcol from file at work - but increase the dark red by 5 to increase discriminability !!!!!!!!!!!!! ###########
        traject,shift_stay,_=read_data(pattern,files=files)
        p_choose_L={q:{} for q in traject.keys()}
        for numQ, data in traject.items():
            tmp_prob={}
            for prob in data.keys():
                tmp_prob[prob]=data[prob][(('Pport', '6kHz'), 'left')]['mean']/(data[prob][(('Pport', '6kHz'), 'left')]['mean']+data[prob][(('Pport', '6kHz'), 'right')]['mean'])
            p_choose_L[numQ]=dict(sorted(tmp_prob.items(),key=lambda item: float(item[0].split(':')[0])-float(item[0].split(':')[1]),reverse=True))
        figB=create_bandit_fig(p_choose_L,numpanels=2,color_dict=newcol) #Fig 10A,B
        actions=[(('Pport', '6kHz'),'left'), (('Pport', '6kHz'),'right')]
        action_text=['6 kHz Left','6 kHz Right']
        tmp_phs=list(traject['2'].keys())
        phases=sorted(tmp_phs,key=lambda tmp_phs: float(tmp_phs.split(':')[0])-float(tmp_phs.split(':')[1]),reverse=True)
        figBT=create_traject_fig(traject,phases,actions,action_text,leg_fs=0,color_dict=newcol) #Fig 10C,D
        '''
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
        '''
    import pandas as pd
    from scipy.stats import ttest_ind
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import scikit_posthocs as sp
    
    ########### Discrim ###########
    if discrim or block_da:
        zero_responses=['extinc_Pport_10kHz_left_Beg','extinc_Pport_10kHz_left_End', 'extinc_Pport_10kHz_right_Beg', 'extinc_Pport_10kHz_right_End',
                        'renew_Pport_10kHz_left_Beg', 'renew_Pport_10kHz_left_End', 'renew_Pport_10kHz_right_Beg', 'renew_Pport_10kHz_right_End',
                        'acquire_Pport_10kHz_left_Beg', 'acquire_Pport_10kHz_left_End', 'acquire_Pport_10kHz_right_Beg', 'acquire_Pport_10kHz_right_End']
        
        test_variables=['acquire_rwd__End','discrim_rwd__End','reverse_rwd__End'] 
        #test_variables=['renew_Pport_6kHz_left_Beg', 'extinc_Pport_6kHz_left_Beg']
                        #'discrim_Pport_6kHz_left_End','discrim_Pport_10kHz_right_End',
                        #'reverse_Pport_6kHz_right_End','reverse_Pport_10kHz_left_End',]
        ########## Sequence ##########
    if sequence:
        test_variables=['rwd__End']#,'Llever_**LL_goR_End'.replace('*','x'), 'Rlever_*LLR_press_End'.replace('*','x'),'Rlever_xxLL_press_End','Rlever_LLRR_goMag_End']
    ########## bandit ##########
    if bandit:
        test_variables=[k+'_rwd__End' for k in ['50:50','10:50','10:90','50:90','90:10','90:50','50:10']]

    df=create_df(pattern,files=files,params=dep_var)
    for dv in dep_var:
        df[dv].fillna(-1,inplace=True)
    if pattern.startswith('Bandit'):
        for prob in traject['2'].keys():
            df[prob+'_probL']=df[prob+'_Pport_6kHz_left_End']/(df[prob+'_Pport_6kHz_left_End']+df[prob+'_Pport_6kHz_right_End'])
        df['mean_reward']=df.loc[:,test_variables].mean(axis=1)
        test_variables=['mean_reward']
    if pattern.startswith('Discrim'):
        df['mean_reward']=df.loc[:,['reverse_rwd__End','discrim_rwd__End']].mean(axis=1)
        test_variables=['mean_reward']
    mean,cnt,sterr=barplot_means(df,dep_var,test_variables)
    
    textname=pattern.replace('?','x').replace('*','all_').split('.npz')[0]+'summary.txt'
    columns=[tv[0:-4] for tv in test_variables]
    header=' '.join(cnt.index.names)+' '+'  '.join(columns)
    rows=list(cnt.index.values)
    figBP=barplot(mean,sterr,test_variables,dep_var[0])
    np.savetxt(textname,np.column_stack((rows*3,np.vstack([cnt,round(mean,3),round(sterr,3)]))),fmt='%s',header=header,comments='') #Ttest tables/Fig 5 - Igor
    if bandit:
        newtv= [k+'_probL' for k in ['90:10','90:50','50:10', '50:50','10:50','50:90','10:90']] 
        print(df.groupby(dep_var)[newtv].mean())
    if sequence and add_barplot:
        df['Llever_1L_goR_End']=df['Llever_xxRL_goR_End']+df['Llever_xx-L_goR_End']
        df['Llever_1L_press_End']=df['Llever_xxRL_press_End']+df['Llever_xx-L_press_End']
        df['mag_ssss_nostart_End']=df['mag_ssss_goR_End']+df['mag_ssss_other_End']+df['mag_ssss_goMag_End']+df['mag_ssss_press_End']
        df['Llever_1L_press_End_Prob']=(df['Llever_1L_press_End'])/(df['Llever_1L_press_End']+df['Llever_xx-L_goR_End']+df['Llever_xx-L_goMag_End']+df['Llever_xx-L_other_End']+df['Llever_xxRL_other_End']+df['Llever_xxRL_goMag_End']+df['Llever_xxRL_goR_End'])
        df['Llever_2L_switch_End_Prob']=df['Llever_xxLL_goR_End']/(df['Llever_xxLL_goR_End']+df['Llever_xxLL_press_End']+df['Llever_xxLL_goL_End']+df['Llever_xxLL_goMag_End']+df['Llever_xxLL_other_End'])
        df['mag_sss_start_End_Prob']=df['mag_ssss_goL_End']/(df['mag_ssss_nostart_End']+df['mag_ssss_goL_End'])
        all_test_variables={#'overstay':['Llever_xxLL_goR_End','Llever_xxLL_press_End','Llever_xxLL_goL_End'], #correct vs over-stay
                        #'B':['Rlever_LLRR_goMag_End','Rlever_LLRR_press_End','Rlever_LLRR_goR_End'], #correct vs stay
                        #'C':['Rlever_xLLR_press_End','Rlever_xLLR_goL_End'], #correct vs switch
                        #'premature':['Llever_xxRL_press_End','Llever_1L_goR_End'],#,'Rlever_1L_press_End'],
                        #'start':['mag_ssss_goL_End', 'mag_ssss_nostart_End', ],# incorrect start
                        'all_three':['mag_ssss_goL_End','Llever_1L_press_End','Llever_xxLL_goR_End'],
                        'prob':['mag_sss_start_End_Prob','Llever_1L_press_End_Prob','Llever_2L_switch_End_Prob']} 
        for kk,test_var in all_test_variables.items():
            mean,cnt,sterr=barplot_means(df,dep_var,test_var)
            #figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Responses per Trial')
            if kk=='prob':
                figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Probability',transpose=True)
            else:
                figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Responses per Trial',transpose=True)
    for tv in test_variables+test_var:
        new_dep_var=[]
        for dv in dep_var:
            if df[dv].nunique()>1:
                new_dep_var.append(dv)
            else:
                print('STATS: only 1 level for variable=', dv)
        if len(dep_var)>len(new_dep_var):
            print('proposed dependent variables:',dep_var,', new dependent variables:',new_dep_var)
        dep_var=new_dep_var
        if df[dep_var].nunique().sum()==2:
            unique_vals=np.unique(df[dep_var[0]])
            tt=ttest_ind(df[df[dep_var[0]]==unique_vals[0]][tv], df[df[dep_var[0]]==unique_vals[1]][tv], equal_var=False)
            print('\n*******',tv,'\n',tt)      
        else:
            dependents=['C('+dv+')' for dv in dep_var]
            model_statement=' ~ '+'+'.join(dependents)
            print(model_statement)
            model=ols(tv+model_statement,data=df).fit()
            print (tv, sm.stats.anova_lm (model, typ=2), '\n',model.summary())
            if len(dep_var)==1: 
                print('post-hoc',sp.posthoc_ttest(df, val_col=tv, group_col=dep_var[0], p_adjust='holm'))
    