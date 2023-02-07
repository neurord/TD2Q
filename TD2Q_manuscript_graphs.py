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

def create_traject_fig(traject,phases,actions,action_text,norm=None,leg_panel=0, leg_loc='best',leg_fs=fsize,sequential=False,leg_text={},color_dict={}):
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
                    cmap=pnum #for block case
                else:
                    cmap=pnum
                cnum=int(nQ+1)*color_inc+color_offset
                leg_cols=len(phases)
            else:
                color_inc=int((255-color_offset)/(len(data.keys())))
                cmap=nQ
                cnum=int(color_offset+(pnum+1)*color_inc)
                leg_cols=len(traject.keys())
            #print('traject_fit',phase,'Q=',numQ,nQ,'c=',cnum)
            if len(color_dict):
                color=color_dict[numQ][pnum]
                #print('col_dict, color=',color)
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
                    if leg_panel>-1:
                        legend_title='      '.join(['numQ='+str(numQ) for numQ in traject.keys()]) 
                    else:
                        legend_title=''
                    trace_label=trace_label+'Q'
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
                    #if pnum>0: #TEMPORARY - TO ELIMINATE LEGEND
                    #    trace_label='_'
                    if act=='rwd':
                        ax[anum].hlines(0,xmin=0,xmax=np.max(block),linestyle='dotted')
                    if norm is not None and act !='rwd': #### yvalues for reward already in units of "per trial"
                        yvals=data[phase][act]['mean']*norm['value']
                        yerr=data[phase][act]['sterr']*norm['value']
                        ymin[action_text[anum]]=-norm['units']*0.05;ymax[action_text[anum]]=norm['units']*1.05
                    else:
                        yvals=data[phase][act]['mean']
                        yerr=data[phase][act]['sterr']
                    ax[anum].errorbar(block,yvals,yerr=yerr,label= trace_label,color=color)
        maxy=np.max([axis.get_ylim()[1] for axis in ax[1:]] ) #make y axis limits the same for all panels
        miny=np.min([axis.get_ylim()[0] for axis in ax[1:]] )
        for anum in range(len(ax)):
            ax[anum].set_ylabel(action_text[anum], fontsize=fsize+1)
            ax[anum].set_ylim([ymin[action_text[anum]],ymax[action_text[anum]]])
            if action_text[anum]=='reward':
                ylim=ax[anum].get_ylim()
                print('ylim',anum,action_text[anum],ylim)
                ax[anum].set_ylim([np.floor(ylim[0]),np.ceil(ylim[1])*1.05]) 
                yticks=np.linspace(np.floor(ylim[0]),np.ceil(ylim[1]),4)
                ylabels=[str(round(f)) for f in yticks]
                ax[anum].set_yticks(yticks,ylabels)
            elif norm is None:
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
            if leg_panel>-1 and leg_fs>0:
                leg=ax[leg_panel].legend(frameon=True,title=legend_title,ncol=leg_cols,loc=leg_loc,fontsize=leg_fs-1,title_fontsize=leg_fs-1,handletextpad=0.2,labelspacing=0.3,columnspacing=1)# markerfirst=True
    ax[anum].set_xlabel('Block', fontsize=fsize+1)
    return fig #so you can adjust size and then do fig.tight_layout()

def create_sequence_traject_fig(traject,actions,action_text,lenQ={},ept=7,norm=None):
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
            if norm is not None and act !='rwd':
                yvals=data[act]['mean']*norm['value']
                yerr=data[act]['sterr']*norm['value']
            else:
                yvals=data[act]['mean']
                yerr=data[act]['sterr']
            ax[anum].errorbar(block,yvals,yerr=yerr,label= numQ,color=color)
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
    if norm is None: ######################### Revisit after plotting new data
        for anum in range(1,numrows):
            if anum==numrows-1 and len(lenQ):
                yticks=[0,20,40,60,80,100]
            else:
                yticks=[0,5,10]
            ylabels=[str(round(f)) for f in yticks]
            ax[anum].set_yticks(yticks,ylabels)
        ax[0].set_ylim(-7.5,10.0)
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
    return fig #so you can adjust size and then do fig.tight_layout()

def read_data(pattern, files=None, keys=None, dep_var=None):
    ########## only saves one traject for each numQ #########
    if not files:
        files=glob.glob(pattern)
    print('pattern',pattern,'files',files)
    traject={}
    all_counts={}
    traject_dict={}
    for k,f in enumerate(files):
        data=np.load(f,allow_pickle=True)
        params=data['par'].item()
        par=data['results'].item()['params']
        if keys is None and dep_var is None:
            numQ=str(params['numQ'])
        elif dep_var:
            numQ=','.join([str(params[k]) for k in dep_var])
        elif keys:
            numQ=keys[k]
        traject[numQ]=data['traject'].item()
        if 'shift_stay' in data.keys():
            all_counts[numQ]=data['shift_stay'].item()
            traject_dict[numQ]=data['traject_dict'].item()
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
    return traject,all_counts,sa_error_keys,par,traject_dict
  
################## Stat Analysis  #########################
def create_df(pattern,files=None,del_variables=[],params=['numQ'],keys=None ):
    if not files:
        files=glob.glob(pattern)
    print('pattern',pattern,'files',files)
    df=[]
    for fnum,f in enumerate(files):
        par={}
        data=np.load(f,allow_pickle=True)
        print(f,list(data.keys()))
        results=data['results'].item()
        print('params',results['params'])
        for p in params:
            if p in results['params'].keys():
                par[p]=results['params'][p][0]
                if not par[p]:
                    par[p]=-1
            elif keys:
                par[p]=keys[fnum]
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
                if '-' in phs:
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

def barplot(mean,sterr,variables,varname,ylabel='Reward',transpose=False,norm=None,legend=True):
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
        ############### Normalize to % of optimal per trial, except reward already normalized to per trial ###########
        lbl='Ctrl' if tv==-1 else tv
        if norm and ('rwd' not in lbl and 'reward' not in lbl): 
            yvalues*=norm['value']
            yerr*=norm['value']
        ax.bar(xvalues+(a-(len(variables)-1)/2)*w, yvalues,width=w,yerr=yerr,label=str(lbl)[0:8]) 
    ax.set_ylabel(ylabel)
    ax.hlines(0,np.min(xvalues),np.max(xvalues),linestyles='dashed',colors='gray')
    ax.set_xlabel('Condition')
    ax.set_xticks(xvalues,xlabels)
    if ('rwd' not in lbl and 'reward' not in lbl and norm ):
        ax.set_ylim([0,1.2*norm['units']])
    if legend:
        ax.legend(loc='best',fontsize=9,frameon=False)
    figS.suptitle('+'.join([str(v) for v in variables]),fontsize=8)
    return figS

def barplot_means(df,dep_var,test_variables):
    print(df.groupby(dep_var)[test_variables].aggregate(['mean','sem'])) #,'count'
    mean=df.groupby(dep_var)[test_variables].mean()
    cnt=df.groupby(dep_var)[test_variables].count()
    sterr=df.groupby(dep_var)[test_variables].std()/np.sqrt(cnt-1)
    return mean,cnt,sterr

def calc_norm(params,percent=False):
    norm={}
    if percent:
        norm['value']=100/params['trials_per_block'][0]
        norm['units']=100
    else:
        norm['value']=1/params['trials_per_block'][0]
        norm['units']=1
    return norm

if __name__ == "__main__":
    #for stats, only run one at a time
    task= 'bandit'#'discrim' # 'block_da'#'sequence' # 
    add_barplot=0 #only relevant for sequence
    shift_stay=0 #only relevant for bandit
    test_var=[]
    traject_fig=False
    ######################### DISCRIM #########################
    if task=='discrim':
        from discrimFiles import pattern,dep_var,files,test_variables,actions,action_text,keys
        traject,_,_,params,_=read_data(pattern,files=files,keys=keys) 
        norm=calc_norm(params,percent=False)
        ##### acquisition ######
        phase=['acquire']
        if traject_fig:
            figA=create_traject_fig(traject,phase,actions,action_text,norm=norm) #Fig 2A,B
                
        ##### extinction, renewal #####
        phase=['acquire','extinc','renew'] #phase=['extinc','renew']
        #phase_text={'extinc':'Context B','renew':'Context A'}
        phase_text={'acquire':'','extinc':'','renew':''}
        if traject_fig:
            #figE=create_traject_fig(traject,phase,actions,action_text,leg_text=phase_text,norm=norm) #Fig 2C
            figE=create_traject_fig(traject,phase,actions,action_text,leg_panel=-1,sequential=True,norm=norm) #Fig 2C
        
        ##### discrimination, reversal #####
        phase=['acquire','discrim','reverse']
        actions=['rwd', (('Pport', '10kHz'),'right'),(('Pport', '10kHz'),'left')]#,(('Pport', '6kHz'),'left')]
        action_text=['reward', '10 kHz Right', '10 kHz Left']#,'6 kHz Left'] #
        if traject_fig:
            figD=create_traject_fig(traject,phase,actions,action_text,leg_panel=-1, sequential=True,norm=norm) #Fig 4
            xlim=[19,61]
            ax=figD.axes
            for axis in ax:
                axis.set_xlim(xlim)

    ######################### block Dopamine #########################
    elif task=='block_da':
        from discrimFiles import pattern,dep_var,files,test_variables,actions,action_text,keys
        print(actions)
        traject,_,_,params,_=read_data(pattern, files, keys) 
        norm=calc_norm(params)
        phase=['acquire','discrim']
        ###fix colors here
        figB=create_traject_fig(traject,phase,actions,action_text,leg_panel=1,leg_loc='upper left', leg_fs=12, sequential=True,norm=norm) #Fig 6
        ax=figB.axes
        #ax[0].set_ylim([0,11])
        #ax[1].set_ylim([-0.3,8]) 
        #ax[2].set_ylim([-0.3,8]) 
    #################### Sequence trajectory ##################
    elif task=='sequence':
        from sequenceFiles import pattern, dep_var, files,barplot_files,keys,test_variables,actions,action_text
        if add_barplot:
            files=barplot_files
            traject,all_lenQ,error_keys,params,_=read_data(pattern,barplot_files,keys=keys)   
        else:
            traject,all_lenQ,error_keys,params,_=read_data(pattern,files,dep_var=dep_var) 
        norm=calc_norm(params)
        if traject_fig:
            if len(traject)==2:
                figS=create_sequence_traject_fig(traject,actions,action_text,lenQ=all_lenQ) #fig 7
            else:
                figS=create_sequence_traject_fig(traject,actions,action_text) #fig 7
        ######## Currently not used ###########
        switch=[kk.replace('*','x')+'_End' for kk in error_keys['switch'].values() if 'press' in kk ]
        premature=[kk+'_End' for kk in ['Llever_xxRL_goR', 'Llever_---L_goR','Rlever_xxRL_press','Rlever_---L_press','Rlever_xLLR_goMag']]
        overstay=[kk.replace('*','x')+'_End' for kk in error_keys['overstay'].values() if 'press' in kk]
        start=[kk.replace('*','x')+'_End' for kk in error_keys['start'].values()]

    #################### Bandit Task Probabilities ##################
    elif task=='bandit':
        from banditFiles import pattern,dep_var,files,test_variables,actions,action_text,keys
        from BanditTask import calc_fraction_left,plot_prob_tracking,plot_prob_traject,perseverance
        runs=40
        colors2=[plt.get_cmap('seismic').__call__(c) for c in [18,50]]+[plt.get_cmap('hsv').__call__(c) for c in [188,204,225]]+[plt.get_cmap('seismic').__call__(c) for c in [192,242]]
        newcol={'2':colors2,'1':colors2}
        traject,shift_stay,_,params,traject_dict=read_data(pattern,files=files,keys=keys,dep_var=dep_var)
        shift_stay=False
        norm=calc_norm(params)
        p_choose_L={q:{} for q in traject.keys()}
        RMS={}
        if len(traject.keys())<=2:
            for numQ, data in traject.items():
                fractionLeft,noL,noR,ratio=calc_fraction_left(traject_dict[numQ],runs)
                popt,pcov,delta,RMSmean,RMSstd,RMS[numQ]=plot_prob_tracking(ratio,fractionLeft,runs,showplot=False)
                print('numQ=',numQ, 'ratio:',{round(ratio[k],3): round(np.nanmean(fractionLeft[k]),3) for k in fractionLeft.keys()} )
                p_choose_L[numQ]=plot_prob_traject(data,params,show_plot=False)
            figB=create_bandit_fig(p_choose_L,numpanels=2,color_dict=newcol) 
            if traject_fig:
                tkeys=list(traject.keys())
                tmp_phs=list(traject[tkeys[0]].keys())
                phases=sorted(tmp_phs,key=lambda tmp_phs: float(tmp_phs.split(':')[0])-float(tmp_phs.split(':')[1]),reverse=True)
                figBT=create_traject_fig(traject,phases,actions,action_text,leg_fs=0,color_dict=newcol,norm=norm) #Fig 10C,D
        if shift_stay:
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

############################# Stat analysis ##########################################
    import pandas as pd
    from scipy.stats import ttest_ind
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import scikit_posthocs as sp
    import os

    df=create_df(pattern,files=files,params=dep_var)
    if task=='block_da':
        df=create_df(pattern,files=files,params=dep_var,keys=keys)
    if 'Bandit' in pattern:
        key1=list(traject.keys())[0]
        ratio={}
        df['sum_squares']=0
        for prob in traject[key1].keys():
            R=float(prob.split(':')[1])
            L=float(prob.split(':')[0])
            ratio[prob]=L/(L+R)
            ###### probability of Left for entire phase #######
            df[prob+'_probL']=df[prob+'_Pport_6kHz_left_End']/(df[prob+'_Pport_6kHz_left_End']+df[prob+'_Pport_6kHz_right_End'])
            df['sum_squares']+=np.square(ratio[prob]-df[prob+'_probL'])
        df['RMS']=np.sqrt(df['sum_squares'])
        df['mean_reward']=df.loc[:,test_variables].mean(axis=1)
        test_variables=['mean_reward','RMS']
    if os.path.basename(pattern).upper().startswith('DISCRIM') and task != 'block_da':
        df['mean_reward']=df.loc[:,['reverse_rwd__End','discrim_rwd__End']].mean(axis=1)
        test_variables=['mean_reward','acquire_rwd__End','reverse_rwd__End','discrim_rwd__End']
        if 'split' in dep_var:
            test_variables=['mean_reward','extinc_Pport_6kHz_left_End','extinc_Pport_6kHz_left_Beg','renew_Pport_6kHz_left_End','renew_Pport_6kHz_left_Beg']
    mean,cnt,sterr=barplot_means(df,dep_var,test_variables)
    
    textname=pattern.replace('?','x').replace('*','all_').split('.npz')[0]+'summary.txt'
    columns=[tv[0:-4] for tv in test_variables]
    header=' '.join(cnt.index.names)+' '+'  '.join(columns)
    rows=list(cnt.index.values)
    figBP=barplot(mean,sterr,test_variables,dep_var[0],norm=norm)
    np.savetxt(textname,np.column_stack((rows*3,np.vstack([cnt,round(mean,3),round(sterr,3)]))),fmt='%s',header=header,comments='') #Ttest tables/Fig 5 - Igor
    if task=='bandit':
        newtv= [k+'_probL' for k in ['90:10','90:50','50:10', '50:50','10:50','50:90','10:90']] 
        print(df.groupby(dep_var)[newtv].mean())
        if 'numQ' in dep_var:
            print('******** Ttest on entire trial RMS, numQ:', ttest_ind(RMS['1'],RMS['2'],equal_var=False))
    if task=='sequence' and add_barplot:
        df['Llever_1L_goR_End']=df['Llever_xxRL_goR_End']+df['Llever_sssL_goR_End']
        if 'Rlever_xxRL_press_End' in df.columns:
            df['Rlever_1L_press_End']=df['Rlever_xxRL_press_End']+df['Rlever_sssL_press_End']
        df['Llever_1L_press_End']=df['Llever_xxRL_press_End']+df['Llever_sssL_press_End']
        df['mag_ssss_nostart_End']=df['mag_ssss_goR_End']+df['mag_ssss_other_End']+df['mag_ssss_goMag_End']+df['mag_ssss_press_End']
        df['Llever_1L_press_End_Prob']=(df['Llever_1L_press_End'])/(df['Llever_1L_press_End']+df['Llever_sssL_goR_End']+df['Llever_sssL_goMag_End']+df['Llever_sssL_other_End']+df['Llever_xxRL_other_End']+df['Llever_xxRL_goMag_End']+df['Llever_xxRL_goR_End'])
        df['Llever_2L_switch_End_Prob']=df['Llever_xxLL_goR_End']/(df['Llever_xxLL_goR_End']+df['Llever_xxLL_press_End']+df['Llever_xxLL_goL_End']+df['Llever_xxLL_goMag_End']+df['Llever_xxLL_other_End'])
        df['mag_sss_start_End_Prob']=df['mag_ssss_goL_End']/(df['mag_ssss_nostart_End']+df['mag_ssss_goL_End'])
        all_test_variables={'overstay':['Llever_xxLL_press_End','Llever_xxLL_goL_End'], #correct vs over-stay
                        #'B':['Rlever_LLRR_goMag_End','Rlever_LLRR_press_End','Rlever_LLRR_goR_End'], #correct vs stay
                        #'C':['Rlever_xLLR_press_End','Rlever_xLLR_goL_End'], #correct vs switch
                        'premature':['Llever_1L_goR_End','Rlever_1L_press_End'],
                        #'start':['mag_ssss_goL_End', 'mag_ssss_nostart_End', ],# incorrect start
                        'all_three':['mag_ssss_goL_End','Llever_1L_press_End','Llever_xxLL_goR_End'],
                        'prob':['mag_sss_start_End_Prob','Llever_1L_press_End_Prob','Llever_2L_switch_End_Prob']} 
        for kk,test_var in all_test_variables.items():
            mean,cnt,sterr=barplot_means(df,dep_var,test_var)
            #figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Responses per Trial')
            if kk=='prob':
                figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Probability',transpose=True,legend=False)
            else:
                figBP=barplot(mean,sterr,test_var,dep_var[0],ylabel='Responses per Trial',transpose=True,legend=False)#,norm=norm)
    for tv in test_variables+test_var: # all_test_variables['all_three']+['rwd__End']: #
        if df[tv].isna().sum():
            testdf=df.dropna()
            print('new mean (after dropping Nans) for',tv, testdf.groupby(dep_var)[tv].aggregate(['mean','std','count']))
        else:
            testdf=df
        new_dep_var=[]
        for dv in dep_var:
            if testdf[dv].nunique()>1:
                new_dep_var.append(dv)
            else:
                print('STATS: only 1 level for variable=', dv)
        if len(dep_var)>len(new_dep_var):
            print('proposed dependent variables:',dep_var,', new dependent variables:',new_dep_var)
        dep_var=new_dep_var
        if testdf[dep_var].nunique().sum()==2:
            unique_vals=np.unique(testdf[dep_var[0]])
            tt=ttest_ind(testdf[testdf[dep_var[0]]==unique_vals[0]][tv], testdf[testdf[dep_var[0]]==unique_vals[1]][tv], equal_var=False)
            print('\n*******',tv,'\n',tt)      
        else:
            dependents=['C('+dv+')' for dv in dep_var]
            model_statement=' ~ '+'+'.join(dependents)
            print('\n****************************************\n',tv, '=',model_statement,'\n')
            model=ols(tv+model_statement,data=testdf).fit()
            print (sm.stats.anova_lm (model, typ=2), '\n',model.summary())
            if len(dep_var)==1: 
                print('\npost-hoc\n',tv,sp.posthoc_ttest(testdf, val_col=tv, group_col=dep_var[0], p_adjust='holm'))
    