#-*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:53:52 2020

@author: kblackw1
"""
'''    
Initial analysis of parameter variations.  Just looks at mean response
re-run best params (and save data) to show trajectories - calculate moving average of reward and some actions
'''

import numpy as np
import glob
import pandas as pd
import operator
from matplotlib import pyplot as plt
import matplotlib as mpl

pd.set_option("display.max_columns", 20)
pd.set_option('display.width',240)

def read_data(fnames,list_params,numQ=2,blockDA=False):
    df=[]
    max_correct=1
    for f in fnames:
        data=np.load(f,allow_pickle=True)
        results=data['allresults'].item()
        par=results['params'] #extract parameters
        del results['params']
        key_combos=list(results.keys())
        if type(results[key_combos[0]])==dict:
            #if results is dictionary of dictonaries, need to flatten it
            new_results={}
            for phase in key_combos:
                for key,vals in results[phase].items():
                    new_results[phase+'_'+key]=vals
            results=new_results
        #add parameter values to the dictionary
        new_list_params=[];single_params=[]
        for k in par.keys():
            #split state_thresh in two series
            if k in list_params:
                for i in range(numQ):
                    results[k+str(i+1)]=[round(p[i],3) for p in par[k]]
                    new_list_params.append(k+str(i+1))
            else:
                if isinstance(par[k][0],float):
                    results[k]=[round(p,3) for p in par[k]]
                    single_params.append(k)
                elif isinstance(par[k][0],dict): #NEEDS TESTING
                    for xx in par[k][0].keys():
                        results[k+xx]=[p[xx] for p in par[k]]
                else:
                    results[k]=par[k]
                    single_params.append(k)
                #else: do nothing
        #put back in correct alpha
        if blockDA and np.std(results['alpha2'])==0:
            a2inc=np.max(np.diff(results['alpha1']))/2
            a2range=np.arange(np.min(results['alpha1'])/2,1.1*np.max(results['alpha1'])/2,a2inc)
            repeats=int(len(results['alpha2'])/len(a2range))
            results['alpha2']=np.tile(a2range,repeats) 
        ############# stdev
        if 'reslist' in data.keys():
            reslist=data['reslist'].item()
            if type(reslist[key_combos[0]])==dict:
                for phase in key_combos:
                    for key,vals in reslist[phase].items():
                        results[phase+'_'+key+'_STD']=[np.std(par_result) for par_result in vals] 
                key_combos=list(new_results.keys())
            else:
                for col in key_combos: #corresponds to phase/action/epoch combo
                    results[col+'_STD']=[np.std(par_result) for par_result in reslist[col]]
            if 'max_correct' in reslist['params'].keys():
                max_correct=reslist['params']['max_correct']
        dfsubset=pd.DataFrame.from_dict(results,orient='index').transpose()
        #print(f,len(dfsubset))
        #add dictionary of result subset to list of  dictionaries
        df.append(dfsubset)
    allresults=pd.concat(df).reset_index() #concatentate everything into big dictionary
    #drop duplicate values
    param_names=new_list_params+single_params
    newdf=allresults.drop_duplicates(subset=param_names)
    print('size of df, before',len(allresults),'after',len(newdf),'params',param_names, 'max_correct', max_correct)
    return newdf,key_combos,param_names,max_correct

def plot_Q2_results(Q2df,key_combos):
    st=[]
    alph=[]
    st_trans=[]
    for i in range(2):
        st.append( Q2df['state_thresh'+str(i+1)].values  )
        #st1= allresults['params']['state_thresh2']   
        #a0=allresults['params']['alpha1']
        alph.append(Q2df['alpha'+str(i+1)].values) 
    for i in range(2):
        st_trans.append(np.where(np.diff(st[i])>0)[0]+1) #add one because np.diff places transition from 1 to 2 into slot 1, but new values begin in slot 2
        #st1_transition=np.where(np.diff(st1)>0)[0]+1
    #all state threshold transitions locations, including the beginning of the array
    st_transition=sorted(np.concatenate(([0],st_trans[0],st_trans[1])))
    rows=len(np.unique(alph[0]))
    cols=len(np.unique(alph[1]))
    
    #now do the plotting
    figset=[]
    for pa in key_combos:#combos:
        fig,axes=plt.subplots(len(np.unique(st[1])),len(np.unique(st[0])))
        ax=fig.axes
        fig.suptitle(pa+'; numQ='+str(int(Q2df['numQ'].iloc[0])))
        zvals=(Q2df[pa]).to_numpy(dtype=float)
        vmin=np.min(zvals);vmax=np.max(zvals)
        for i,at in enumerate(st_transition):
            #transpose, because a0 has greater range than a1
            if len(np.unique(st[0][at:at+rows*cols]))>1 or len(np.unique(st[1][at:at+rows*cols]))>1:
                print('**************** uh oh ************',at)
            plotz=np.reshape(zvals[at:at+rows*cols],(rows,cols)).T
            ax[i].set_title(str(round(st[0][at],3))+','+str(round(st[1][at],3)),fontsize=8)
            #transpose alpha values and verify that plots are correct
            #use x and y to label x and y axes
            y=np.reshape(alph[1][at:at+rows*cols],(rows,cols)).T[:,0]
            x=np.reshape(alph[0][at:at+rows*cols],(rows,cols)).T[0]
            im=ax[i].imshow(plotz,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],cmap=plt.get_cmap('gray'),vmin=vmin,vmax=vmax,origin='lower')
        for ii in range(np.shape(axes)[0]):
            axes[ii,0].set_ylabel('alpha 2')
        for jj in range(np.shape(axes)[1]):
            axes[-1,jj].set_xlabel('alpha 1')
        cax = fig.add_axes([0.27, 0.9, 0.5, 0.03])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        fig.show()
        figset.append(fig)
    return fig
def plot_Q1_results(Q2df,key_combos):
    x=Q1df['alpha1'].to_numpy(dtype=float)
    y=Q1df['state_thresh1'].to_numpy(dtype=float)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cmap = mpl.cm.gray
    fig,ax=plt.subplots(len(key_combos))
    for i,pa in enumerate(key_combos): #enumerate(combos):
        zvals=(Q1df[pa]).to_numpy(dtype=float)
        vmin=np.min(zvals);vmax=np.max(zvals)
        plotz=np.reshape(zvals,(len(np.unique(y)),len(np.unique(x))))
        ax[i].imshow(plotz,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],cmap=plt.get_cmap('gray'),vmin=vmin,vmax=vmax,origin='lower')
        ax[i].set_xlabel('alpha')
        ax[i].set_ylabel('state_thresh')
        ax[i].set_title(pa)
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="10%", pad=0.1)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=cax, orientation='vertical')#, label=phase)
    fig.show()
    return fig

########################### MAIN ################################################
############# Task specific filenames and events ################################
''' Variation in parameters simulated in December
sequence task reward output was number of times max reward obtained during trial_subset
trial_subset=int(0.05*numevents) = 0.05*10,000 = 500
#therfore, optimal rewards = 500/events_per_trial, which is 83 for HxLen=3 and 71 for HxLen=4
discrim task reward output was mean reward.  Optimal performance is:
    reward of 10 received on every third event, basic effort of -1 on two other events
    thus optimal reward is 8 for trial (per 3 events)
    Since mean was calculated over events (and not multiplied by 3 prior to Jan 3)
    optimal reward per trial is 8/3 prior to Jan 3 and 8 after Jan 3
'''
paradigm= 'sequence'  #'discrim' #'block_Da'# 'block_Da_b6' #'sequence' #'discrim_b6'#
datadir='NormEuclid2021dec10/' # Euclid/discrimRuleNone/  # discrim/ # discrim_beta6/ # Euclid/sequenceRuleNone/ #
min_param_combos=5  #keep lowering the percentiles until this many "best" parameter combinatins have been found
HxLen='4' #only relevant for sequence task
op=operator.gt
optimal_rwd=8 #used for histogram if maximum reward not included if the npz file
block=False #blockDa is false unless otherwise specified

if paradigm == 'sequence':
    Q1end='_Q1_all'#'_Q1*' if HxLen=='4' else '_Q1_all' #this was fur Guassian Mixture model
    file_pattern={'Q1':datadir+'Sequence_paramsHxLen'+HxLen+Q1end,'Q2':datadir+'Sequence_HxLen'+HxLen+'_Q2*q2o0.2*_all'}
    phase_act=[] #if empty, will use key_combos from function
    rwd_column='rwd__End'
else:
    #these are the interesting state-action combinations
    phase_act=['acquire_rwd__End','discrim_rwd__End','reverse_rwd__End',
           'acquire_Pport_6kHz_left_End', 'extinc_Pport_6kHz_left_End','renew_Pport_6kHz_left_Beg','renew_Pport_6kHz_left_End',
           'discrim_Pport_6kHz_left_End','discrim_Pport_10kHz_right_End',
           'reverse_Pport_10kHz_left_End','reverse_Pport_6kHz_right_End']
    rwd_column='tot_rwd'
    rwd_events=['acquire_rwd__End','discrim_rwd__End' ] #first event: what is max observed, 2nd event - more difficult task
    action_events=['acquire_Pport_6kHz_left_End','renew_Pport_6kHz_left_Beg']#first event: what is max observed, 2nd event - more difficult task

#These are updated for some task variants 
if paradigm == 'discrim':
    #discrim task - results stored as action events
    file_pattern={'Q1':datadir+'Discrim2021-12-10_Q1_all','Q2':datadir+'Discrim2021-12-10_Q2_q2o0.1_*'}
   
    '''#uncomment to read earlier data using 1st version of code,
       #saved different state_action combos, used the Gaussian mixture model; Q1 & Q2 learning rules the same
    phase_act=[p+'_End' for p in ['extinc_left','renew_left','acquire_rwd','discrim_rwd', 'reverse_rwd']]+['renew_left_Beg'] # if empty, will use key_combos from function
    #the next 2 events used to find the best results
    rwd_events=['acquire_rwd_End','discrim_rwd_End' ] #first event: what is max observed, 2nd event - more difficult task
    action_events[1]=['acquire_left_End','renew_left_Beg']
    '''   
if paradigm=='block_Da_b6':
    file_pattern={'Q1':'none', 'Q2':'Discrim_beta0.6blockDadip_Q2_*_0'}
    phase_act=['acquire_rwd__End','discrim_rwd__End',
           'acquire_Pport_6kHz_left_End', 'extinc_Pport_6kHz_left_End',
           'discrim_Pport_6kHz_left_End','discrim_Pport_10kHz_right_End']
    action_events[1]='extinc_Pport_6kHz_left_End' # extinc should work with DA_block,
    op=operator.le
    block=True

if paradigm=='block_Da':
    file_pattern={'Q1':'none', 'Q2':datadir+'Discrim_blockDadip_Q2_*_alphaEnd0'}
    phase_act=[p+'_End' for p in ['extinc_left','acquire_rwd','discrim_rwd']]+['extinc_left_Beg'] # if empty, will use key_combos from function
    action_events=['acquire_left_End','extinc_left_End']#first event: what is max observed, 2nd event - more difficult task
    rwd_events=['acquire_rwd_End','discrim_rwd_End' ]
    op=operator.le
    block=True

######## Parameters relevant to all tasks
list_params=['state_thresh','alpha']
sort_pars=['state_thresh1','state_thresh2','alpha1','alpha2']
both_df=[]; both_labels=[];par_names=[]
############################ Analysis for 2 Q matrices ##############
#All simulations didn't run.  Need to re-run some, read in and concatenate several files
#read in all the results

print(file_pattern)
fnamesQ2=glob.glob(file_pattern['Q2']+'.npz')
if len(fnamesQ2):
    Q2df,key_combos,Q2par_names,max_correct=read_data(fnamesQ2,list_params,numQ=2,blockDA=block)
    sort_order=[sp for sp in sort_pars if sp in Q2par_names]
    Q2df.sort_values(sort_order,inplace=True)
    q2mean=Q2df.mean()
    q2std=Q2df.std()
    if rwd_column not in Q2df.columns:
        Q2df[rwd_column]=Q2df[rwd_events].sum(axis=1)    #[max_rwd_event]+ Q2df[rwd_measure2]
    both_df.append(Q2df)
    both_labels.append('Q2')
    par_names.append(sort_order)

    if len(phase_act):
        key_combos=phase_act
    elif paradigm=='sequence':
        key_combos=[pa for pa in key_combos if pa.endswith('End')]
    
    #now create color maps for each state-action, for each set of state transition thresholds
    plot_Q2_results(Q2df, key_combos)
    
####################### Repeat for 1 Q matrix ######################
#read in all the results
fnamesQ1=glob.glob(file_pattern['Q1']+'.npz')
if len(fnamesQ1):
    Q1df,key_combos,Q1par_names,max_correct=read_data(fnamesQ1,list_params)
    sort_order=[sp for sp in sort_pars if sp in Q1par_names]
    Q1df.sort_values(sort_order,inplace=True)
    both_df.append(Q1df)
    both_labels.append('Q1')
    par_names.append(sort_order)

    if len(phase_act):
        key_combos=phase_act
    elif paradigm=='sequence':
        key_combos=[pa for pa in key_combos if pa.endswith('End')]
    #plot
    plot_Q1_results(Q1df, key_combos)
    ##create data series with mean and std
    q1mean=Q1df.mean()
    q1std=Q1df.std()
    if rwd_column not in Q1df.columns:
        Q1df[rwd_column]=Q1df[rwd_events].sum(axis=1)    
######################## Summary over both Q1 and Q2 #####################
if len(fnamesQ1) and len(fnamesQ2):
    summary=pd.DataFrame({'Q1mean':q1mean,'Q1std':q1std,'Q2mean':q2mean,'Q2std':q2std})
elif len(fnamesQ2):
    summary=pd.DataFrame({'Q2mean':q2mean,'Q2std':q2std})
elif len(fnamesQ1):
    summary=pd.DataFrame({'Q1mean':q1mean,'Q1std':q1std})
else:
    print('no files found for Q1 or Q2!!!')

##############  Finish creating summary df
drop_idx=[x for x in summary.index if x.endswith('_STD')]
for idx in drop_idx:
    summary.drop(idx,axis=0, inplace=True)
#print(summary)

############### print results for best parameters #############################

if paradigm=='sequence':# or max_rwd_event=='tot_rwd':
    for label,df,par_names in zip(both_labels,both_df,par_names):
        stdkeys=[kc+'_STD' for kc in key_combos if kc+'_STD' in df.columns]
        found1=0
        for crit in np.arange(0.99,0.8,-0.01):
            rwd_crit=df[rwd_column].quantile(crit)
            best=df.loc[(df[rwd_column] >= rwd_crit)]
            if len(best) and len(best)>found1:
                found1=len(best)
                #/max_correct to convert to percent
                print('*** crit =',round(crit,3),'for',label,'\n',best[par_names],'\n',best[key_combos],'\n',best[stdkeys])
                if len(best)>min_param_combos:
                    break #break out of loop, once best performance found
            else:
                print('**** no records found for', label,'using criteria as alow as', round(crit,3))
else:
    for label,df,par_names in zip(both_labels,both_df,par_names):
        stdkeys=[kc+'_STD' for kc in key_combos if kc+'_STD' in df.columns]
        found1=0
        for crit in np.arange(0.99,0.7,-0.01):
            max_rwd=df[rwd_column].max()
            best_row=df[df[rwd_column]==max_rwd]
            rwd_crit=df[rwd_events[1]].quantile(crit)
            if op==operator.le or op==operator.lt:
                act_crit=df[action_events[1]].quantile(1-crit)
            else:
                act_crit=df[action_events[1]] .quantile(crit)               
            best=df.loc[(df[rwd_events[1]] >= rwd_crit) & op(df[action_events[1]], act_crit)]
            if len(best) and len(best)>found1:
                found1=len(best)
                print('*** crit=',round(crit,3),'on',rwd_events[1],action_events[1],'for',label,'\n',best[par_names],'\n',best[key_combos],'\n',best[stdkeys])
                if len(best)>min_param_combos:
                    break #break out of loop, once best performance found
            else:
                print('**** no records found for', label,'using criteria of ', round(crit,3),rwd_events[1],round(rwd_crit,3),action_events[1],round(act_crit,3))
print(summary.loc[key_combos])

########### Graph showing effect of parameters on reward ##########
#### only for Q1 because too many dimensions for Q2 #### 
rwd_combos=[a for a in key_combos if 'rwd' in a and 'End' in a]
if not paradigm.startswith('block_DA'):               
    fig,axis=plt.subplots(nrows=len(rwd_combos),ncols=1,sharex=True,sharey=True)
    ax=fig.axes
    fig.suptitle(paradigm+' 1Q')
    x=Q1df['alpha1'].to_numpy(dtype=float)
    y=Q1df['state_thresh1'].to_numpy(dtype=float)
    plotx=np.reshape(x,(len(np.unique(y)),len(np.unique(x))))
    ploty=np.reshape(y,(len(np.unique(y)),len(np.unique(x))))
    for jj,pa in enumerate(rwd_combos):
        zvals=(Q1df[pa]).to_numpy(dtype=float)/max_correct
        plotz=np.reshape(zvals,(len(np.unique(y)),len(np.unique(x))))
        for i,row in enumerate(plotz):
            ax[jj].plot(plotx[i],row,label=str(ploty[i][0]))
            ax[jj].set_ylabel(pa[0:7]+', % max)')
        ax[-1].set_xlabel('alpha')
        ax[-1].legend()
#copy plotz into igor?  name columns according to st, create alpha wave

######## histogram (pdf and CDF) showing effect of parameters on reward #######
if max_correct==1:
    binmax=optimal_rwd #or np.max(Q2df[rwd_combos].max())
    add_label=''
else:
    binmax=100
    add_label=' (% max)'
if paradigm.startswith('block_DA'):
    fig,axis=plt.subplots(nrows=1,ncols=2)
    fig.suptitle(paradigm)    
    for pa in rwd_combos:
        hist,bin_edges=np.histogram(df[pa]/max_correct,bins=25,range=(0,binmax))
        plot_bins=[(bin_edges[i]+bin_edges[i+1])/2 for i in range (len(hist))]
        axis[0].plot(plot_bins,hist/np.sum(hist),label=pa[0:7])
        axis[1].plot(plot_bins,np.cumsum(hist/np.sum(hist)),label=pa[0:7])
        axis[0].set_ylabel('pdf')
        axis[0].set_ylabel('CDF')
        for kk in [0,1]:
            axis[kk].set_xlabel(' (fraction of optimal)') 
            axis[kk].legend()    
else:
    save_hist={}
    fig,axis=plt.subplots(nrows=len(rwd_combos),ncols=2)
    ax=fig.axes
    if paradigm=='sequence':
        fig.suptitle(paradigm+', press Hx len='+HxLen)
    else:
        fig.suptitle(paradigm)
    for df,lbl in zip(both_df,both_labels):
        for jj,pa in enumerate(rwd_combos):
            hist,bin_edges=np.histogram(df[pa]/max_correct,bins=25,range=(0,binmax))
            save_hist[lbl]=hist
            plot_bins=[(bin_edges[i]+bin_edges[i+1])/2 for i in range (len(hist))]
            ax[jj*2].plot(plot_bins,hist/np.sum(hist),label=lbl)
            ax[jj*2].set_ylabel('fraction')
            ax[jj*2+1].plot(plot_bins,np.cumsum(hist/np.sum(hist)),label=lbl)
            ax[jj*2+1].set_ylabel('CDF')
            for kk in [0,1]:
                ax_num=jj*2+kk
                ax[ax_num].set_xlabel(pa[0:7]+add_label) 
                ax[ax_num].legend()
    hist_txt=plot_bins
    header='plot_bins'
    for k,v in save_hist.items():
        hist_txt=np.column_stack(([hist_txt,v]))
        header=header+'   '+k
    np.savetxt(paradigm+'_HxLen'+str(HxLen)+'_histogram.txt',hist_txt,header=header,fmt='%.3f')

#mutliply plot_bins (x values) by 100 for percent of optimal reward
