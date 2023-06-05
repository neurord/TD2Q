# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:44:25 2021

@author: kblackw1
"""
import numpy as np

import matplotlib.pyplot as plt
import copy
plt.ion()

# possibly accumulate Qhx, as done in Sequence task - 
#       separate out this function into accumulating and plotting
#       CALL THE qhx accum function in the for r in runs loop
# possibly save Qhx, and read in for the plot
import string
letters=string.ascii_uppercase
blank=0.03
fsize=12
fsizeSml=10 

def Qhx_multiphase(states,actions,agents,numQ):
    #find the state number corresponding to states for each learning phase
    state_digits=1
    state_nums={q:{state:[] for state in states} for q in range(numQ)}
    ideal_states={q:{} for q in range(numQ)}
    num_states=1
    for rl in agents:        
        for q in range(numQ):
            for state in states:
                for stnum,st in rl.agent.ideal_states[q].items():
                    if int(round(st[0]))==rl.env.states['loc'][state[0]] and int(round(st[1]))==rl.env.states['tone'][state[1]]:
                        state_nums[q][state].append(stnum)
                state_nums[q][state]= list(np.unique(state_nums[q][state]))
                ideal_states[q][state]=[[str(round(a,state_digits)) for a in rl.agent.ideal_states[q][stnum]] for stnum in state_nums[q][state]]
                num_states=max(num_states,len(state_nums[q][state]))
    #concatenate Q value history across phases for above states and actions
    #Qhx={q:{state:{ac:[[] for n in range(num_states)] for ac in actions} for state in state_nums[q].keys()} for q in state_nums.keys()}
    Qhx={q:{state:{ac:[[] for n in range(len(state_nums[q][state]))] for ac in actions} for state in state_nums[q].keys()} for q in state_nums.keys()}
    #boundary stores x values for drawing phase boundaries
    boundary={q:{state:[0] for state in state_nums[q].keys()} for q in Qhx.keys()}
    #
    for q in Qhx.keys():
        for rl in agents:
            for state,stnums in state_nums[q].items():
                boundary[q][state].append(boundary[q][state][-1]+len(rl.agent.Qhx[q])/rl.agent.events_per_trial)
                for ac,arr in Qhx[q][state].items():
                    for arr_num,stn in enumerate(stnums):
                        if stn<np.shape(rl.agent.Qhx[q])[1]:
                            if len(Qhx[q][state][ac][arr_num]):
                                Qhx[q][state][ac][arr_num]=np.concatenate((Qhx[q][state][ac][arr_num],rl.agent.Qhx[q][:,stn,rl.agent.actions[ac]]))
                            else:
                                Qhx[q][state][ac][arr_num]=rl.agent.Qhx[q][:,stn,rl.agent.actions[ac]]
                        else:
                            #add array of zeros if a state not represented in Q matrix
                            if len(Qhx[q][state][ac][arr_num]):
                                 Qhx[q][state][ac][arr_num]=np.concatenate((Qhx[q][state][ac][arr_num],np.zeros(len(rl.agent.Qhx[q]))))
                            else:
                                 Qhx[q][state][ac][arr_num]=np.zeros(len(rl.agent.Qhx[q]))  
    return Qhx, boundary,ideal_states

def plot_Qhx_2D(Qhx,boundary,ept,phases,ideal_states=None,fig=None,ax=None,title=''):
    ######### plot Qhx for bandit and Discrim task #############
    from matplotlib import pyplot as plt
    colors=[plt.get_cmap('Blues'),plt.get_cmap('Reds'),plt.get_cmap('Purples'),plt.get_cmap('Greys')]
    if fig is None:
        fig,axis=plt.subplots(len(Qhx[list(Qhx.keys())[0]]),len(Qhx),sharex=True)
        ax=fig.axes
        #numQ=len(Qhx)
        #title_prefix=str(numQ)+'Q, '
        title_prefix=''
    else:
        title_prefix=''
    fig.suptitle(title)
    for col,q in enumerate(Qhx.keys()):
        if isinstance(q,int):
            Qname={0:'G',1:'N'}
            ax[col].set_title(title_prefix+Qname[q]+' values',fontsize=fsize-2)
            #ax[col].set_title(title_prefix+'Q'+str(q+1)+' values',fontsize=fsize-2)
        label_inc=1/len(Qhx[q].keys()) #used for putting subplot labels
        for row,state in enumerate(Qhx[q].keys()): 
            axnum=col+row*len(Qhx)
            print('ax',axnum,'Q',q,'state',state,'row',row)
            for cnum,ac in enumerate(Qhx[q][state].keys()):
                col_inc=colors[cnum].N*(2/3)/len(Qhx[q][state][ac])
                for arr_num,arr in enumerate((Qhx[q][state][ac])):
                    color=colors[cnum%len(colors)].reversed().__call__(int(arr_num*col_inc))
                    if ideal_states is not None and len(Qhx[q][state][ac])>1:
                        #label=' '.join(ideal_states[q][state][arr_num]
                        #label='context='+ideal_states[q][state][arr_num][-1]
                        label=int(float(ideal_states[q][state][arr_num][-1])) #for Qhx figure in manuscript
                        leg_cols=2
                    else:
                        label=''
                        leg_cols=1
                    Xvals=np.arange(len(arr))/ept
                    ax[axnum].plot(Xvals,arr,label=ac+' '+ str(label),color=color)
            #Next 3 lines are just for block case.
            startx=0#160#
            endx=Xvals[-1]
            ax[axnum].set_xlim(startx,endx+(endx-startx)*0.05)
            handles,labels=ax[axnum].get_legend_handles_labels()
            if leg_cols==2:
                newlabels=[letters[int(lbl.split()[-1])] for lbl in labels]#for Qhx figure in manuscript
                leg_ttl=list(np.unique([lbl.split()[0] for lbl in labels]))
                leg_loc='center right'#'best'#
                ax[axnum].legend(handles,newlabels,loc=leg_loc,ncol=leg_cols,title='     '.join(leg_ttl),fontsize=fsizeSml,title_fontsize=fsizeSml,handletextpad=0.2,labelspacing=0.3,columnspacing=1)
            else:
                if len(ax) > 1:
                    leg_loc='lower left'
                    ax[1].legend(loc='lower left',ncol=leg_cols,fontsize=fsizeSml)
                    ax[0].legend(loc='upper right',ncol=leg_cols,fontsize=fsizeSml)
            if isinstance(state,tuple) or isinstance(state,list):
                ax[axnum].set_ylabel(','.join(list(state)),fontsize=fsizeSml+1)
            elif isinstance(state,str):
                ax[axnum].set_ylabel(state.split()[0],fontsize=fsizeSml+1)
            elif isinstance(q,str):#when re-arranging dictionary, then q has state and state has q val
                ax[axnum].set_ylabel('Q'+str(state+1)+' values',fontsize=fsizeSml+1)
            if row==len(Qhx[q].keys())-1:
                ax[axnum].set_xlabel('Trial',fontsize=fsizeSml+1)
            ylim=ax[axnum].get_ylim()
            maxQ=max(ylim)
            minQ=min(ylim)
            Qrange=maxQ-minQ
            ax[axnum].set_ylim([round(minQ-0.1*Qrange),round(maxQ+0.2*Qrange)]) #inset the curve to make room for text
            ax[axnum].tick_params(axis='both', which='major', labelsize=fsizeSml)
            for jj,xval in enumerate(boundary[q][state][1:]):
                textx=(xval+boundary[q][state][jj])/2 #ax[axnum].transData
                textx=0.5*(xval+boundary[q][state][jj]-startx)/boundary[q][state][-1] #transAxes
                if isinstance(phases[0],str):
                    phs=phases[jj]
                elif isinstance(phases[0],list) and isinstance(state,str):
                    if state.split()[-1].isdigit():
                        phs=phases[int(state.split()[-1])][jj]
                elif isinstance(phases[0],list) and isinstance(q,str): #when re-arranging dictionary, then q has state and state has q val
                    if q.split()[-1].isdigit():
                        phs=phases[int(q.split()[-1])][jj]
                #ax[axnum].text(textx,1*round(maxQ+0.05*Qrange),phs,ha='center',transform=ax[axnum].transData)
                #ax[axnum].text(textx,0.9,phs,ha='center',transform=ax[axnum].transAxes,fontsize=fsizeSml)
                ax[axnum].vlines(xval,round(minQ),round(maxQ),linestyles='dashed',color='grey')
            y=(1-blank)-(row*label_inc) #subtract because 0 is at bottom
            #if len(Qhx[q].keys())>1:
            #    fig.text(0.02,y,letters[row], fontsize=fsize)
    return fig
#normalize to optimal?  But what is optimal here?  
def agent_response(runs,random_order,num_blocks,traject_dict,fig=None,ax=None,norm=1):
    for rr,r in enumerate(runs):
        if rr>0 or (fig is None):
            fig,ax=plt.subplots()
            fig.suptitle('agent '+str(r))
        left=np.zeros(len(random_order[r])*num_blocks)
        right=np.zeros(len(random_order[r])*num_blocks)
        for k,key in enumerate(random_order[r]):
            start=k*num_blocks;end=(k+1)*num_blocks
            left[start:end]=np.array(traject_dict[key][(('Pport', '6kHz'), 'left')][r])*norm
            right[start:end]=np.array(traject_dict[key][(('Pport', '6kHz'), 'right')][r])*norm
        #now plot the single trials            
        ax.plot(left,'blue',label='left')
        ax.plot(right,'red',label='right')
        ax.set_ylabel('Response Rate')
        ax.set_xlabel('Block')
        ylim=ax.get_ylim()
        for k,key in enumerate(random_order[r]):
            ax.text((k+0.5)*num_blocks,ylim[1],key,ha='center',transform=ax.transData)
            ax.vlines(k*num_blocks,0,10.1,linestyles='dashed')
        ax.set_xlim([0,len(left)])        
        ax.set_ylim([ylim[0],ylim[1]*1.1])
        ax.legend()#loc='center')
    return fig

def plot_Qhx_sequence(Qhx,actions,ept,numQ):                                                        
    ### some states are practically zero, delete these from Qhx to be plotted
    import copy
    newQhx=copy.deepcopy(Qhx)
    for state in Qhx.keys(): 
        minQboth={row:[] for row in Qhx[state][0]};maxQboth={row:[] for row in Qhx[state][0]}
        for col,q in enumerate(Qhx[state].keys()):
            for press_hx in Qhx[state][q].keys():
                maxQboth[press_hx].append(np.floor(np.max([np.max(arr) for arr in Qhx[state][q][press_hx].values()])))
                minQboth[press_hx].append(np.ceil(np.min([np.min(arr) for arr in Qhx[state][q][press_hx].values()])))
        deleterow=[]
        for row in maxQboth.keys():
            for q in range(len(maxQboth[row])):
                if maxQboth[row][q]==minQboth[row][q]:
                    deleterow.append(row)
        print('for state=',state,', not plotting these press histories:',np.unique(deleterow))
        for row in np.unique(deleterow):
            for q in newQhx[state].keys():
                del newQhx[state][q][row] 
    #
    ######### Now create the plot #############            
    figures={}
    for state in newQhx.keys(): 
        fig,axis=plt.subplots(len(newQhx[state][0]),len(newQhx[state]),sharex=True)
        figures[state]=fig
        ax=fig.axes
        for col,q in enumerate(newQhx[state].keys()):
            ax[col].set_title(str(numQ)+'Q, Q'+str(q+1)+' values')
            for row, press_hx in enumerate(newQhx[state][q]):
                axnum=col+row*len(newQhx[state])
                maxQ=round(np.max([np.max(arr) for arr in newQhx[state][q][press_hx].values()]),1)
                minQ=round(np.min([np.min(arr) for arr in newQhx[state][q][press_hx].values()]),1)
                for ac,color in actions.items():
                    #Average Q values across runs
                    Yvals=np.mean(newQhx[state][q][press_hx][ac],axis=0)
                    #trial number = event/events_per_trial (ept)
                    Xvals=np.arange(len(Yvals))/ept
                    ax[axnum].plot(Xvals,Yvals,label=ac,color=color)
                if col==0:
                    ax[axnum].set_ylabel(press_hx)
                ax[axnum].set_ylim([minQ,maxQ])
            ax[axnum].legend()
            ax[axnum].set_xlabel('Trial')
    plt.show()
    return figures
def replace_star(states):
    new_states=[]
    for state in states:
        star_index=[i for i, letter in enumerate(state[1]) if letter =='*']
        if len(star_index)==1:
            for symb in ['L','R','-']:
                new_states.append((state[0],state[1].replace('*',symb)))
        elif len(star_index)==2:
            newst=[]
            for symb in ['L','R','-']:
                newst.append(state[1].replace('*',symb,1))
            for st in newst:
                for symb in ['L','R']:                             
                    new_states.append((state[0],st.replace('*',symb)))
    return new_states

def plot_Qhx_sequence_1fig(allQhx,plot_states,actions_colors,ept,actions_lines):                                                        
    numcols=3  ######## change this to numQ if 1 figure per numQ
    plot_state_trunc=[state[0][0:3]+','+state[1] for state in plot_states]
    star_states=[ps for ps in plot_states if '*' in ps[1]]
    if len(star_states):
        new_states=replace_star(star_states)
        plot_states=plot_states+new_states
        for st in star_states:
            plot_states.remove(st)
        print('********NEW',plot_states)
    plot_state_string=[','.join(list(ps)) for ps in plot_states]
    fig,axis=plt.subplots(len(plot_states),numcols,sharex=True)
    ax=fig.axes
    for numQ,Qhx in allQhx.items(): ####### Remove for 1 figure per numQ
        for state in Qhx.keys():
            for q in Qhx[state].keys(): #q 1 if nq=0, q is either 1 or 2 ir nq=1
                col=int(numQ)-1+int(q)  ####### col = enumerate over Qhx[state] if 1 figure per numQ
                for press_hx in Qhx[state][q].keys():
                    maxQ=0
                    minQ=0
                    found=False
                    if press_hx in plot_state_trunc:
                        row=plot_state_trunc.index(press_hx)
                        found=True
                    elif press_hx in plot_state_string:
                        row=[','.join(list(ps)) for ps in plot_states].index(press_hx)
                        found=True
                    if found:
                        axnum=row*numcols+col
                        print(numQ,q,row,col,axnum,state,press_hx)
                        if row==0:
                            Qtype={'10':'Q','11': 'Q2', '20':'G', '21':'N'} #Q2: if plotting more than 
                            ax[axnum].set_title(str(numQ)+'Q, '+Qtype[str(numQ)+str(q)]+' values') #replace Q'+str(q+1)+
                        maxQ=round(np.max([np.max(arr) for arr in Qhx[state][q][press_hx].values()]),1) #comment out when Q1 or Q2 inactivated
                        minQ=round(np.min([np.min(arr) for arr in Qhx[state][q][press_hx].values()]),1)
                        for ac,color in actions_colors.items():
                            #Average Q values across runs
                            Yvals=np.mean(Qhx[state][q][press_hx][ac],axis=0)
                            maxQ=max(maxQ,np.max(Yvals))
                            minQ=min(minQ,np.min(Yvals))
                            #trial number = event/events_per_trial (ept)
                            Xvals=np.arange(len(Yvals))/ept
                            if ac != 'goMag':
                                ax[axnum].plot(Xvals,Yvals,label=ac,color=color, linestyle=actions_lines[ac])
                        if ('1' in allQhx.keys() and col==0) or (col==1 and '1' not in allQhx.keys()):
                            ax[axnum].set_ylabel(','.join(list(plot_states[row])),fontsize=11)
                        if minQ<maxQ:
                            ax[axnum].set_ylim([minQ,maxQ])
                        if row==len(plot_states)-1:
                            ax[axnum].set_xlabel('Trial',fontsize=12)
                    #else: print('************ not found ********',numQ,state,q,press_hx)
    axis[0][1].legend(loc='upper left')
    for col in range(numcols):
        ylim=[axe.get_ylim() for axe in ax[col::3]]
        ymin=np.floor(np.min([a[0] for a in ylim]))*1.1
        ymax=np.ceil(np.max([a[1] for a in ylim]))*1.1
        for axe in ax[col::3]:
            if ymax>ymin:
                axe.set_ylim([ymin,ymax])
    plt.show()
    return fig

def staticQ_barplot(Q,actions,title='',labels=None,state_subset=None):
    """Visualize the Q table by bar plot"""
    fig,axis=plt.subplots(len(Q),1,sharex=True)
    axes=fig.axes
    #fig.suptitle(title)
    Na=len(actions)
    colors=plt.get_cmap('inferno') #plasma, viridis, inferno or magma
    color_increment=int((len(colors.colors)-40)/(Na-1)) #40 to avoid to light colors
    for i in Q.keys():
        newQ=[];statenums=[];xlabels=[]
        for s,row in enumerate(Q[i]):
            if np.any(row):
                newQ.append(row)
                statenums.append(s)
                if labels is not None:
                    xlabels.append(labels[i][s])
        if len(xlabels) and state_subset is not None:
            Qsubset=[]
            keep_state=[(ii,lbl)  for ii,lbl in enumerate(xlabels) for ss in state_subset if ss in lbl]
            for (j,lbl) in keep_state:
                Qsubset.append(newQ[j])
            plotQ=np.array(Qsubset)
            xlabels=[ks[1] for ks in keep_state]
            statenums=[ks[0] for ks in keep_state]
        else:
            plotQ=np.array(newQ)
        w = 1./(Na+0.5) # bar width
        for a in range(Na):
            cnum=a*color_increment
            axes[i].bar(np.arange(len(statenums))+(a-(Na-1)/2)*w, plotQ[:,a], w,color=colors.colors[cnum])  
        if labels is not None:
            xticks=[' '.join(lbl[0:2]) for lbl in xlabels]
        else:
            xticks=statenums
        axes[i].set_ylabel("Q"+str(i+1)+" value")
        axes[i].set_xticks(range(len(plotQ)),xticks)
    #axes[-1].set_xlabel("state")
    fig.legend(list(actions.keys()),bbox_to_anchor = (0.98, 0.98),ncol=2)#loc='upper right')
    for i in Q.keys():
        for ll in range(len(plotQ)-1):
            ylim=axes[i].get_ylim()
            axes[i].vlines(ll+0.5,ylim[0],ylim[1],'grey',linestyles='dashed')    #make vertical grid - between groups of bars
            label_inc=(1-blank)/len(Q) 
            x=1-blank-(i*label_inc) #subtract because 0 is at bottom
            fig.text(blank,x,letters[i], fontsize=fsize)
    return fig

def combined_bandit_Qhx_response(random_order,num_blocks,traject_dict,Qhx,boundaries,ept,phases,agent_num=-1,all_beta=[],Qlen=[],norm=1):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    
    fig=plt.figure()
    #gs=GridSpec(2,2) # 2 rows, 2 columns
    #ax=[]
    #ax.append(fig.add_subplot(gs[0,:])) # First row, span all columns
    #ax.append(fig.add_subplot(gs[1,0])) # 2nd row, 1st column
    #ax.append(fig.add_subplot(gs[1,1])) # 2nd row, 2nd column

    if len(Qlen):
        numrows=5
    else:
        numrows=4
    gs=GridSpec(numrows,2)
    ax=[]
    for row in range(numrows):
        ax.append(fig.add_subplot(gs[row,:]))
    
    agent_response([agent_num],random_order,num_blocks,traject_dict,fig,ax[0],norm=norm)
    fig=plot_Qhx_2D(Qhx,boundaries,ept,phases,fig=fig,ax=[ax[1],ax[2]]) 
    Xvals=np.arange(len(all_beta[agent_num]))/ept
    ax[3].plot(Xvals,all_beta[agent_num])
    ax[3].set_ylabel(r'$\beta$1')
    if len(Qlen):
        for q,data in Qlen[agent_num].items():
            ax[4].plot(Xvals,data,label='Q'+str(q+1))
        ax[4].legend()
        ax[4].set_ylabel('States')
    ax[-1].set_xlabel('Trial')
    ##### add boundaries ####
    add_boundaries(numrows,Qhx,boundaries,ax)

    #add subplot labels 
    fsize=14
    blank=0.03
    label_inc=1/len(fig.axes)
    for row in range(len(fig.axes)):
        y=(1-blank)-(row*label_inc) #subtract because 0 is at bottom
        fig.text(0.02,y,letters[row], fontsize=fsize)
    fig.tight_layout()
    return fig

def staticQ(f,figs,nQ):
    data=np.load('staticQ'+f+'.npz',allow_pickle=True)
    Q=data['allQ'].item()
    labels=data['labels'].item()
    state_subset=list(data['state_subset'])
    actions=data['actions'].item()
    figs[nQ]['static']=staticQ_barplot(Q,actions,title=str(nQ)+'Q',labels=labels,stabetate_subset=state_subset)
    return figs

def add_labels(fig,numcols):
    axes=fig.axes
    for axnum,ax in enumerate(axes):
        position=ax.get_position()
        x=position.x0-0.06
        y=position.y1
        row=int(axnum/numcols)
        col=axnum%numcols
        fig.text(x,y,letters[col]+str(row+1), fontsize=fsize)
    return

def combined_discrim_Qhx(Qhx,boundaries,ept,phases,all_ideals,all_beta=[],Qlen=[],agent=-1):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    
    Qkeys=list(Qhx.keys())
    beta_axnum=len(Qhx[Qkeys[0]].keys())
    if len(Qlen):
        numrows=beta_axnum+2 
    else:
        numrows=beta_axnum+1
    
    fig=plt.figure()
    gs=GridSpec(numrows,2)
    ax=[]
    for row in range(numrows):
        ax.append(fig.add_subplot(gs[row,:]))

    fig=plot_Qhx_2D(Qhx,boundaries,ept,phases,all_ideals,fig=fig,ax=ax[0:beta_axnum]) 
    if isinstance(all_beta,list):
        beta=all_beta[agent]
        Xvals=np.arange(len(beta))/ept
        ax[beta_axnum].plot(Xvals,beta)
        #ax[beta_axnum].plot(Xvals,np.mean(beta,axis=0))
    else:
        for key in all_beta.keys():
            beta=all_beta[key]
            Xvals=np.arange(np.shape(beta)[1])/ept
            ax[beta_axnum].plot(Xvals,np.mean(beta,axis=0),label=key)
    ax[beta_axnum].set_ylabel(r'$\beta$1')
    min_beta=np.min(beta)
    max_beta=np.max(beta)
    yticks=[str(round(f,1)) for f in np.arange(min_beta,max_beta+.01,0.5)]
    ax[beta_axnum].set_yticks(np.arange(min_beta,max_beta+.01,0.5),yticks)
    #ax[beta_axnum].legend()
    if len(Qlen):
        qnum=beta_axnum+1
        for key1,data in Qlen.items(): 
            if isinstance(data,dict):
                for key2,d2 in Qlen[key1].items():
                    ax[qnum].plot(Xvals,d2[agent],label=key1+',Q'+str(key2))
                ymax=np.max([d2[agent] for d2 in Qlen[key1].values()] )
            else:           
                ax[qnum].plot(Xvals,data[agent],label='Q'+str(key1+1))
                ymax=np.max([data[agent] for data in Qlen.values()] )
        ax[qnum].legend()
        ax[qnum].set_ylabel('States',fontsize=fsizeSml+1)
        yticks = np.linspace(0,np.ceil(ymax)+np.ceil(ymax)%2,3,endpoint=True)
        ylabels=[str(round(f)) for f in yticks]
        ax[qnum].set_yticks(yticks,ylabels)
    ax[-1].set_xlabel('Trial',fontsize=fsizeSml+1)

    #length_q=[qq for q_row in data for qq in q_row]
    #ax[3].plot(Xvals,length_q,label='Q'+str(q+1))

    ##### add vertical line showing phase boundaries ####
    add_boundaries(numrows,Qhx,boundaries,ax)

    #add subplot labels 
    '''fsize=14
    blank=0.03
    label_inc=1/len(fig.axes)
    for row in range(len(fig.axes)):
        y=(1-blank)-(row*label_inc) #subtract because 0 is at bottom
        fig.text(0.02,y,letters[row], fontsize=fsize)'''
    fig.tight_layout()
    return fig

def add_boundaries(numrows,Qhx,boundaries,ax):
    for axnum in range(3,numrows):
        q=list(Qhx.keys())[0]
        state=list(Qhx[q].keys())[0]
        ylim=ax[axnum].get_ylim()
        for jj,xval in enumerate(boundaries[q][state][1:]):
            ax[axnum].vlines(xval,ylim[0],ylim[1],linestyles='dashed',color='grey')

def new_xlim(figs,xlim):
    for figname in figs.keys():
        ax=figs[figname].axes
        for axis in ax:
            axis.set_xlim(xlim)

def load_data(Qfil,trial,parname='params'):
    data=np.load(Qfil,allow_pickle=True)
    try:
        allQhx=data['all_Qhx'].item()
    except:
         allQhx=data['all_Qhx']
    try:
        all_bounds=data['all_bounds'].item()
        all_ideals=data['all_ideals'].item()
    except:
        all_bounds=data['all_bounds']
        all_ideals=data['all_ideals']
    if isinstance(allQhx,np.ndarray) or isinstance(allQhx,list):
        Qhx=allQhx[trial]
    if 'all_beta' in data.keys(): #files beginning on 3 June 2022
        try:
            all_beta=data['all_beta'].item()
            all_lenQ=data['all_lenQ'].item()
        except:
            all_beta=data['all_beta']
            all_lenQ=data['all_lenQ']
    else:
        all_beta={}
        all_lenQ={}
    if parname in data.keys():
        events_per_trial=data[parname].item()['events_per_trial']
        trials_per_block=data[parname].item()['trials_per_block']
    else:
        events_per_trial=data['events_per_trial']
        trials_per_block=10 #
    phases=[[p for p in phs] for phs in data['phases']]
    return allQhx,all_bounds,all_ideals,all_beta,all_lenQ,events_per_trial,trials_per_block,phases

if __name__ == "__main__":
    import os
    task='Bandit' #'Discrim' # 'AIP'#'sequence'#
    if task=='sequence':
        from sequenceFiles import fil,param_name        
        figs={q:{} for q in fil.keys()}
        trial=-1
        #
        state_action_combos=[('Llever','---L'),('Llever','--LL'),('Llever', 'RRLL'),('Llever', 'RLLL'),('Rlever','--LL'),('Rlever','LLLL'),('Rlever','RLLL')] #,('Rlever','-LLL')
        state_action_combos=[('Llever','---L'),('Llever','--LL'),('Rlever','--LL'),('Rlever','LLLL'),('Rlever','-LLR'),('Rlever','LLLR')]#,('Rlever','LLRR')]
        actions_colors={'goL':'b','goR':'r','press':'k','goMag':'gray'}
        actions_lines={'goL':'solid','goR':'solid','press':'dashed','goMag':'dotted'}
        allQhx={}
        for nQ,f in fil.items():
            data=np.load(f+'.npz',allow_pickle=True)
            events_per_trial=data[param_name].item()['events_per_trial']
            trials_per_block=data[param_name].item()['trials_per_block']
            try:
                allQhx[nQ]=data['Qhx'].item()
            except:
                allQhx[nQ]=data['Qhx']          
            #numcols+=data['par'].item()['numQ']
            #fig=plot_Qhx_sequence_1fig(allQhx[nQ],state_action_combos,actions_colors,events_per_trial)
            #figs=staticQ(f,figs,nQ)
        fig=plot_Qhx_sequence_1fig(allQhx,state_action_combos,actions_colors,events_per_trial,actions_lines)
    else:
        if task=='Bandit':
            from banditFiles import fil
            trial=1 #Bandit, [3,23,39] for June 3 2022 #[2,5,28] - for June 6, 2022
        elif task=='Discrim':
            from discrimFiles import fil
            trial=0
        elif task=='AIP':
            from discrimFiles import AIPfil as fil
            trial=0
        figs={q:{} for q in fil.keys()}
        for nQ,f in fil.items():
            if len(os.path.dirname(f)):
                Qfil=os.path.dirname(f)+'/Qhx'+os.path.basename(f)+'.npz'
            else:
                Qfil='Qhx'+os.path.basename(f)+'.npz'
            all_Qhx,all_bounds,all_ideals,all_beta,all_lenQ,events_per_trial,trials_per_block,phases=load_data(Qfil,trial)
            if task=='Bandit': #bandit task only, Qhx for all trials is saved
                data=np.load(Qfil,allow_pickle=True)
                norm=1/trials_per_block 
                phases=data['phases']
                #if 'random_order' not in other.keys():
                #    random_order=[phases]
                if 'random_order' in data.keys():
                    random_order=data['random_order']
                else:
                    #order of probability not saved, except for last trial
                    random_order=[phases]
                    trial=-1
                Qhx=all_Qhx[trial]
                bounds=all_bounds[trial]
                ideals=all_ideals[trial]               
                if 'num_blocks' in data.keys():
                    num_blocks=data['num_blocks'].item()
                else: #forgot to store num_blocks.  calculate it
                    numevents=len(all_lenQ[0][0])
                    num_trials=numevents/events_per_trial
                    trials_per_phase=num_trials/len(phases)
                    num_blocks=int(trials_per_phase/trials_per_block) #10
                print(Qfil,num_blocks)
                data2=np.load(f+'.npz',allow_pickle=True)
                traject_dict=data2['traject_dict'].item()
                agent_response([trial],random_order,num_blocks,traject_dict,norm=norm)
                fig_combined=combined_bandit_Qhx_response(random_order,num_blocks,traject_dict,Qhx,bounds,events_per_trial,phases,agent_num=trial,all_beta=all_beta,norm=norm)#,all_lenQ)
            else: #Discrim, or older Bandit files, random order not saved, all_Qhx is single dictionary,
                states=['Pport,6kHz 0','Pport,10kHz 0'] #for discrim,reverse
                #states=['Pport,6kHz 1'] #for acq,extinct
                if task=='AIP':
                    states=['Pport,10kHz 0'] #for acq,discrim with block
                if len(all_Qhx)>1:
                    Qhx=all_Qhx[trial]
                    bounds=all_bounds[trial]
                    ideals=all_ideals[trial]
                else:
                    Qhx=all_Qhx[0]
                    bounds=all_bounds[0]
                    ideal=all_ideals[0]
                for state in states:
                    if int(nQ)>1:
                        Qhx_subset={state:{k:v[state] for k,v in Qhx.items()}}
                        bounds_subset={state:{k:v[state] for k,v in bounds.items()}}
                        ideals_subset={state:{k:v[state] for k,v in ideals.items()}}
                        keynum=int(state.split()[-1])
                        if len(all_beta):
                            bkey=list(all_beta.keys())[keynum]
                            beta=copy.deepcopy(all_beta[bkey])
                            if task=='AIP':
                                figs[nQ]['qhx'+state]=combined_discrim_Qhx(Qhx_subset,bounds_subset,events_per_trial,phases,ideals_subset,all_beta=beta)
                            else:
                                figs[nQ]['qhx'+state]=combined_discrim_Qhx(Qhx_subset,bounds_subset,events_per_trial,phases,ideals_subset,all_beta=beta,Qlen=all_lenQ[bkey])
                            print('figures for',state,f,',bkey =',bkey)
                        else:
                            figs[nQ]['qhx']=plot_Qhx_2D(Qhx_subset,bounds_subset,events_per_trial,phases,ideals_subset,title=state)
                        print('figures for all:',f)
                if int(nQ)>1:
                    figs[nQ]['qhx_all']=plot_Qhx_2D(Qhx,bounds,events_per_trial,phases,ideals,title='all phases')
                else:
                    newQhx={1:Qhx,2:{}}
                    newbounds={1:bounds,2:{}}
                    newideals={1:ideals,2:{}}
                    figs[nQ]['qhx_all']=combined_discrim_Qhx(newQhx,newbounds,events_per_trial,phases,newideals,all_beta=all_beta,Qlen=all_lenQ)
                #figs=staticQ(f,figs,nQ)
                if 'Pport,10kHz 0' in states and task != 'AIP':
                    new_xlim(figs[nQ],[190,610])