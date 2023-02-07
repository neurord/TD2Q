import numpy as np

def phase_to_prob(phs):
    Lprob=int(phs.split(':')[0])
    Rprob=int(phs.split(':')[1])
    return {'L':Lprob,'R':Rprob}
def opposite_turn(turn):
    return 'L' if turn=='R' else 'R'

def count_priors(prior_phase,prior,turn,prior_counts,random_keys,run):
    prior_phase['all'+turn].append(prior)
    first_probs=phase_to_prob(random_keys[run][0]) #reward prob on 1st block
    if prior != None:
        prior_probs=phase_to_prob(prior)
        if prior_probs[turn]==90: #prior_probs[turn] is reward ratio for same turn in prior block
            #print('all',turn,' with prior=', prior,'on run=', run, 'all phases=',random_keys[run])
            prior_counts['all'+turn]['90']+=1
        elif (prior_probs[turn]==50 and prior_probs[opposite_turn(turn)]==10):
            #print('all',turn,' prior=', prior,'on run=', run, 'all phases=',random_keys[run])
            prior_counts['all'+turn]['50:10']+=1
        elif first_probs[turn]>first_probs[opposite_turn(turn)]:  #probs[0] has reward prob for direction that matches
            #print('all',turn,' with FIRST phase=', random_keys[run][0],'on run=', run, 'all phases=',random_keys[run])
            prior_counts['all'+turn]['first_block']+=1
        else:
            print('what is going on?',turn,run,random_keys[run])
    else:
        prior_counts['all'+turn]['none']+=1
    return prior_phase,prior_counts

def perseverance(traject_dict,num_runs,random_keys,phase):
    persever={'allL':0,'allR':0};prior_phase={'allL':[],'allR':[]}
    prior_counts={k:{r:0 for r in ['90','50:10','none','first_block']} for k in prior_phase.keys()}
    for run in range(num_runs):
        L=np.sum(traject_dict[phase][(('Pport', '6kHz'),'left')][run])
        R=np.sum(traject_dict[phase][(('Pport', '6kHz'),'right')][run])
        phase_index=list(random_keys[run]).index(phase)
        if phase_index>0:
            prior=random_keys[run][phase_index-1]
        else:
            prior=None
        if L==0: #R=10, always choose R
            turn='R' #, no 'L' ~= all R
        if R==0:
            turn='L' #np 'R' ~= all L
        if L==0 or R==0:
            persever['all'+turn]+=1
            prior_phase,prior_counts=count_priors(prior_phase,prior,turn,prior_counts,random_keys,run) #Lprob is prob of L (Same) on 1st block
    persever['prior']=prior_counts
    return persever,prior_phase

if __name__=='__main__':
    import sys
    import glob
    #fnames=[sys.argv[1]] #specify one filename on command line
    #fnames=['Bandit2023-01-30numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min1.5_beta1.5_gamma0.82_splitTrue.npz'] #specify filename
    fnames=glob.glob('Bandit*.npz') #evaluate set of files
    for fname in fnames:
        data=np.load(fname,allow_pickle=True)
        traject=data['traject_dict'].item()
        qdata=np.load('Qhx'+fname,allow_pickle=True)
        random_order=qdata['random_order'] 
        runs=len(random_order)
        persever,prior_phase=perseverance(traject,runs,random_order,'50:50')
        print(fname,persever)