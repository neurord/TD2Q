submission=3
actions=[(('Pport', '6kHz'),'left'), (('Pport', '6kHz'),'right')]
action_text=['6 kHz Left','6 kHz Right']
test_variables=[k+'_rwd__End' for k in ['50:50','10:50','10:90','50:90','90:10','90:50','50:10']]
if submission==1:
    pattern='Bandit2021-12-14_numQ*_alpha*beta0.1.npz'#'Bandit2021-05-28*beta0.1.npz'#'Bandit2021-05-28_numQ2_alpha0.6_0.3_beta0.7.npz'#
    dep_var=['split','beta_min'] #,'numQ']#'Q2other', 'decision_rule']#'trial_subset']# 
    #bandit sims from 2021 dec 14-16 used for statistics and trajectory plots
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
    #Qhx graphs
    #new Bandit sim (2022jun3) used for Fig 5 - example
    fil={'2':'Bandit2022-06-03_numQ2_alpha0.4_0.2_q2o0.1_beta0.1_splitTrue_window3'}#'Bandit2021-12-16_numQ2_alpha0.4_0.2_q2o0.1_beta0.9_splitTrue'} #}

elif submission==2:
    subdir0='ManuscriptFiles/'#'NormEuclidPLoSsubmission2_Q2other0/' #q2other=0.0 
    test='numQ' #'OpAL'#'rwd'#'alpha'#'beta'#'gamma' #'split'#,'beta_min', 'decision_rule' 'AIP' 'Q2other'#
    dep_var=[test]
    files=None
    # for Q2 trajectory
    keys=None
    fil=None
    meanQ=False #make true if you know that meanQ has been added to file, e.g. if use_oldQ is False
    pattern=subdir0+'Bandit2023-01-1?numQ2*decision_ruleNone_beta_min0.5_beta1.5_gamma*_splitTrue.npz'
    #For stat test assessing numQ (and split)
    if test=='numQ':
        fil={'1':subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue',
             '2':subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue'}
        files=[subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTruesimple.npz',
                subdir0+'Bandit2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTruesimple.npz']
        files=[subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz',
                subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz']
    elif test=='split':
        keys=['split','no split']
        files=[ subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz',
                subdir0+'Bandit2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitFalse.npz']
    elif test=='gamma':
        pattern=subdir0+'Bandit2023-01-10numQ2*decision_ruleNone_beta_min0.5_beta1.5_gamma*_splitTrue.npz'
    elif test=='beta':
        pattern=subdir0+'Bandit2023-01-10numQ2*decision_ruleNone_beta_min0.5_beta*_gamma0.82_splitTrue.npz'
    elif test=='split':
        pattern=subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_split*'
        files=None
elif submission==3:
    subdir0='Bandit2023may19/'
    test='numQ' #'window_bmin'#'OpAL'#'initQ'#'beta'#'gamma' #'rwd'#'alpha'#,'beta_min', 'decision_rule' 'AIP' 'Q2other'#
    dep_var=[test]#  ,'beta_min'] for statistic test of constant beta.  Move beta_min0.1 files out of directory
    files=None
    # for Q2 trajectory
    keys=None
    fil=None
    pattern=subdir0+'Bandit2023-05-19numQ*'
    meanQ=False #make true if you know that meanQ has been added to file, e.g. if use_oldQ is False
    if test=='numQ':
        fil={'1':subdir0+'Bandit2023-05-26numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue',
            '2':subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue'}
        files=[subdir0+'Bandit2023-05-26numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue.npz',
                subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue.npz']
        #files[1]=subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrueWin1.npz'
    elif test=='gamma':
        pattern=subdir0+'Bandit2023-05-19numQ2*decision_ruleNone_beta_min0.5_beta1.5_gamma*_use_OpalFalse_step1False*initQ-1_rwd10_-1*.npz'
    elif test=='beta':
        pattern=subdir0+'Bandit2023-05-19numQ2*decision_ruleNone_beta_min*_beta*_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1*.npz'
    elif test=='initQ':
        pattern=subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ*_rwd10_-1_wanderTrue.npz'
        files=None        
    elif test=='rwd': #old test.  Files deleted
        fil={'5':'Bandit2023-05-02numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd5',
            '10':'Bandit2023-05-02numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10'}
        files=[ 'Bandit2023-05-01numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd5',
                'Bandit2023-05-02numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10']
    elif test=='window_bmin':
        keys=['win1','ctrl','bmin0.1']
        files=[subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrueWin1.npz',
                subdir0+'Bandit2023-05-26numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue.npz',
                subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_ruleNone_beta_min0.1_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue.npz']
    elif test=='OpAL': 
        files=[subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_ruledelta_beta_min1_beta1_gamma0.1_use_OpalTrue_step1False_D2_ruleOpal_initQ1_rwd8_0_wanderTrue.npz',
            subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue.npz']
        keys=['OpAL','TD2Q']
        fil={'OpAL':subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_ruledelta_beta_min1_beta1_gamma0.1_use_OpalTrue_step1False_D2_ruleOpal_initQ1_rwd8_0_wanderTrue',
            'TD2Q':subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_use_OpalFalse_step1False_D2_ruleNone_initQ-1_rwd10_-1_wanderTrue'}
        pattern=subdir0+'Bandit2023-05-19numQ2_beta_GPi10_decision_rule*_beta_min*_beta*_gamma*_use_Opal*_step1False_D2_rule*_init*_wanderTrue'


