submission=2
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
    subdir0='NormEuclidPLoSsubmission2_Q2other0/' #q2other=0.0
    test='numQ' #'beta'#'gamma' #'split'#,'beta_min', 'decision_rule' 'AIP' 'Q2other'#
    dep_var=[test]
    files=None
    # for Q2 trajectory
    keys=None
    fil=None
    pattern=subdir0+'Bandit2023-01-1?numQ2*decision_ruleNone_beta_min0.5_beta1.5_gamma*_splitTrue.npz'
    #For stat test assessing numQ (and split)
    if test=='numQ':
        fil={'1':subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue',
             '2':subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue'}
        files=[ subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz',
                subdir0+'Bandit2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz']
        files=[subdir0+'Bandit2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTruesimple.npz',
                subdir0+'Bandit2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTruesimple.npz']
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
