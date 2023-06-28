submission=3
zero_responses=['extinc_Pport_10kHz_left_Beg','extinc_Pport_10kHz_left_End', 'extinc_Pport_10kHz_right_Beg', 'extinc_Pport_10kHz_right_End',
                'renew_Pport_10kHz_left_Beg', 'renew_Pport_10kHz_left_End', 'renew_Pport_10kHz_right_Beg', 'renew_Pport_10kHz_right_End',
                'acquire_Pport_10kHz_left_Beg', 'acquire_Pport_10kHz_left_End', 'acquire_Pport_10kHz_right_Beg', 'acquire_Pport_10kHz_right_End']

test_variables=['acquire_rwd__End','discrim_rwd__End','reverse_rwd__End'] 
#test_variables=['renew_Pport_6kHz_left_Beg', 'extinc_Pport_6kHz_left_Beg']
                #'discrim_Pport_6kHz_left_End','discrim_Pport_10kHz_right_End',
                #'reverse_Pport_6kHz_right_End','reverse_Pport_10kHz_left_End',]

actions=['rwd',(('Pport', '6kHz'),'left')]
action_text=['reward','6 kHz Left']

if submission==1:
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
    #Qhx graphs
    fil={'1':'Discrim2021-12-13_numQ1_alpha0.3_0_st1.0_0', '2':'Discrim2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625'}
    AIPfil={'2':'NormEuclidPLoSSubmission1/DiscrimD2AIP2021-12-13_numQ2_alpha0.2_0.1_st0.75_0.625'}
    #new set of Discrim simulations used for Fig 2,3 - examples and mean trajectory
    fil={'2':'DiscrimD2AIP2022-06-06_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue'}
    fil={#'1':'Discrim2022-06-14_numQ1_alpha0.3_0_st1.0_0_q2o0.1_beta0.5_splitTrue',
             '2':'Discrim2022-06-14_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.1_beta0.5_splitTrue' }

elif submission>=2:
    subdir0='NormEuclidPLoSsubmission2_Q2other0/' #q2other=0.0
    subdir0='ManuscriptFiles/'
    test='numQ'#'AIP' #'split'# 'alpha'#'gamma' #'beta'# 'beta_min',  'decision_rule' 
    test_variables=['rwd__End']
    dep_var=[test]
    keys=None
    files=None
    pattern='discrim2022-12-01_numQ?_alpha*.npz'
    #keys=['bmin0.1','bmin0.5'] #1 key per file
    if test=='numQ':
        files=[subdir0+'Discrim2022-12-19_numQ1_alpha0.3_0_st1.0_0_q2o0.1_gamma0.82_bmin0.5_bmax1.5_splitTrue_ruleNone.npz',
            subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz']
        fil={'2':subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue',
            '1':subdir0+'Discrim2022-12-19_numQ1_alpha0.3_0_st1.0_0_q2o0.1_gamma0.82_bmin0.5_bmax1.5_splitTrue_ruleNone'} #Qhx file for 1Q has been lost
    if test=='split':
        keys=['initQ=0','split']#, 'initQ=1']
        files=[subdir0+'Discrim2023-01-10_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.0_gamma0.82_bmin0.5_bmax1.5_splitFalse_ruleNone.npz',
               subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz']#,
               #subdir0+'Discrim2023-04-27_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.0_gamma0.82_bmin0.5_bmax1.5_split1_ruleNone.npz']
    if test=='alpha':
        import glob
        newfiles=glob.glob(subdir0+'Discrim2023-04-21_numQ2_alpha0.2_*_st0.75_0.625_q2o0.0_gamma0.82_bmin0.5_bmax1.5_split-1_ruleNone.npz')
        files=[]
        for f in newfiles:
            files.append(f)
    if test=='AIP':
        #actions=['rwd',(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        #action_text=['reward','6 kHz Left', '10 kHz Right', '10 kHz Left']
        actions=[(('Pport', '6kHz'),'left'),(('Pport', '10kHz'),'right'), (('Pport', '10kHz'),'left')]
        action_text=['6 kHz Left', '10 kHz Right', '10 kHz Left']
        pattern='DiscrimD2AIP*.npz'
        keys=['block', 'ctrl']
        test_variables=['acquire_rwd__End','discrim_rwd__End']
        fil={}
        AIPfil={'2':subdir0+'DiscrimD2AIP2023-01-10_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.0_gamma0.82_bmin0.5_bmax1.5_splitTrue_ruleNone'}
        files=[subdir0+'DiscrimD2AIP2023-01-10_numQ2_alpha0.2_0.1_st0.75_0.625_q2o0.0_gamma0.82_bmin0.5_bmax1.5_splitTrue_ruleNone.npz',
            subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma0.82_splitTrue.npz']
    elif test=='gamma':        #test gamma
        pattern=subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta1.5_gamma*_splitTrue*.npz'
    elif test=='beta':        #test beta
        pattern=subdir0+'Discrim2023-01-10numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta*_gamma0.82_splitTrue*.npz'
