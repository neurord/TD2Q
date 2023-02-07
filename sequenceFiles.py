submission=2
actions=['rwd',(('Llever', '**LL'), 'goR'), (('Rlever', '*LLR'), 'press')]
action_text=['reward', '**LL, go Right','*LLR press R']
if submission==1:
    param_name='params'
    pattern='Sequence2022-08-16_*.npz'
    barplot_files=[ 'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz',
                'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue_inactiveD1.npz',
                'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue_inactiveD2.npz']

    keys=['Ctrl','inactiveD1','inactiveD2']#
    dep_var=['inact']
    test_variables=['rwd__End']#,'Llever_**LL_goR_End'.replace('*','x'), 'Rlever_*LLR_press_End'.replace('*','x'),'Rlever_xxLL_press_End','Rlever_LLRR_goMag_End']
    #dep_var=['numQ'] #'split','beta_min']#, 'Q2other','split','beta_min'] #'trial_subset']# 'decision_rule']#
    #files=['Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz','Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz']
    files=[ 'Sequence2022-08-10_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue.npz','Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue.npz']
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
    #Qhx files 
    fil={'1':'Sequence2021-12-16_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue','2':'Sequence2021-12-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue'}
    fil={'1':'Sequence2022-08-10_HxLen4_numQ1_alpha0.2_0_st0.75_0_q2o0.1beta0.9splitTrue','2':'Sequence2022-08-16_HxLen4_numQ2_alpha0.2_0.35_st0.75_0.625_q2o0.1beta0.9splitTrue'}
    
elif submission==2:
    subdir0='NormEuclidPLoSsubmission2_Q2other0/' #q2other=0.0
    param_name='par'
    test='numQ' #'inact'#'gamma'#'beta' #'Q2other'#
    testsplit=True
    test_variables=['rwd__End']
    fil={'1':subdir0+'Sequence2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue', 
        '2':subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue'}
    #fil={'1':subdir1+'Sequence2022-12-22numQ1_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue', #D2 gamma=D1 gamma
    #    '2':subdir1+'Sequence2022-12-22numQ2_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue'}#D2 gamma=D1 gamma
    #fil={'2':subdir1+'Sequence2022-12-22numQ2_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue_inactiveD1',#D2 gamma=D1 gamma and inact effect on final choice
    #      '1':subdir1+'Sequence2022-12-22numQ2_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue_inactiveD2'}#D2 gamma=D1 gamma and inact effect on final choice
    if test=='inact':        
        ##### To test effect of inactivation ########
        keys=['Ctrl','inactiveD1','inactiveD2']#
        dep_var=['inact'] #using Q2other0.1, D2 had factor=0.6, D1 had factor=2
        #test_variables=test_variables+['Llever_**LL_goR_End'.replace('*','x'), 'Rlever_*LLR_press_End'.replace('*','x'),'Rlever_xxLL_press_End','Rlever_LLRR_goMag_End']
        #dep_var=['numQ'] #'split','beta_min']#, 'Q2other','split','beta_min'] #'trial_subset']# 'decision_rule']#
        files=[subdir0+'Sequence2023-01-04numQ2_Q1other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_0.45_rwd15_splitTrue.npz', #D2 gamma=0.5*D1 gamma, inact effect on final choice
               subdir0+'Sequence2023-01-04numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_0.45_rwd15_splitTrue.npz'] #D2 gamma=0.5*D1 gamma, inact effect on final choice
        barplot_files=[subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz',#
                subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue_inactiveD1_2.npz', 
                subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue_inactiveD2_0.5.npz'] 
        files=[subdir0+'Sequence2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz', #D2 gamma=0.5*D1 gamma, inact effect on final choice
               subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz'] #D2 gamma=0.5*D1 gamma, inact effect on final choice
        pattern=subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue*'
        '''######### effect of Q2other
        pattern='Sequence2022-12-??numQ2_Q2other0.?_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz'
        barplot_files=files=['Sequence2022-12-22numQ2_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz',
                             'Sequence2022-12-22numQ1_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz',
                             'Sequence2022-12-30numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz',]
        dep_var=['Q2other']
        keys=['Q2_0.1','Q1_0.1','Q2_0.0']
        '''
    else:
        barplot_files=[]
        dep_var=[test]
        keys=None
        files=None
        if test=='numQ':
                #Q2other=0
                files=[subdir0+'Sequence2023-01-17numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz',
                        subdir0+'Sequence2023-01-17numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz']
                splitfalse=[subdir0+'Sequence2023-01-11numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitFalse.npz',
                        subdir0+'Sequence2023-01-11numQ1_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitFalse.npz']
                pattern=subdir0+'Sequence2023-01-17numQ*_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.95_rwd15_splitTrue.npz'
                if testsplit:
                        files=files+splitfalse
                        dep_var=dep_var+['split']
        elif test=='gamma':        #test gamma
                pattern=subdir0+'Sequence2023-01-10numQ2_Q2other0.0_decision_ruleNone_beta3_beta_min0.5_gamma*_beta_GPi10_rwd15_splitTrue*'
        elif test=='beta':        #test beta
                pattern=subdir0+'Sequence2023-01-11numQ2_Q2other0.0_decision_ruleNone_beta*_beta_min0.5_gamma0.95_beta_GPi10_rwd15_splitTrue*'
        elif test=='Q2other':
                pattern=subdir0+'Sequence2022-12-??numQ2_Q2other0.?_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue*'
                files=[subdir0+'Sequence2022-12-22numQ2_Q2other0.1_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz',
                        subdir0+'Sequence2022-12-30numQ2_Q2other0.0_beta_GPi10_decision_ruleNone_beta_min0.5_beta3_gamma0.9_rwd15_splitTrue.npz',]

