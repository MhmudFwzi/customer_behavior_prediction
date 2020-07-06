import pandas as pd
import numpy as np
from tabulate import tabulate


def create_2cat_table(df, cat_var1, cat_var2, observation):
    '''
    cerate tables for anova observation
    input: 
        df: dfset type of panda dfframe
        cat_var1: label of first cat variable
        cat_var2: label of second cat variable
        observation: label of traget variable
        
    
    '''
    
    # list for storing tabulate df
    tabulate_list = []

    # iterate over first var
    for cat_1 in np.unique(df[cat_var1]):

        sub_group=''
        sub_len=''
        sub_mean=''
        sub_std=''

        # iterate over second var
        for cat_2 in np.unique(df[cat_var2]):

            # get value for given cat_1 and cat_2
            tmp_df = df[(df[cat_var1]==cat_1) & (df[cat_var2]==cat_2)][observation]

            # labels
            sub_group=sub_group+'\n'+str(cat_2)
            
            # values
            sub_len=sub_len+'\n'+str(len(tmp_df))
            sub_mean=sub_mean+'\n'+ '%.2f' % np.mean(tmp_df)
            sub_std=sub_std+'\n'+ '%.2f' % np.std(tmp_df)

        # add to main tabulate list
        tabulate_list.append([cat_1, sub_group, sub_len, sub_mean, sub_std,
                             np.mean(df[df[cat_var1]==cat_1][observation]),
                             np.std(df[df[cat_var1]==cat_1][observation])])

    # print tabulate
    print(tabulate(tabulate_list, headers=[cat_var1, cat_var2, 'N', 
                                        observation+'\nmean', 
                                        observation+'\nstd',
                                        observation+'\nmean_all', 
                                        observation+'\nstd_all'], tablefmt='grid'))
    
    
    
def create_1cat_table(df, cat_var1, observation):
    '''
    cerate tables for anova observation
    input: 
        df: dfset type of panda dfframe
        cat_var1: label of first cat variabl
        observation: label of traget variable
    '''
    
    # list for storing tabulate df
    tabulate_list = []

    # iterate over first var
    for cat_1 in np.unique(df[cat_var1]):
     
        # get value for given cat_1 and cat_2
        tmp_df = df[df[cat_var1]==cat_1][observation]

        # add to main tabulate list
        tabulate_list.append([cat_1, len(tmp_df), 
                              '%.2f' % np.mean(tmp_df),
                              '%.2f' % np.std(tmp_df)])

    # print tabulate
    print(tabulate(tabulate_list, headers=[cat_var1, 'N', 
                                        observation+'\nmean', 
                                        observation+'\nstd'], tablefmt='grid'))