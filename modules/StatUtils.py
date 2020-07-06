'''
contains some of our stat tools
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class StatUtils(object):
    

    def remove_outliers(self, df, col_name):
        '''
        clear outliers using IQR.

        input:
            df: panda dataframe
            col_name: name of the target column 

        output: return modified df

        '''

        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # drop outliers
        df.drop(df[(df[col_name] > upper_bound) | (df[col_name] < lower_bound) ].index , inplace=True)
        
        return df
    
    
    def independence_test(self, lm, title='Independence of variables'):
        '''
        gets fitted lm model and plot residuals vs fitted valuesfor checking independence
        '''

        fig = plt.figure()

        fig.axes[0] = sns.residplot(lm.fittedvalues, lm.resid,
                              lowess=True,
                              scatter_kws={'alpha': 0.5},
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        fig.axes[0].set_title(title)
        fig.axes[0].set_xlabel('Fitted values')
        fig.axes[0].set_ylabel('Residuals')
        