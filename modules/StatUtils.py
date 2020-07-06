'''
contains some of our stat tools
'''

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
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
        
        
        
    def chi_squared_test(self, table):
        '''
        Prints the contingency table. The chi-squared test is calculated and 
        the degrees of freedom is indicated. The critical value is calculated and
        interpreted, finding that indeed the variables are independent or not.
        
        inputs:
        
            table: cross tab of given categorical (nominal) variables
        '''
        stat, p, df, expected = stats.chi2_contingency(table)

        print('df=%d\n' % df)
        
        print('Expected values:\n')
        expected = pd.DataFrame(expected)
        expected.columns = table.columns
        expected.index = table.index
        print(expected)
    
        # interpret p-value
        significance = 0.05
        print('\nchi=%.4f, p-value=%.4f\n' % (stat, p))
        
        if p <= significance:
            print('At significance=%.3f, reject null hypotheses: thus categorical variables are dependent' % significance)
        else:
            print('At significance=%.3f, accept null hypotheses: thus categorical variables are independent' % significance)
            
            
    def mcnemar_test(self, table):
      
        '''
        Caculates mcnemar test for categorical (nominal) varables, used when any frequency is small (<5) 
        inputs:
            table: cross tab of given categorical (nominal) variables
        '''
        
        # calculate mcnemar test
        result = mcnemar(table, exact=True)
        # summarize the finding
        print('statistic=%.4f, p-value=%.4f\n' % (result.statistic, result.pvalue))
        # interpret the p-value
        significance = 0.05
        if result.pvalue <= significance:
            print('At significance=%.4f, reject null hypotheses: categories has impact on each other' % significance)
        else:
            print('At significance=%.4f, accept null hypotheses: categories do not have impact on each other' % significance)

        