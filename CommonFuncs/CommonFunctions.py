# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:24:17 2019

@author: Asha
"""
import math

class Binning(object):
    def mann_wald_test(self,var,c):
        N = len(var)
        k = 4 * ((2*(N-1)**2/c**2))**(1/5)
        return int(k)
    def struges(self,var):
        N = len(var)
        k = int(1 + 3.322 * math.log(N,10))
        return int(k)

# Differentiating Continuous and Categorical Variables
class DescriptiveAnalysis(object):
    
    def __init__(self,df):
        self.df = df
        
    def cont_categ_variables(self):
        cont = []
        categ = []
        for each in list(self.df):
            if self.df[each].dtype.kind in 'bifc':            
                cont.append(each)
            else:
                categ.append(each)
        return cont,categ
    
    def descriptive_stats_cont(self,colname):
        meandf = self.df[colname].mean()
        mediandf = self.df[colname].median()
        modedf = self.df[colname].mode()
        stddf = self.df[colname].std()
        skewdf = self.df[colname].skew()
        kurtdf = self.df[colname].kurt()
        mindf = self.df[colname].min()
        maxdf = self.df[colname].max()
        print("Mean",meandf)
        print("Median",mediandf)
        print("Mode",modedf)
        print("Skew",skewdf)
        print("Kurtosis",kurtdf)
        print("Standard Deviation",stddf)
        print("Min",mindf)
        print("Max",maxdf)
        return({'mean':meandf,'median':mediandf,'mode':modedf,'std':stddf,'kurtosis':kurtdf,'skew':skew,'min':mindf,'max':maxdf})
        
    def descriptive_stats_categ(self,colname):
        print(self.df.groupby(colname).value_counts())
    
        