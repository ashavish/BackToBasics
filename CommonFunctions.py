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
def cont_categ_variables(df):
    cont = []
    categ = []
    for each in list(df):
        if df[each].dtype.kind in 'bifc':            
            cont.append(each)
        else:
            categ.append(each)
    return cont,categ