# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:47:31 2019

@author: Asha
"""

# Normality Tests in Python

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sn
import math
import statsmodels.api as sm
import scipy
import statistics
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import kstest

import scipy
import statistics
    
# Class to check normality using python
class Normality(object): 
    def check_normality(self,data):
        # Descriptive - Mean, median, mode, skew, kurtosis
        mean_ = np.mean(data)
        median_ = np.median(data)
        mode_ = stats.mode(data)
        kurtosis_ = kurtosis(data)
        skew_ = skew(data)
        print("Mean is ",mean_)
        print("Median is ",median_)
        print("Mode is ",mode_)
        print("Kurtosis is ",kurtosis_)
        print("Skew is ",skew_)

        # Histograms & Distribution Plots
        p = plt.figure()
        bins_ = int(1+ 3.322 * math.log(len(data),10))
        distplot = sn.distplot(data,hist=True,kde=True,bins=bins_)

        # PP Plot
        p = plt.figure()
        probplot = sm.ProbPlot(data)
        ppplot = probplot.ppplot(line='45')    

        # Q-Q Plot
        p = plt.figure()
        q = qqplot(data, line='s')

        # Box Plot
        p = plt.figure()
        box = sn.boxplot(data)

        # Shapiro Wilks Test
        stat, p = shapiro(data)
        print("P-value from the Shapiro Wilks Test is ",p)

        #  D’Agostino’s K^2 Test
        stat, p = normaltest(data)
        print("P-value from the D'Agostino's Test is ",p)

        # Anderson-Darling Test
        result = anderson(data)
        print("Result from Anderson-Darling Test is ",result)

        # Kolmogorov-Smirnov Test
        result =kstest(data, 'norm', args=(5, 3))
        print("Result from Kolmogorov-Smirnov Test is ",result)

    def chi_sq_normality_test(self,var,n,bin_vals=None):
        N = len(var)
        low_val = min(var)
        max_val = max(var)    
        bin_width = (max_val - low_val)*1.0/n
        bin_low_val = []
        bin_max_val = []    
        observed_freq = []
        expected_freq = []
        mean = statistics.mean(var)
        stdev = statistics.stdev(var)
        for i,bin in enumerate(range(0,int(n))):        
            if bin_vals == None:
                bin_low_val_ = low_val
                bin_upper_val_ = bin_low_val_ + bin_width
            else:
                bin_low_val_ = bin_vals[i][0]
                bin_upper_val_ = bin_vals[i][1]
            bin_low_val.append(bin_low_val_)
            bin_max_val.append(bin_upper_val_)
            low_val = bin_upper_val_
            val = 0
            for each in var:
                if each >= bin_low_val_ and each < bin_upper_val_:
                    val = val + 1
            observed_freq.append(val)
            expected_val = (scipy.stats.norm(mean, stdev).cdf(bin_upper_val_) - scipy.stats.norm(mean, stdev).cdf(bin_low_val_)) * N
            expected_freq.append(expected_val)
        obs_exp_diff = []
        for i in range(0,len(observed_freq)):
            obs_exp_diff.append((observed_freq[i]-expected_freq[i])**2/expected_freq[i])
        df = pd.DataFrame({'Bin_id':range(0,len(bin_low_val)),'Lower_bin_val':bin_low_val,'Upper_bin_val':bin_max_val,'Observed_freq':observed_freq,'Expected_freq':expected_freq,'(Obs-Exp)^2/Exp':obs_exp_diff})
        return sum(obs_exp_diff),df      

    def combine_bins(self,bin_lower,bin_upper,chi_table_bins):
        # Removing the old bins
        chi_table_bins = chi_table_bins[round(chi_table_bins['Lower_bin_val'],2) != round(bin_lower,2)]
        chi_table_bins = chi_table_bins[round(chi_table_bins['Upper_bin_val'],2) != round(bin_upper,2)]
        # Inserting the new bin
        chi_table = chi_table_bins[['Lower_bin_val','Upper_bin_val']]
        df = pd.DataFrame({'Lower_bin_val':[bin_lower],'Upper_bin_val':[bin_upper]})
        new_df = pd.concat([chi_table_bins,df],axis=0,sort=False).sort_values('Lower_bin_val')  
        return new_df