# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:24:17 2019

@author: Asha
"""
import math
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
from sklearn.metrics import matthews_corrcoef
        
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
        return({'mean':meandf,'median':mediandf,'mode':modedf,'std':stddf,'kurtosis':kurtdf,'skew':skewdf,'min':mindf,'max':maxdf})
        
    def descriptive_stats_categ(self,colname):
        print(self.df.groupby(colname)[colname].count())
    
    def check_missing(self,colname=None):
        if colname is None:
            missing = len(self.df[self.df.isnull().any(axis=1)])
        else:
            missing = len(self.df[self.df[colname].isnull()])
        complete = len(self.df) - missing
        print("Total missing",missing)
        print("Total complete cases",complete)
        
    def check_val(self,colname,val):        
        if not isinstance(val,list):
            count = len(self.df[self.df[colname] == val])
            print("Count is",count)
            return count
        else:
            count = len(self.df[self.df[colname].isin(val)])
            print("Count is",count)
            return count                       
    
    def pearson_corr_cont(self,col1,col2):
        # Gaussian distribution
        corr, _ = pearsonr(self.df[col1], self.df[col2])
        print(corr)
        return corr
    
    def spearman_corr_cont(self,col1,col2):
        #Non - parametric method
        corr, _ = spearmanr(self.df[col1], self.df[col2])
        print(corr)
        return corr
    
    def data_frame_corr(self,cols,method="pearson"):
        # Method = pearson, spearman and kendall (non parametric)
        print(self.df[[cols]].corr(method=method))
        
    def point_biserial(self,col1,col2binary):
        corr = pointbiserialr(self.df[col1], self.df[col2binary])
        print(corr)
        
    def phi_coff(self,col1binary,col2binary):
        corr = matthews_corrcoef(self.df[col1binary],self.df[col2binary])
        print(corr)
            

class DataPreprocessing(object):
    df = pd.DataFrame()
    
    def __init__(self,df):
        self.df = df
        
    def remove_missing(self,colname = None):
        if colname is None:
            self.df = self.df.dropna()
            return self.df
        else:
            self.df = self.df[self.df[colname].notnull()]
            return self.df
        
    def remove_val(self,colname,val):
        if not isinstance(val,list):        
            self.df = self.df[self.df[colname] != val]
            return self.df
        else:
            self.df = self.df[~self.df[colname].isin(val)]
            return self.df              
           
    def select_vals(self,colname,vals):
        self.df = self.df[self.df[colname].isin(vals)]
        return self.df
    
    def mark_missing(self,colname,val):
        # Mark values as missing
        self.df[colname] = self.df[colname].replace(val, np.NaN)
        return self.df
    
    def impute_missing(self,colname,method="mean",n=3):
        if method == "mean":
            self.df[colname].fillna(self.df[colname].mean(), inplace=True)
            return self.df
        if method == "median":
            self.df[colname].fillna(self.df[colname].median(), inplace=True)
            return self.df            
        if method == "most_frequent":
            imp = SimpleImputer(strategy="most_frequent")
            self.df[colname] = imp.fit_transform(self.df[[colname]])
            return self.df
        if method == "knn":
            from fancyimpute import KNN              
            self.df[colname] = KNN(k=n).fit_transform(self.df[colname])
            return self.df
        
    def dummy_categ(self,colname):
        self.df[colname] = self.df[colname].astype('category')
        self.df[colname + "_cat"] = self.df[colname].cat.codes
        return self.df
    
    def one_hot_encoder(self,colname,exclude_one=False):
        x = pd.get_dummies(self.df[colname])
        colnames = list(x)
        new_colnames = [colname+"_"+each for each in colnames]
        new_col_dict = {colnames[i]:new_colnames[i] for i in range(0,len(colnames))}
        x = x.rename(columns = new_col_dict)
        
        if exclude_one:
            x.drop(x.columns[len(x.columns)-1], axis=1, inplace=True)
            self.df = pd.concat([self.df,x],axis=1,sort=False)
        else:
            self.df = pd.concat([self.df,x],axis=1,sort=False)
        return self.df
    
    def binary_encoding(self,colname,valueif1):        
        x = self.df[colname].str.contains(valueif1)
        self.df.loc[x,colname] = 1
        self.df.loc[~x,colname] = 0
        return self.df
    
    def bin_continuous(self,colname,equal=True,n_bins=5,custom_bins=None,labels=None):
        if equal:
            if labels is None:
                labels = list(range(0,n_bins))
            self.df[colname + '_binned'] = pd.cut(self.df[colname], bins=n_bins, labels=labels)
        else:        
            self.df[colname + '_binned'] = pd.cut(self.df[colname], bins=custom_bins, labels=labels)
        
    def centering(self,colname):
        x_mean = self.df[colname].mean()
        self.df.loc[:,colname + "_centered"] = self.df[colname] - x_mean
        return self.df
    
    def standard_scaling(self,colname):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.df.loc[:,colname+ "_scaled"] = scaler.fit_transform(self.df[[colname]])
        return self.df
    
    def interaction_var_mult(self,col1,col2):
        self.df[col1+"*"+col2] = self.df[col1] * self.df[col2]
        return self.df                
            
    def non_linear_transformations(self,colname,transform="square"):
        # Supported Transforms - square, cube, square root, cube root, ln,log
        if transform == "square":
            self.df[colname+"_sq"] = self.df[colname]**2
        if transform == "cube":
            self.df[colname+"_cube"] = self.df[colname]**3
        if transform == "square root":
            self.df[colname+"_sqrt"] = self.df[colname]**(1/2)
        if transform == "cube root":
            self.df[colname+"_cubert"] = self.df[colname]**(1/3)            
        if transform == "ln":
            self.df[colname+"_ln"] = np.log(self.df[colname])
        if transform == "log":
            self.df[colname+"_log"] = np.log10(self.df[colname])
        return self.df
            

