# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:24:17 2019

@author: Asha
"""

# Visualization functions in Python
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.graphics.mosaicplot import mosaic
from scipy import stats


class Visualizer(object):
    
    # Plots for univariate continuous variables       
    def univariate_cont(self,cont_var,data,bins_=10):    
        sn.set_palette("muted")
        print("Count {}".format(len(data[cont_var])))
        print("Null found {}".format(sum(data[cont_var].isnull())))
        print(data[[cont_var]].describe())
        plt.figure(1)
        plt.subplot(221)
        histogram = plt.hist(data[cont_var],bins=bins_)
        histogram
        plt.subplot(222)
        distplot = sn.distplot(data[cont_var])
        distplot
        plt.subplot(223)
        box = sn.boxplot(data[cont_var])    
        box
        plt.subplot(224)
        probplot = stats.probplot(data[cont_var],plot=plt)
        probplot
        
    # Plots for univariate categorical variables
    def univariate_categ(self,categ_var,df):
        sn.set_palette("muted")
        print(data[[categ_var]].describe())
        print(df.groupby(categ_var).count())
        
    # Plots for bivariate categorical - continuous variable    
    def bivariate_categ_cont(self,categ_var,cont_var,df):
        qualitative_colors = sn.color_palette("muted")
        print(df.groupby(categ_var)[cont_var].median())
        plt.figure(1)    
        plt.subplot(221)
        sn.barplot(x=categ_var,y=cont_var,data=df)
        plt.subplot(222)
        sn.boxplot(x=categ_var,y=cont_var,data=df)
        plt.subplot(223)
        i = 0
        for val in set(list(df[categ_var])):
            sn.distplot(df[df[categ_var]==val][cont_var],color=qualitative_colors[i],label=val)    
            i=i+1

    # Plots for bivariate categorical - categorical variable
    def bivariate_categ_categ(self,categ_var1,categ_var2,df):
        mosaic(df,[categ_var1,categ_var2])

    # Plots for bivariate continuous - continuous variable
    def bivariate_cont_cont(self,cont_var1,cont_var2,df,legend=None):
        sn.set_palette("muted")
        plt.figure(1)
        plt.subplot(221)
        if legend:
            plt.scatter(x=df[cont_var1],y=df[cont_var2],hue=df[legend])
        else:
            plt.scatter(x=df[cont_var1],y=df[cont_var2])
        plt.subplot(222)
        sn.regplot(x=cont_var1,y=cont_var2,data=df)

    # Plots for multiple continuous - continuous variables as a list
    def bivariate_contlist_contlist(self,cont_var_list,df,legend=None):
        sn.set_palette("muted")
        plt.figure(1)
        if legend:
            sn.pairplot(df[cont_var_list],hue=df[legend],height=2)
        else:
            sn.pairplot(df[cont_var_list],height=2)
        plt.figure(2)
        print(df[cont_var_list].corr())
        plt.figure(3)
        
        ax = sn.heatmap(
            df[cont_var_list].corr(), 
            vmin=-1, vmax=1, center=0,
            cmap=sn.diverging_palette(20, 220, n=200),
            square=True )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
            );

    # Plots for 1 continuous and 2 categorical variables  
    def multivariate_1cont_2categ(self,cont_var,categ_var1,categ_var2,df):
        sn.set_palette("muted")
        with sns.axes_style(style='ticks'):
            g = sns.factorplot(categ_var1, cont_var, categ_var2, data=df, kind="box")
            g.set_axis_labels(categ_var1, cont_var);
