#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:22:15 2021

@author: fritz
"""


import pandas as pd
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
from classification import *




def plot_descriptorBoxplots(feature, name,subj_dir,space,measure):
    feat0,feat1=feature
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(19, 14))
    for i in range(-1,3):

        axes[0][i].boxplot(feat0[i],showfliers=False)
        axes[0][i].set_title(name+' BoxPlot dimension 0 of band '+band_dic[i])
        axes[1][i].boxplot(feat1[i],showfliers=False)
        axes[1][i].set_title(name+' BoxPlot dimension 1 of band '+band_dic[i])
        #a.set_xlim(-0.05,x_max)
        #a.set_ylim(0,y_max)
    fig.suptitle('{0} Boxplots of the {1} for\n different frequency bands and motivational state'.format(name,space),fontsize=24)
    fig.tight_layout(pad=1.00)
    fig.subplots_adjust(top=0.8)
    
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+space+'/'+measure+'/'+'descriptor_tables')
    
    plt.savefig(subj_dir+space+'/'+measure+'/descriptor_tables/'+name+'_boxplots''.png')
    plt.close()
    
    
    


