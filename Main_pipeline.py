 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:51:44 2021

@author: fritz
"""
import scipy.io as sio
import os
from EEGs_persistence import *
import time  

def define_subject_dir(i_sub):
    """
    Creates the directory if it doesn't exist
    :param i_sub: subject id
    :return: directory path
    """
    res_dir = "results/subject_" + str(i_sub) + "/"
    if not os.path.exists(res_dir):
        print("create directory:", res_dir)
        os.makedirs(res_dir)
    return res_dir

def load_data(i_sub,space='both'):
    """
    Loads data from electrode space, font space 
    or both for a given subject
    :param i_sub: subject id
    :param space: electrode/font_space
    :return: data,directory path
    """
    subj_dir = define_subject_dir(i_sub)
    raw_data = sio.loadmat('data/dataClean-ICA3-'+str(i_sub)+'-T1.mat')
    
    if space=='electrode_space':
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        return elec_space,subj_dir
    elif space=='font_space':
        font_space=raw_data['ic_data3']
        return font_space,subj_dir
    else:
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        font_space=raw_data['ic_data3']
        return elec_space,font_space,subj_dir

if __name__ == "__main__":    
    #subjects=[25]
    subjects=list(range(26,36))
    for subject in subjects:
        t=time.time()
        
        #elec_space,subj_dir=load_data(subject,space='electrode_space')
        elec_space,font_space,subj_dir=load_data(subject)
        
        print('computing persistence of electrode space of subject',subject)
        elec_space_pers=compute_persistence_from_EEG(elec_space,measure='intensities',subj_dir=subj_dir,space='electrode_space',save=True) #pointcloud shape (432, 50)
        
        print('computing persistence of electrode space (with correlations) of subject',subject)
        elec_space_pers_corr=compute_persistence_from_EEG(elec_space,measure='correlation',subj_dir=subj_dir,space='electrode_space',save=True) #pointcloud shape (432, 50)
        print('computing persistence of font space of subject',subject)
        font_space_pers=compute_persistence_from_EEG(font_space,measure='intensities',subj_dir=subj_dir,space='font_space',save=True)
        print('computing persistence of font space (with correlations) of subject',subject)
        font_space_pers_corr=compute_persistence_from_EEG(font_space,measure='correlation',subj_dir=subj_dir,space='font_space',save=True)
        
        print('plotting and saving data of subject',subject)
        elec_space_descriptor_vector_dic,labels=compute_topological_descriptors(elec_space_pers,subj_dir,space='electrode_space',measure='intensities')
        
        font_space_descriptor_vector_dic,labels=compute_topological_descriptors(font_space_pers,subj_dir,space='font_space',measure='intensities')
        elec_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(elec_space_pers_corr,subj_dir,space='electrode_space',measure='correlation')
        font_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(font_space_pers_corr,subj_dir,space='font_space',measure='correlation')
        
        get_accuracies_per_band(elec_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='electrode_space',measure='intensities')
        
        get_accuracies_per_band(font_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='font_space',measure='intensities')
        get_accuracies_per_band(elec_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='electrode_space',measure='correlation')
        get_accuracies_per_band(font_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='font_space',measure='correlation')
    
           
        print((time.time()-t)/60, 'minuts for subject',subject)
