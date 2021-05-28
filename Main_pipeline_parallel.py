 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:51:44 2021

@author: fritz
"""
import sys
import scipy.io as sio
import os
#from EEGs_persistence import *
from preprocess_data import *
from TDApipeline import *
import time  
import sklearn.pipeline as skppl
import sklearn.linear_model as skllm
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprp
import pandas as pd
import sklearn.neighbors as sklnn
from joblib import Memory
from shutil import rmtree

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



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
    
    if space=='electrodeSpace':
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        return elec_space,subj_dir
    elif space=='font_space':
        font_space=raw_data['ic_data3']
        return font_space,subj_dir
    else:
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        font_space=raw_data['ic_data3']
        return elec_space,font_space,subj_dir
    


# This is the main
    
# Deal with the arguments
'''
if(len(sys.argv) == 5):      
    i_band = int(sys.argv[1])
    i_measure = int(sys.argv[2])
    i_dim = int(sys.argv[3])
    i_vector = int(sys.argv[4])
                      
    
# BAD NUM OF ARGUMENTS
else:
    # END THE PROGRAM
    print(len(sys.argv))
    sys.exit('EXIT! INCORRECT NUMBER OF ARGUMENTS')
'''

##DEBUUUUGEAR

debug=True    
if debug: 
    
    subject=25
    space='electrodeSpace'

    elec_space,subj_dir=load_data(subject,space=space)
    
    if not os.path.exists(subj_dir+space):
        print("create directory(plot):",subj_dir+space)
        os.makedirs(subj_dir+'/'+space)
    
    print('cleaning and filtering data of electrode space of subject',subject)
    elec_space_preprocessor=Preprocessor(elec_space)
    filtered_ts_dic=elec_space_preprocessor.get_filtered_ts_dic()
    ts_band,labels=elec_space_preprocessor.get_trials_and_labels()
    
    band_dic={-1: 'noFilter', 0:'alpha',1:'betta',2:'gamma'}
    
    
    
    
    bands=[-1,0,1,2]
    #bands=[-1,2]
    n_band=len(bands)
    measures=["intensities","correlation","quaf","dtw"]
    #measures=["quaf","dtw"]
    n_measure=len(measures)
    dimensions=["zero","one"]
    #dimensions=[]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]

    n_vectors=len(feat_vect)
    
    
    resolut=1000

    
#descomentar lo d-adalt
    
   
    dimensions.append('both')
    n_dim+=1
    classifiers=[skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000))]),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')  ]
    n_classifiers=len(classifiers)
    
    
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    n_rep = 10 # number of repetitions
    
    perf = np.zeros([n_rep,n_classifiers]) # (last index: MLR/1NN)
    perf_shuf = np.zeros([n_rep,n_classifiers])# (last index: MLR/1NN)
    conf_matrix = np.zeros([n_rep,n_classifiers,3,3]) # (fourthindex: MLR/1NN)
                              
    # start parallelization
    for i_classifier in range(n_classifiers):
        
        topo_pipe= skppl.Pipeline([("band_election", Band_election(bands[i_band])),("persistence", PH_computer(measure=measures[i_measure])),("scaler",DimensionDiagramScaler(dimensions=dimensions[i_dim])),("TDA",feat_vect[i_vector]),('clf',classifiers[i_classifier])])
  
        for i_rep in range(n_rep):
            for ind_train, ind_test in cv_schem.split(ts_band,labels): # false loop, just 1 
                print('band',bands[i_band],'measure',measures[i_measure],'dim',dimensions[i_dim],'vector',i_vector,'classifier',i_classifier,'repetition:',i_rep)

                topo_pipe.fit(ts_band[ind_train,:], labels[ind_train])
            
                perf[i_rep,i_classifier] = topo_pipe.score(ts_band[ind_test,:], labels[ind_test])
                conf_matrix[i_rep,i_classifier,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=topo_pipe.predict(ts_band[ind_test,:]))  
                
                
                shuf_labels = np.random.permutation(labels)

                topo_pipe.fit(ts_band[ind_train,:], shuf_labels[ind_train])
                perf_shuf[i_rep,i_classifier]= topo_pipe.score(ts_band[ind_test,:], shuf_labels[ind_test])
                       


    # save results       
    np.save(subj_dir+space+'/perf'+str(i_band)+'_'+str(i_measure)+'_'+str(i_dim)+'_'+str(i_vector)+'.npy',perf)
    np.save(subj_dir+space+'/perf_shuf'+str(i_band)+'_'+str(i_measure)+'_'+str(i_dim)+'_'+str(i_vector)+'.npy',perf_shuf)
    np.save(subj_dir+space+'/conf_matrix'+str(i_band)+'_'+str(i_measure)+'_'+str(i_dim)+'_'+str(i_vector)+'.npy',conf_matrix)                     
    
    
        
