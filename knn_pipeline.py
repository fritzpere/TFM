#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:48:11 2021

@author: fritzpere
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sklearn.neighbors as sklnn
import sklearn.model_selection as skms
import sklearn.metrics as skm




def knn_intensity_classifier(subj_dir,space,PC,labels,i_band):

    n_rep=10 
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'}
    band = band_dic[i_band]
    #Initiialize matrices where we will save several information (accuracies distribution, confusion matrix, random predictions matrix)


    perf = np.zeros(n_rep)
    conf_matrix =np.zeros((n_rep,3,3))
    #Initialize 1 Nearest Neighbor classifier
    clf= sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    if not os.path.exists(subj_dir+space+'/1nn_clf'):
        print("create directory(plot):",subj_dir+space+'/1nn_clf')
        os.makedirs(subj_dir+space+'/1nn_clf')
    
    t_1nn=time.time()
    
    trials_per_m=min((labels==0).sum(),(labels==1).sum(),(labels==2).sum())
    if trials_per_m==0: #If there is a motivational state without a point we will not classify
        np.save(subj_dir+space+'/1nn_clf/'+band+'perf_intensity.npy',perf)  
        return

    
    #We begin the classificatino
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for i_rep in range(n_rep):   
       
        for ind_train, ind_test in cv_schem.split(PC,labels):
            #Save test size, define X_train and y_train and initialize prediction matrix

            X_train=PC[ind_train]
            y_train=labels[ind_train]
            pred=np.zeros(len(ind_train))
            clf.fit(X_train,y_train)
            pred=clf.predict(PC[ind_test])
            
            perf[i_rep]=skm.accuracy_score(pred, labels[ind_test])
            conf_matrix[i_rep,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=pred,normalize='true')               
    print((time.time()-t_1nn)/60, 'minuts for classification')
    #We plot accuracies and confusion matrices

    np.save(subj_dir+space+'/1nn_clf/'+band+'perf_intensity.npy',perf)  
    np.save(subj_dir+space+'/1nn_clf/'+band+'conf_matrix_intensity.npy',conf_matrix)
    fmt_grph = 'png'
    cmapcolours = ['Blues','Greens','Oranges','Reds']
    plt.rcParams['xtick.labelsize']=16 
    plt.rcParams['ytick.labelsize']=8
    plt.figure(figsize=[16,9])

    plt.violinplot(perf)
    chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
    #plt.plot([-1,2],[chance_level]*2,'--k')

    plt.ylabel('accuracy_'+band,fontsize=8)
    plt.title(band+' 1nn classification')
    plt.yticks([0, 0.2,0.4, 0.6,0.8,1])
    
    plt.savefig(subj_dir+space+'/1nn_clf/1nn_accuracies_intensity_'+band+'.png', format=fmt_grph)
    plt.close()
    plt.rcParams['xtick.labelsize']=24
    plt.rcParams['ytick.labelsize']=24
    plt.rcParams.update({'font.size': 24})
    
    plt.figure(figsize=[16,9])


    disp = skm.ConfusionMatrixDisplay(conf_matrix[:,:,:].mean(0),display_labels=['M0','M1','M2'])
    disp.plot(include_values=True,cmap=cmapcolours[i_band],colorbar=True)
    
    plt.xlabel('true label',fontsize=12)
    plt.ylabel('predicted label',fontsize=12)
    plt.title('Confusion Matix for band '+band+' and a 1NN classifier',fontsize=18)


    plt.savefig(subj_dir+space+'/1nn_clf/1nn_confusion_matrix_intensities_'+band+'.png', format=fmt_grph)
    plt.close()
    return 