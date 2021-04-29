#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:05:26 2021

@author: fritzpere
"""
import numpy as np
import sklearn.pipeline as skppl
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skprp

import matplotlib.pyplot as plt
import os

def feature_vector_per_band(avg_life, std_life, entropy, pooling, avg_midlife, std_midlife):
    
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    feat_vect_size=8
    feat_vect={}
    labels=[]

    for i_band in range(-1,3):
        feat_vect[band_dic[i_band]]={}
        trials=0
        for i_state in range(3):
            trials+=len(avg_life[0][i_band][i_state])
        feat_vect[band_dic[i_band]]['dim0']=np.zeros((trials,feat_vect_size))
        feat_vect[band_dic[i_band]]['dim1']=np.zeros((trials,feat_vect_size))
        feat_vect[band_dic[i_band]]['dim0dim1']=np.zeros((trials,feat_vect_size*2))
        
        feat_vect[band_dic[i_band]]['dim0pooling']=np.zeros((trials,10))
        feat_vect[band_dic[i_band]]['dim1pooling']=np.zeros((trials,10))
        feat_vect[band_dic[i_band]]['dim0dim1pooling']=np.zeros((trials,10*2))
    for i_band in range(-1,3):
        for i_state in range(3):
            trials=len(avg_life[0][i_band][i_state])
            for k in range(trials):
                feat_vect[band_dic[i_band]]['dim0'][k]=np.concatenate((np.array([avg_life[0][i_band][i_state][k],std_life[0][i_band][i_state][k],avg_midlife[0][i_band][i_state][k],std_midlife[0][i_band][i_state][k],entropy[0][i_band][i_state][k]]),pooling[0][i_band][i_state][k][:3]),axis=0)
                feat_vect[band_dic[i_band]]['dim1'][k]=np.concatenate((np.array([avg_life[1][i_band][i_state][k],std_life[1][i_band][i_state][k],avg_midlife[1][i_band][i_state][k],std_midlife[1][i_band][i_state][k],entropy[1][i_band][i_state][k]]),pooling[1][i_band][i_state][k][:3]),axis=0)
                feat_vect[band_dic[i_band]]['dim0dim1'][k]=np.concatenate((feat_vect[band_dic[i_band]]['dim0'][k],feat_vect[band_dic[i_band]]['dim1'][k]),axis=0)
                
                feat_vect[band_dic[i_band]]['dim0pooling'][k]=pooling[0][i_band][i_state][k]
                feat_vect[band_dic[i_band]]['dim1pooling'][k]=pooling[1][i_band][i_state][k]
                feat_vect[band_dic[i_band]]['dim0dim1pooling'][k]=np.concatenate((feat_vect[band_dic[i_band]]['dim0pooling'][k],feat_vect[band_dic[i_band]]['dim1pooling'][k]),axis=0)
                
                if i_band==-1 : #to only do it once
                    labels.append(i_state)
    return feat_vect,labels




def get_accuracies_per_band(feature_vector_dic,labels,subj_dir,space,measure):
    c_MLR = skppl.Pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])   
    c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')          
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    n_rep = 15 # number of repetitions
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    labels=np.array(labels)
    
    feat_vectors=[*feature_vector_dic['no_filter']]
    '''
    # RFE wrappers
    RFE_pow = skfs.RFE(c_MLR, n_features_to_select=3)
    RFE_FC = skfs.RFE(c_MLR, n_features_to_select=90)'''
    
    # record classification performance 
    n_vector=len(feat_vectors) 
    perf = np.zeros([4,n_vector,n_rep,2]) # (last index: MLR/1NN)
    perf_shuf = np.zeros([4,n_vector,n_rep,2]) # (last index: MLR/1NN)
    conf_matrix = np.zeros([4,n_vector,n_rep,2,3,3]) # (fourthindex: MLR/1NN)
    #rk_pow = np.zeros([n_bands,n_rep,N],dtype=np.int) # RFE rankings for power (N feature)
    #rk_FC = np.zeros([n_bands,2,n_rep,int(N*(N-1)/2)],dtype=np.int) # RFE rankings for FC-type measures (N(N-1)/2 feature)
    #pearson_corr_rk = np.zeros([n_bands,n_measures,int(n_rep*(n_rep-1)/2)]) # stability of rankings measured by Pearson correlation
    
    # repeat classification for several splits for indices of sliding windows (train/test sets)
    
    for i_band in range (-1,3):
        for i_vector in range(n_vector):
            vect_features=feature_vector_dic[band_dic[i_band]][feat_vectors[i_vector]]    
            for i_rep in range(n_rep):
                for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
                    # train and test for original data
                    c_MLR.fit(vect_features[ind_train,:], labels[ind_train])
                    perf[i_band,i_vector,i_rep,0] = c_MLR.score(vect_features[ind_test,:], labels[ind_test])
                    conf_matrix[i_band,i_vector,i_rep,0,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test,:]))  
                
                    c_1NN.fit(vect_features[ind_train,:], labels[ind_train])
                    perf[i_band,i_vector,i_rep,1] = c_1NN.score(vect_features[ind_test,:], labels[ind_test])
                    conf_matrix[i_band,i_vector,i_rep,1,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_1NN.predict(vect_features[ind_test,:]))  
                
                    # shuffled performance distributions
                    shuf_labels = np.random.permutation(labels)
            
                    c_MLR.fit(vect_features[ind_train,:], shuf_labels[ind_train])
                    perf_shuf[i_band,i_vector,i_rep,0] = c_MLR.score(vect_features[ind_test,:], shuf_labels[ind_test])
            
                    c_1NN.fit(vect_features[ind_train,:], shuf_labels[ind_train])
                    perf_shuf[i_band,i_vector,i_rep,1] = c_1NN.score(vect_features[ind_test,:], shuf_labels[ind_test])

    

    if not os.path.exists(subj_dir+space+'/'+measure+'/acc'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/acc')
        os.makedirs(subj_dir+space+'/'+measure+'/acc')
    if not os.path.exists(subj_dir+space+'/'+measure+'/conf_matrix'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/conf_matrix')
        os.makedirs(subj_dir+space+'/'+measure+'/conf_matrix')
    
    
    # save results       
    np.save(subj_dir+space+'/'+measure+'/perf.npy',perf)
    np.save(subj_dir+space+'/'+measure+'/perf_shuf.npy',perf_shuf)
    np.save(subj_dir+space+'/'+measure+'/conf_matrix.npy',conf_matrix)

    fmt_grph = 'png'
    cmapcolours = ['Blues','Greens','Oranges']
    
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 24))

    for i_band in range(-1,3):
        band = band_dic[i_band]
        for i_vector in range(n_vector):
            measure_label=feat_vectors[i_vector]
    
            # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
            chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
        
            # plot performance and surrogate
            #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
            axes[i_band][i_vector].violinplot(perf[i_band,i_vector,:,0],positions=[-0.2],widths=[0.3])
            axes[i_band][i_vector].violinplot(perf[i_band,i_vector,:,1],positions=[0.2],widths=[0.3])
            axes[i_band][i_vector].violinplot(perf_shuf[i_band,i_vector,:,0],positions=[0.8],widths=[0.3])
            axes[i_band][i_vector].violinplot(perf_shuf[i_band,i_vector,:,1],positions=[1.2],widths=[0.3])
            axes[i_band][i_vector].plot([-1,2],[chance_level]*2,'--k')
            axes[i_band][i_vector].axis(xmin=-0.6,xmax=1.6,ymin=0,ymax=1.05)
            #plt.xticks([0,1],['Pearson Correlation','surrogate'],fontsize=8)
            axes[i_band][i_vector].set_ylabel('accuracy_'+band+'_'+str(i_vector),fontsize=8)
            axes[i_band][i_vector].set_title(band+', '+measure_label)
    plt.savefig(subj_dir+space+'/'+measure+'/acc/accuracies.png', format=fmt_grph)
    plt.close()
    
    fig2, axes2 = plt.subplots(nrows=4, ncols=6, figsize=(24, 24))

    for i_band in range(-1,3):
        band = band_dic[i_band]
        for i_vector in range(n_vector):
            measure_label=feat_vectors[i_vector]
    
            # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
            chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
        
            # plot performance and surrogate
            #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
            axes2[i_band][i_vector].imshow(conf_matrix[i_band,i_vector,:,0,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
            #plt.colorbar()
            axes2[i_band][i_vector].set_xlabel('true label',fontsize=8)
            axes2[i_band][i_vector].set_ylabel('predicted label',fontsize=8)
            axes2[i_band][i_vector].set_title(band+', '+measure_label)
    plt.savefig(subj_dir+space+'/'+measure+'/conf_matrix/confusion_matrix_MLR.png', format=fmt_grph)
    plt.close()

