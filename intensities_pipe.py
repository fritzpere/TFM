#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:39:25 2021

@author: fritzpere
"""
import numpy as np
import sklearn.model_selection as skms
from TDApipeline import *
from collections import defaultdict
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import time
import os



def intensity(subj_dir,space,ts_band,labels,i_band):
    dimensions=["zero","one"]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    n_vectors=len(feat_vect)
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    ts=np.abs(ts_band[:,i_band,:,:])
    ts_intens=ts.mean(axis=1)
    dim_persistence=[]
    X_motiv=[]
    tda_vect=defaultdict(lambda: defaultdict(lambda: []))
    
    n_rep=10
    
    perf = np.zeros([n_dim+1,n_vectors+1,n_rep])
    perf_shuf = np.zeros([n_dim+1,n_vectors+1,n_rep])
    conf_matrix = np.zeros([n_dim+1,n_vectors+1,n_rep,3,3])
    
    if not os.path.exists(subj_dir+space+'/intensities'):
                print("create directory(plot):",subj_dir+space+'/intensities')
                os.makedirs(subj_dir+space+'/intensities')
    
    t_int=time.time()
    
    

    for i_rep in range(n_rep):
        t_rep=time.time()
        for ind_train, ind_test in cv_schem.split(ts_intens,labels):
            
            X_train=ts_intens[ind_train]
            y_train=labels[ind_train]
            pred=np.zeros(len(ind_train))
            pred_array=np.zeros((len(ind_test),n_vectors+1,n_dim,3))
            for i_motiv  in range(3):
                X_motiv.append(X_train[y_train==i_motiv])
                n_coor=X_motiv[i_motiv].shape[0]
                matrix = np.zeros((n_coor, n_coor))
                row,col = np.triu_indices(n_coor,1)
                distancies=pdist(X_motiv[i_motiv])
                matrix[row,col] = distancies
                matrix[col,row] = distancies
        
                Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                persistence=Rips_simplex_tree_sample.persistence()
                dim_list=np.array(list(map(lambda x: x[0], persistence)))
                point_list=np.array(list(map(lambda x: x[1], persistence)))
                zero_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)]
                one_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)]
                persistence=(zero_dim,one_dim)
                for i_dim in range(n_dim):
                    dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                    dimensionscaler.fit(persistence)
                    dim_persistence=np.array(dimensionscaler.transform(persistence))
                    for i_vector in range(n_vectors):
                        tda_compt=feat_vect[i_vector]
                        tda_compt.fit([dim_persistence])
                        tda_vect[i_vector][i_dim]=tda_compt.transform([dim_persistence])
                    tda_vect[n_vectors][i_dim]=dim_persistence
            i=0
            for index in ind_test:
                for i_motiv  in range(3):
                    X_temp=np.concatenate((X_motiv[i_motiv],ts_intens[index].reshape(1,-1)),axis=0)
                    n_coor=X_temp.shape[0]
                    matrix = np.zeros((n_coor, n_coor))
                    row,col = np.triu_indices(n_coor,1)
                    distancies=pdist(X_temp)
                    matrix[row,col] = distancies
                    matrix[col,row] = distancies
            
                    Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                    #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                    Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                    persistence=Rips_simplex_tree_sample.persistence()
                    dim_list=np.array(list(map(lambda x: x[0], persistence)))
                    point_list=np.array(list(map(lambda x: x[1], persistence)))
                    zero_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)]
                    one_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)]
                    persistence=(zero_dim,one_dim)
                    for i_dim in range(n_dim):
                        dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                        dimensionscaler.fit(persistence)
                        dimensional_persistence=np.array(dimensionscaler.transform(persistence))
                        for i_vector in range(n_vectors):
                            tda_compt=feat_vect[i_vector]
                            tda_compt.fit([dimensional_persistence])
                            
                            pred_array[i,i_vector,i_dim,i_motiv]=np.linalg.norm(tda_compt.transform([dimensional_persistence])-tda_vect[i_vector][i_dim])
                        pred_array[i,n_vectors,i_dim,i_motiv]=gd.bottleneck_distance(dimensional_persistence,tda_vect[n_vectors][i_dim],0.01)
                i=i+1
        
    
            for i_vector in range(n_vectors+1):
                for i_dim in range(n_dim):
                    pred=np.argmin(np.array(pred_array[:,i_vector,i_dim,:]),axis=1)
                    
                    perf[i_dim,i_vector,i_rep] = skm.accuracy_score(pred, labels[ind_test])
                    conf_matrix[i_dim,i_vector,i_rep,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=pred) 
            #print((time.time()-t_rep)/60, 'minuts for classification for repetition',i_rep)
        
        # save results       
    np.save(subj_dir+space+'/intensities/perf_intensity.npy',perf)
    np.save(subj_dir+space+'/intensities/conf_matrix_intensity.npy',conf_matrix)                     
            
      
    fmt_grph = 'png'
    cmapcolours = ['Blues','Greens','Oranges','Reds']
    
    fig, axes = plt.subplots(nrows=n_dim, ncols=1, figsize=(24, 12))
        
    band_dic={-1: 'noFilter', 0:'alpha',1:'betta',2:'gamma'}

    band = band_dic[i_band]
    for i_dim in range(n_dim):
            
        # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
        chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
    
        # plot performance and surrogate
        #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
        axes[i_dim].violinplot(perf[i_dim,0,:],positions=[-0.2],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,1,:],positions=[0.2],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,2,:],positions=[0.6],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,3,:],positions=[1],widths=[0.3])
        #axes[i_dim].violinplot(perf[i_dim,3,:],positions=[1],widths=[0.3])


        
        axes[i_dim].plot([-1,2],[chance_level]*2,'--k')
        axes[i_dim].axis(xmin=-0.6,xmax=1.4,ymin=0,ymax=1.05)

        axes[i_dim].set_ylabel('accuracy '+band,fontsize=8)
        axes[i_dim].set_title(band+dimensions[i_dim])
    plt.savefig(subj_dir+space+'/intensities/accuracies_intensity_'+band+'.png', format=fmt_grph)
    plt.close()


    
    fig2, axes2 = plt.subplots(nrows=1, ncols=n_vectors*n_dim, figsize=(96, 24))
    i=0
    for i_vector in range(n_vectors):
        for i_dim in range(n_dim):
       
            axes2[i].imshow(conf_matrix[i_dim,i_vector,:,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
            #plt.colorbar()
            axes2[i].set_xlabel('true label',fontsize=8)
            axes2[i].set_ylabel('predicted label',fontsize=8)
            axes2[i].set_title(band+dimensions[i_dim]+str(i_vector))
            i+=1
    fig.tight_layout(pad=0.5)
    plt.savefig(subj_dir+space+'/intensities/confusion_matrix_intensities_'+band+'.png', format=fmt_grph)
    plt.close()
    
    
    print('======TIME======') 
    print((time.time()-t_int)/60, 'minuts for classification w intensities for band',band)
    with open('control.txt', 'w') as f:
        print((time.time()-t_int)/60, 'minuts for classification w intensities for band',band, file=f)
    return                  
