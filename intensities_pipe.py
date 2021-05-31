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





def intensity(ts_band,labels,band):
    dimensions=["zero","one"]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    n_vectors=len(feat_vect)
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    ts=np.abs(ts_band[:,band,:,:])
    ts_intens=ts.mean(axis=1)
    dim_persistence=[]
    X_motiv=[]
    tda_vect=defaultdict(lambda: defaultdict(lambda: []))
    
    i_rep=0
    n_rep=1
    
    perf = np.zeros([n_dim+1,n_vectors+1,n_rep])
    perf_shuf = np.zeros([n_dim+1,n_vectors+1,n_rep])
    conf_matrix = np.zeros([n_dim+1,n_vectors+1,n_rep,3,3])
    
  
    for ind_train, ind_test in cv_schem.split(ts_intens,labels):
        X_train=ts_intens[ind_train]
        y_train=labels[ind_train]
        pred=np.zeros(len(ind_train))
        pred_array=np.zeros((len(ind_test),3,2,3))
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
            i=i+1
    

        for i_vector in range(n_vectors):
            for i_dim in range(n_dim):
                pred=np.argmin(np.array(pred_array[:,i_vector,i_dim,:]),axis=1)
                perf[i_dim,i_vector+1,i_rep] = skm.accuracy_score(pred, labels[ind_test])
                conf_matrix[i_dim,i_vector+1,i_rep,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=pred) 
    
        '''
        pred=##bottleneck 1nn
        perf[i_dim,i_vector+1,i_rep] = #bottleneck 1nn
        conf_matrix[i_dim,i_vector+1,i_rep,:,:] += #bottleneck 1nn'''
        
        # save results       
        np.save(subj_dir+space+'/perf.npy',perf)
        np.save(subj_dir+space+'/conf_matrix.npy',conf_matrix)                     
                
                                
    