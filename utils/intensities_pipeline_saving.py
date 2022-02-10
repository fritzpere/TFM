#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:39:25 2021

@author: fritzpere
"""
import numpy as np
import sklearn.model_selection as skms
from utils.TDApipeline import *
from collections import defaultdict
import sklearn.metrics as skm
import sklearn.neighbors as sklnn
import matplotlib.pyplot as plt
import time
import os



def topological_clf(arr):
    """
    Classifies the trials in a nearest neighbor fasihion, se select the motivational state whose topology ahs changed less.
    :param arr: array of distances to each motivational state
    :return: predictions, the number of random selections
    """
    n=len(arr)
    pred=-np.ones(n)
    not_random=(arr==0).sum(axis=1)<=1
    pred[not_random]=np.argmin((arr[not_random]),axis=1) #We select the motivational state whose topology ahs changed less
    random_selections=n-not_random.sum() #We save the number of trials we can't classify (if the topology of 2 or 3 motivational states doesn't change)
    pred[~not_random]=[np.random.choice(np.array(list(range(3)))[arr[~not_random][i]==0],1).item() for i in range(random_selections)]
    return pred,(random_selections/pred.shape[0])



def tda_intensity_classifier2(subj_dir,space,PC,labels,i_band, n_repet=10):
    """
    Pipeline of a Topological Classifier
    
    :param subj_dir: Directory of the subject where we will save the accuracies and Confusion Matrix
    :param space: If electrode space or font space
    :param PC: Point Cloud we will classify
    :param labels: labels of the points
    :param i_band: frequancy band
    :return: test size,random selections matrix, accuracy of dimension 0 silhouettes
    """
    
    #We define the dimensions and the feature vectors we will use, as well as the frequency band, and define the number of times we will repeat the classification
    dimensions=["zero"]
    n_dim=len(dimensions)
    feat_vect=[DimensionSilhouette()]
    feat_vect_names=['Silhouettes']
    n_vectors=len(feat_vect)
    n_rep=n_repet 
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'}
    band = band_dic[i_band]
    #Initiialize matrices where we will save several information (accuracies distribution, confusion matrix, random predictions matrix)
    rand_n=np.zeros((n_rep,n_vectors+1,n_dim))
    test_size=np.zeros(n_rep)
    topo_perf = np.zeros([n_dim,n_vectors+1,n_rep])
    
    knn_perf = np.zeros(n_rep)
    knn_conf_matrix =np.zeros((n_rep,3,3))
    #Initialize 1 Nearest Neighbor classifier
    clf= sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    if not os.path.exists(subj_dir):
        print("create directory(plot):",subj_dir)
        os.makedirs(subj_dir)
    #perf_shuf = np.zeros([n_dim,n_vectors+1,n_rep])
    topo_conf_matrix = np.zeros([n_dim,n_vectors+1,n_rep,3,3])

    
    t_int=time.time()
    #We lool which motivational state has less points
    trials_per_m=min((labels==0).sum(),(labels==1).sum(),(labels==2).sum())
    if trials_per_m==0: #If there is a motivational state without a point we will not classify
        np.save(subj_dir+space+'/topological_clf/'+band+'perf_intensity.npy',topo_perf)
        np.save(subj_dir+space+'/1nn_clf/'+band+'perf_intensity.npy',knn_perf) 
        return -1,np.zeros((n_vectors+1,n_dim)),-1
    

    #We balabce the dataset by downsampling

    
    #We begin the classificatino
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for i_rep in range(n_rep): 
        X_m0_dwnsamp=PC[labels==0][np.random.choice(len(PC[labels==0]),trials_per_m)]
        X_m1_dwnsamp=PC[labels==1][np.random.choice(len(PC[labels==1]),trials_per_m)]
        X_m2_dwnsamp=PC[labels==2][np.random.choice(len(PC[labels==2]),trials_per_m)]
        PC_dwnsamp=np.concatenate((X_m0_dwnsamp,X_m1_dwnsamp,X_m2_dwnsamp),axis=0)
        labels_dwnsamp=np.concatenate((np.zeros(trials_per_m),np.ones(trials_per_m),np.ones(trials_per_m)*2))
        X_motiv=[]
        tda_vect={0:defaultdict(lambda: defaultdict(lambda: [])),1:defaultdict(lambda: defaultdict(lambda: [])),2:defaultdict(lambda: defaultdict(lambda: []))}
        for ind_train, ind_test in cv_schem.split(PC_dwnsamp,labels_dwnsamp):
            #Save test size, define X_train and y_train and initialize prediction matrix
            test_size[i_rep]=len(ind_test)
            X_train=PC_dwnsamp[ind_train]
            y_train=labels_dwnsamp[ind_train]
            #1nn
            knn_pred=np.zeros(len(ind_train))
            clf.fit(X_train,y_train)
            knn_pred=clf.predict(PC_dwnsamp[ind_test])
            knn_perf[i_rep]=skm.accuracy_score(knn_pred, labels_dwnsamp[ind_test])
            knn_conf_matrix[i_rep,:,:] += skm.confusion_matrix(y_true=labels_dwnsamp[ind_test], y_pred=knn_pred,normalize='true')  
            #topological classifier
            topo_pred=np.zeros(len(ind_train))
            topo_pred_array=np.zeros((len(ind_test),n_vectors+1,n_dim,3))
            #For each motivational state we compute Persistence Diagrams
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
                #For each dimension we compute different topological feature vectors.
                for i_dim in range(n_dim):
                    dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                    dimensionscaler.fit(persistence)
                    dim_persistence=np.array(dimensionscaler.transform(persistence))
                    for i_vector in range(n_vectors):
                        tda_compt=feat_vect[i_vector]
                        tda_compt.fit([dim_persistence])
                        tda_vect[i_motiv][i_vector][i_dim]=tda_compt.transform([dim_persistence])
                np.save(subj_dir+'_sil80_m'+str(i_motiv), tda_vect[i_motiv][0][0])       

            #For each point of the test set we add this point to all three Point Clouds
            i=0
            for index in ind_test:
                for i_motiv  in range(3):
                    X_temp=np.concatenate((X_motiv[i_motiv],PC[index].reshape(1,-1)),axis=0)
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
                    #For each dimension and feature vector we compute the euclidean norm to assign a distance on how much the topology has changed
                    for i_dim in range(n_dim):
                        dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                        dimensionscaler.fit(persistence)
                        dimensional_persistence=np.array(dimensionscaler.transform(persistence))
                        for i_vector in range(n_vectors):
                            tda_compt=feat_vect[i_vector]
                            tda_compt.fit([dimensional_persistence])
                            silhouette = tda_compt.transform([dimensional_persistence])
                            
                            topo_pred_array[i,i_vector,i_dim,i_motiv]=np.linalg.norm(silhouette-tda_vect[i_motiv][i_vector][i_dim])
                        
                    np.save(subj_dir+'_sil20_m'+str(i_motiv)+'_p'+str(i), silhouette)  
                i=i+1
        
            #We predict and compute accuracy and confusion martix
            for i_vector in range(n_vectors):
                for i_dim in range(n_dim):
                    topo_pred,rand_n[i_rep,i_vector,i_dim]=topological_clf(topo_pred_array[:,i_vector,i_dim,:])

                    topo_perf[i_dim,i_vector,i_rep] = skm.accuracy_score(topo_pred, labels_dwnsamp[ind_test])
            np.save(subj_dir+'_sil20_class', topo_perf[i_dim,i_vector,i_rep])  
    print((time.time()-t_int)/60, 'minuts for classification')
    
    
   
