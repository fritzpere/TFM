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
from sklearn.metrics import ConfusionMatrixDisplay

def topological_clf(arr):
    pred=-np.ones(len(arr))

    pred[(arr==0).sum(axis=1)<=1]=np.argmin((arr[(arr==0).sum(axis=1)<=1]),axis=1)
    random_selections=((arr==0).sum(axis=1)>1).sum()
    pred[(arr==0).sum(axis=1)>1]=np.random.choice(3,random_selections)
    return pred,random_selections



def intensity(subj_dir,space,PC,labels,i_band):
    dimensions=["zero","one"]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    feat_vect_names=['Landscapes','Silhouettes','Descriptors','Bottleneck']
    n_vectors=len(feat_vect)
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    
    n_rep=3 ##canviar
    rand_n=np.zeros((n_rep,n_vectors+1,n_dim))
    test_size=np.zeros(n_rep)
    perf = np.zeros([n_dim,n_vectors+1,n_rep])

    perf_shuf = np.zeros([n_dim,n_vectors+1,n_rep])
    conf_matrix = np.zeros([n_dim,n_vectors+1,n_rep,3,3])

    
    if not os.path.exists(subj_dir+space+'/clf'):
        print("create directory(plot):",subj_dir+space+'/clf')
        os.makedirs(subj_dir+space+'/clf')
    
    t_int=time.time()
    
    trials_per_m=min((labels==0).sum(),(labels==1).sum(),(labels==2).sum())
    
    if trials_per_m==0:
        return -1,np.zeros((n_vectors+1,n_dim)),-1
    
    X_m0_dwnsamp=PC[labels==0][np.random.choice(len(PC[labels==0]),trials_per_m)]
    X_m1_dwnsamp=PC[labels==1][np.random.choice(len(PC[labels==1]),trials_per_m)]
    X_m2_dwnsamp=PC[labels==2][np.random.choice(len(PC[labels==2]),trials_per_m)]
    
    PC_dwnsamp=np.concatenate((X_m0_dwnsamp,X_m1_dwnsamp,X_m2_dwnsamp),axis=0)
    labels_dwnsamp=np.concatenate((np.zeros(trials_per_m),np.ones(trials_per_m),np.ones(trials_per_m)*2))

    for i_rep in range(n_rep):
        t_rep=time.time()

        
            
        X_motiv=[]
        tda_vect={0:defaultdict(lambda: defaultdict(lambda: [])),1:defaultdict(lambda: defaultdict(lambda: [])),2:defaultdict(lambda: defaultdict(lambda: []))}
        for ind_train, ind_test in cv_schem.split(PC_dwnsamp,labels_dwnsamp):
            test_size[i_rep]=len(ind_test)
            X_train=PC_dwnsamp[ind_train]
            y_train=labels_dwnsamp[ind_train]
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
                        tda_vect[i_motiv][i_vector][i_dim]=tda_compt.transform([dim_persistence])
                    tda_vect[i_motiv][n_vectors][i_dim]=dim_persistence
                    
                    
            descriptors0=np.concatenate((tda_vect[0][2][0],tda_vect[1][2][0],tda_vect[2][2][0]),axis=0)
            descriptors1=np.concatenate((tda_vect[0][2][1],tda_vect[1][2][1],tda_vect[2][2][1]),axis=0)
            
            
            max0=descriptors0.max(axis=0)
            max1=descriptors1.max(axis=0)
            
            min0=descriptors0.min(axis=0)
            min1=descriptors1.min(axis=0)
              
            descriptors0=(descriptors0-min0)/(max0-min0)
            descriptors1=(descriptors1-min1)/(max1-min1)
            
            maxs=[max0,max1]
            mins=[min0,min1]
            
            for m in range(3):
                tda_vect[m][2][0]=descriptors0[m]
                tda_vect[m][2][1]=descriptors1[m]
            
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
                    for i_dim in range(n_dim):
                        dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                        dimensionscaler.fit(persistence)
                        dimensional_persistence=np.array(dimensionscaler.transform(persistence))
                        for i_vector in range(n_vectors-1):
                            tda_compt=feat_vect[i_vector]
                            tda_compt.fit([dimensional_persistence])
                            
                            pred_array[i,i_vector,i_dim,i_motiv]=np.linalg.norm(tda_compt.transform([dimensional_persistence])-tda_vect[i_motiv][i_vector][i_dim])
                        
                        tda_compt=feat_vect[n_vectors-1]
                        tda_compt.fit([dimensional_persistence])
                        
                        pred_array[i,n_vectors-1,i_dim,i_motiv]=np.linalg.norm(((tda_compt.transform([dimensional_persistence])-mins[i_dim])/(maxs[i_dim]-mins[i_dim]))-tda_vect[i_motiv][n_vectors-1][i_dim])

                        pred_array[i,n_vectors,i_dim,i_motiv]=gd.bottleneck_distance(dimensional_persistence,tda_vect[i_motiv][n_vectors][i_dim],0.01)
                i=i+1
        
    
            for i_vector in range(n_vectors+1):
                for i_dim in range(n_dim):
                    pred,rand_n[i_rep,i_vector,i_dim]=topological_clf(pred_array[:,i_vector,i_dim,:])

                    perf[i_dim,i_vector,i_rep] = skm.accuracy_score(pred, labels_dwnsamp[ind_test])
                    conf_matrix[i_dim,i_vector,i_rep,:,:] += skm.confusion_matrix(y_true=labels_dwnsamp[ind_test], y_pred=pred)
                        
                    
                    
    print((time.time()-t_int)/60, 'minuts for classification')
 
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'}
    band = band_dic[i_band]
    np.save(subj_dir+space+'/clf/'+band+'perf_intensity.npy',perf)  
    np.save(subj_dir+space+'/clf/'+band+'conf_matrix_intensity.npy',conf_matrix)
    fmt_grph = 'png'
    cmapcolours = ['Blues','Greens','Oranges','Reds']
    plt.rcParams['xtick.labelsize']=16 
    plt.rcParams['ytick.labelsize']=8
    fig, axes = plt.subplots(nrows=n_dim, ncols=1, figsize=(24, 12))
        



    for i_dim in range(n_dim):
            
        # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
        chance_level = np.max(np.unique(labels_dwnsamp, return_counts=True)[1]) / labels_dwnsamp.size

        axes[i_dim].violinplot(perf[i_dim,0,:],positions=[-0.2],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,1,:],positions=[0.2],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,2,:],positions=[0.6],widths=[0.3])
        axes[i_dim].violinplot(perf[i_dim,3,:],positions=[1],widths=[0.3])



        
        axes[i_dim].plot([-1,2],[chance_level]*2,'--k')
        axes[i_dim].axis(xmin=-0.6,xmax=1.4,ymin=0,ymax=1.05)

        axes[i_dim].set_ylabel('accuracy '+band,fontsize=16)
        axes[i_dim].set_title('band '+band+' dimension '+dimensions[i_dim],fontsize=24)
        fig.suptitle('Accuracies for different dimensions and feature vectors of band '+band,fontsize=36)
        plt.setp(axes, xticks=[-0.2, 0.2, 0.6,1], xticklabels=feat_vect_names,yticks=[0, 0.2,0.4, 0.6,0.8,1])
    
    plt.savefig(subj_dir+space+'/clf/accuracies_intensity_'+band+'.png', format=fmt_grph)
    plt.close(fig)
    plt.rcParams['xtick.labelsize']=24
    plt.rcParams['ytick.labelsize']=24
    plt.rcParams.update({'font.size': 24})
    
    fig2, axes2 = plt.subplots(nrows=n_dim, ncols=n_vectors+1, figsize=(60, 30))

    for i_vector in range(n_vectors+1):
        for i_dim in range(n_dim):
            disp = ConfusionMatrixDisplay(conf_matrix[i_dim,i_vector,:,:,:].mean(0),
                                  display_labels=['M0','M1','M2'])
            disp.plot(ax=axes2[i_dim][i_vector],include_values=True,cmap=cmapcolours[i_band],colorbar=True)
            
            axes2[i_dim][i_vector].set_xlabel('true label',fontsize=24)
            axes2[i_dim][i_vector].set_ylabel('predicted label',fontsize=24)
            axes2[i_dim][i_vector].set_title('band '+band+' dimension '+dimensions[i_dim]+' w/ '+feat_vect_names[i_vector],fontsize=36)

            fig2.suptitle('Confusion Matrices for different dimensions and feature vectors of band '+band,fontsize=48)

            plt.subplots_adjust(top=0.75)
            plt.setp(axes, xticks=[0, 1, 2],yticks=[0, 1, 2])

    #fig2.tight_layout(pad=0.5)
    plt.savefig(subj_dir+space+'/clf/confusion_matrix_intensities_'+band+'.png', format=fmt_grph)
    plt.close(fig2)
    return test_size.mean(),rand_n.mean(axis=0),perf[0,1,:].mean()