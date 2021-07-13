 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:51:44 2021

@author: fritz
"""
import scipy.io as sio
import os
#from EEGs_persistence import *
from preprocess_data import *
from TDApipeline import *
from intensities_pipe import *
import time  
import sklearn.pipeline as skppl
import sklearn.linear_model as skllm
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprp
import pandas as pd
import sklearn.neighbors as sklnn
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sn
'''
from joblib import Memory
from shutil import rmtree

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
'''


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
    elif space=='fontSpace':
        font_space=raw_data['ic_data3']
        return font_space,subj_dir
    else:
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        font_space=raw_data['ic_data3']
        return (elec_space,font_space),subj_dir,raw_data['indexM']
    


if __name__ == "__main__":
    subjects=[25,27,28]
    
    intensities=False
    exploratory=False
    classification=False
    PCA=True
    last=False
    
        
    bloc_dic={}
    bloc_subj_dic={}
    bloc_subj_dic[27]=np.array([[1, 2, 10, 6, 7, 3],[5, 2, 4, 1, 8, 9]])
    bloc_subj_dic[25]=np.array([[1, 2, 8, 3, 5, 4],[6, 7, 2, 10, 1, 9]])
    bloc_subj_dic[28]=np.array([[1, 2, 9, 7, 4, 6],[5, 3, 2, 8, 1, 10]])
    
    
    bands=[0,1,2,-1]  
    #bands=[-1,2]
    n_band=len(bands)
    measures=["euclidean","correlation","quaf"]#,"dtw"]
    #measures=["quaf","dtw"]
    n_measure=len(measures)
    dimensions=["zero","one"]
    #dimensions=[]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    
    
    #data_table=np.zeros((2*len(subjects),29))
    subj_t=0
    
    n_vectors=len(feat_vect)
    for subject in subjects:
        
        space='both'
        data_space,subj_dir,index=load_data(subject,space=space)

        spaces=['electrodeSpace','fontSpace']
        index=index[0]

        cont1=0
        cont2=0
        for ind in range(12):
            if index[ind]==1:
                cont1+=1
                if cont1==2:
                    index[ind]=11
            if index[ind]==2:
                cont2+=1
                if cont2==2:
                    index[ind]=12 
        
        bloc_subj_dic[subject][1][bloc_subj_dic[subject][1]<=2]=bloc_subj_dic[subject][1][bloc_subj_dic[subject][1]<=2]+10
        bloc_session=np.where([ind in bloc_subj_dic[subject][1] for ind in index],2,1 )
        for sp in range(2):
            t=time.time()
            space=spaces[sp]
            
            subject_table=np.zeros((8,15))
            
            if not os.path.exists(subj_dir+space):
                print("create directory(plot):",subj_dir+space)
                os.makedirs(subj_dir+'/'+space)
            print('cleaning and filtering data of',space,'of subject',subject)
            preprocessor=Preprocessor(data_space[sp])
            #filtered_ts_dic=preprocessor.get_filtered_ts_dic()
            ts_band,labels_original=preprocessor.get_trials_and_labels()
            
            subject_table[:,1]=preprocessor.N
            
            
            #data_table[subj_t,0]=preprocessor.N
            #data_table[subj_t,1]=ts_band.shape[0]
            '''
            if debug:
                ts_band=np.concatenate((ts_band[:50,:],ts_band[432:482,:],ts_band[-50:,:]),axis=0)
                labels=np.concatenate((np.zeros(50),np.ones(50),np.ones(50)*2))'''
            
            if intensities:
                for i_band in bands:
                    print('intensities for band ', i_band)
                    PC=np.abs(ts_band[:,i_band,:,:]).mean(axis=1)
                    intensity(subj_dir,space,PC,labels_original,i_band)

            #data_table[subj_t,6]=(labels_original==0).sum()
            #data_table[subj_t,7]=(labels_original==1).sum()
            #data_table[subj_t,8]=(labels_original==2).sum()
            
            
            blocs=[]
            blocs.append(np.array(list(range(12)))[bloc_session==1])
            blocs.append(np.array(list(range(12)))[bloc_session==2])
            band_dic={-1: 'noFilter', 0:'alpha',1:'betta',2:'gamma'}    
            if PCA: 
                t_pca=time.time()
                N=ts_band.shape[-1]
                persistence={}
                #persistence_2d={}
                fig = plt.figure(figsize=[18,8])
                for i_band in bands:
                    
                    persistence[i_band]={}
                    #persistence_2d[i_band]={}
                    bloc_i=1
                    fig = plt.figure(figsize=[18,8])
                    PC_all=np.abs(ts_band[:,i_band,:,:]).mean(axis=1)
                    PC_all=PC_all.reshape((-1,N))
                    labels_all=labels_original
                                        
                    tr2bl=preprocessor.tr2bl_ol
                    
                    table_i=-1
                    for bl in blocs:
                        table_i+=1
                        subject_table[table_i,0]=band_dic[i_band]+str(bloc_i)

                        temp=[tr_bl in bl for tr_bl in tr2bl]
                        PC=PC_all[temp]
                        labels=labels_all[temp]
                        
                        subject_table[table_i,2]=len(labels)
                        subject_table[table_i,3]=len(labels==0)
                        subject_table[table_i,4]=len(labels==1)
                        subject_table[table_i,5]=len(labels==2)
                        
                        PC,labels=preprocessor.reject_outliers(PC,labels)
                        
                        subject_table[table_i,6]=len(labels)
                        subject_table[table_i,7]=len(labels==0)
                        subject_table[table_i,8]=len(labels==1)
                        subject_table[table_i,9]=len(labels==2)
                        #data_table[subj_t,3+i_band]=PC.shape[0]
                        
                        X =(PC - np.mean(PC, axis=0)).T #X.shape: (42,632)
                        n = X.shape[1]
                        Y =  X.T/np.sqrt(n-1)
                    
                        u, s, vh = la.svd(Y, full_matrices=False)
                        r=np.sum(np.where(s>1e-12,1,0))
                        #pca = vh[:r,:] @ X[:,:] # Principal components
                        variance_prop = s[:r]**2/np.sum(s[:r]**2) # Variance captured
                        acc_variance = np.cumsum(variance_prop)
                        std = s[:r]
                        

                        fig, axs = plt.subplots(1, 2, figsize=(18, 4))
                      
                        # 3/4 of the total variance rule
                        axs[0].scatter(range(len(acc_variance)),acc_variance*100)
                        axs[0].set_xticks(range(len(acc_variance)), minor=False)
                        axs[0].hlines(75, xmin=0, xmax=len(std), colors='r', linestyles='dashdot')
                        axs[0].set_title('3/4 of the total variance rule')
                        axs[0].set_xlabel('PCA coordinates')
                        axs[0].set_ylabel('accumulated variance')
                        # Kraiser rule: Keep PC with eigenvalues > 1
                        # Scree plot: keep PCs before elbow
                        axs[1].scatter(range(len(std)),(std**2))
                        axs[1].set_xticks(range(len(acc_variance)), minor=False)
    
                        axs[1].hlines(1, xmin=0, xmax=len(std), colors='r', linestyles='dashdot')
                        axs[1].set_title('Scree Plot')
                        axs[1].set_xlabel('PCA coordinates')
                        axs[1].set_ylabel('eigenvalue')
                  
                        if not os.path.exists(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i)):
                            print("create directory(plot):",subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i) )
                            os.makedirs(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i) )
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_plots.png')
                        plt.close()
                        #print('acumulated variance:',acc_variance)
                        #data_table[subj_t,22+i_band]=acc_variance[2]
                        #data_table[subj_t,26+i_band]=acc_variance[3]
                        subject_table[table_i,10]=acc_variance[3]
                        
                        
                        
                        pca = vh[:3,:] @ X[:,:] 
                        pca=pca.T
                        
                        pca,labels=preprocessor.reject_outliers(pca,labels)
                        
                        
                        subject_table[table_i,11]=len(labels)
                        subject_table[table_i,12]=len(labels==0)
                        subject_table[table_i,13]=len(labels==1)
                        subject_table[table_i,14]=len(labels==2)
                        

                        
                        print('intensities for band ', i_band, 'and session', bloc_i)
                        intensity(subj_dir,space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i),pca,labels,i_band)
                        
                        fig = plt.figure(figsize=[18,8])
                        ax =fig.add_subplot(2, 3, 1, projection='3d')
                        fig.add_axes(ax)
                        #fig.add_subplot(projection='3d')
                        
                        pca_M0=pca[labels==0]
                        pca_M1=pca[labels==1]
                        pca_M2=pca[labels==2]
                        
                        
                        #data_table[subj_t,10+i_band]=(labels==0).sum()
                        #data_table[subj_t,14+i_band]=(labels==1).sum()
                        #data_table[subj_t,18+i_band]=(labels==2).sum()
                        
                        ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='z')
                        ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='z')
                        ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='z')
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC direction z')
                        
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Y$')
                        ax.set_zlabel('$Z$')
                    
                        #plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca projection_z_PC.png')
                        #plt.close()
                        
                        #fig = plt.figure()
                        ax = fig.add_subplot(2, 3, 2, projection='3d')
                        fig.add_axes(ax)
    
                        
                        ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='y')
                        ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='y')
                        ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='y')
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC direction y')
                        
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Z$')
                        ax.set_zlabel('$Y$')
                        '''plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca projection_y_PC.png')
                        plt.close()
                        
                        
                        fig = plt.figure()'''
                        ax = fig.add_subplot(2, 3, 3, projection='3d')
                        fig.add_axes(ax)
    
                        
                        ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='x')
                        ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='x')
                        ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='x')
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC direction x')
                        
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$Z$')
                        ax.set_ylabel('$Y$')
                        ax.set_zlabel('$X$')
                        '''
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca projection_PC.png')
                        plt.close()'''
                        #tr2bl=preprocessor.tr2bl
                        ax = fig.add_subplot(2, 3, 4, projection='3d')
                        fig.add_axes(ax)
    
                        #for bloc in range (12):
                            #ax.scatter(pca_M0[:,0][tr2bl[labels==0]==bloc],pca_M0[:,1][tr2bl[labels==0]==bloc],pca_M0[:,2][tr2bl[labels==0]==bloc],label=bloc,alpha=0.5,zdir='x')
                        ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',alpha=0.5,c='r',zdir='z')
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC motivation 0')
                            
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Y$')
                        ax.set_zlabel('$Z$')
                        
                        ax = fig.add_subplot(2, 3, 5, projection='3d')
                        fig.add_axes(ax)
    
                        #for bloc in range (12):
                            #ax.scatter(pca_M1[:,0][tr2bl[labels==1]==bloc],pca_M1[:,1][tr2bl[labels==1]==bloc],pca_M1[:,2][tr2bl[labels==1]==bloc],label=bloc,alpha=0.5,zdir='x')
                        ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',alpha=0.5,c='g',zdir='z')
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC motivation 1')
                        
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Y$')
                        ax.set_zlabel('$Z$')
                        
                        ax = fig.add_subplot(2, 3, 6, projection='3d')
                        fig.add_axes(ax)
        
                        #for bloc in range (12):    
                            #ax.scatter(pca_M2[:,0][tr2bl[labels==2]==bloc],pca_M2[:,1][tr2bl[labels==2]==bloc],pca_M2[:,2][tr2bl[labels==2]==bloc],label=bloc,alpha=0.5,zdir='x')
                        ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',alpha=0.5,c='b',zdir='z')    
                        ax.legend()
                        ax.set_title(band_dic[i_band]+' pca projection PC motivation 2')
                        
                        ax.set_xlim3d(-1, 1)
                        ax.set_ylim3d(-1, 1)
                        ax.set_zlim3d(-1, 1)
                        
                        ax.set_xlabel('$X$')
                        ax.set_ylabel('$Y$')
                        ax.set_zlabel('$Z$')
                        
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca projection_PC.png')
                        plt.close()

                    
                        pca_list=[pca_M0,pca_M1,pca_M2]
                        band_dic={-1: 'noFilter', 0:'alpha',1:'betta',2:'gamma'}
                        
                        
                        for i in range(3):
                            
                            n_coor = pca_list[i].shape[0]
                            matrix = np.ones((n_coor, n_coor))
                            row,col = np.triu_indices(n_coor,1)
                            distancies=pdist(pca_list[i])
                            matrix[row,col] = distancies
                            matrix[col,row] = distancies
    
                        
                            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                            #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                            persistence[i_band][i]=Rips_simplex_tree_sample.persistence()
        

                    
                    
                        ##2D PCA
                        # pca = vh[:2,:] @ X[:,:] 
                        # pca=pca.T
                        # fig = plt.figure()
                        # ax = plt.axes()
                        # pca_M0=pca[labels==0]
                        # pca_M1=pca[labels==1]
                        # pca_M2=pca[labels==2]
                        # ax.scatter(pca_M0[:,0],pca_M0[:,1],label='M0',c='r',alpha=0.5)
                        # ax.scatter(pca_M1[:,0],pca_M1[:,1],label='M1',c='g',alpha=0.5)
                        # ax.scatter(pca_M2[:,0],pca_M2[:,1],label='M2',c='b',alpha=0.5)
                        # ax.legend()
                        # ax.set_title(band_dic[i_band]+' pca projection PC 2d')
                        
                        # ax.set_xlim(-1, 1)
                        # ax.set_ylim(-1, 1)
                        # plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca_2d_projection_PC.png')
                        # plt.close()
                        # pca_list=[pca_M0,pca_M1,pca_M2]
                        # for i in range(3):
                        #     matrix=cdist(pca_list[i],pca_list[i])
                        #     Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                        #     #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                        #     Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                        #     persistence_2d[i_band][i]=Rips_simplex_tree_sample.persistence()
                
                        
                        
                        
                        # ##Connected Componentes
                        
                        # fig = plt.figure()
                        # ax = plt.axes()
                        # tr2bl=preprocessor.tr2bl
                        # for i in range(3):
                        #     bl1=tr2bl%2==0
                        #     connected_commponents_0=pca_list[i][bl1[labels==i]]
                        #     connected_commponents_1=pca_list[i][~bl1[labels==i]]
                        #     ax.scatter(connected_commponents_0[:,0],connected_commponents_0[:,1],c='r',alpha=0.5)
                        #     ax.scatter(connected_commponents_1[:,0],connected_commponents_1[:,1],c='b',alpha=0.5)
                        # ax.set_title(band_dic[i_band]+' pca projection PC connected components')
                        
                        # ax.set_xlim(-1, 1)
                        # ax.set_ylim(-1, 1)
                        # plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca projection_ConectedComponents_PC.png')
                        # plt.close()
                
                        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
                        plot_func=lambda x,axes: gd.plot_persistence_diagram(x,legend=True,max_intervals=1000,axes=axes)#,inf_delta=0.5)
    
                        aux_lis=np.array([persistence[i_band][0],persistence[i_band][1],persistence[i_band][2]], dtype=object)
                        x_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][0],y))),aux_lis)))+0.05
                        y_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][1] if x[1][1]!=np.inf  else 0 ,y))),aux_lis)))*1.2
                        for j in range(3):
                            a=plot_func(persistence[i_band][j],axes=axes[j])
                            a.set_title('{0} persistence diagramsof \n motivational state {1} and band {2}'.format(space,j,band_dic[i_band]))
                            a.set_xlim(-0.05,x_max)
                            a.set_ylim(0,y_max)
                        fig.suptitle('Persistence diagrams of the {0} for\n frequency band {1} and motivational state PCA'.format(space,band_dic[i_band]),fontsize=24)
                        fig.tight_layout(pad=0.5)
                        fig.subplots_adjust(top=0.8)
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_persistence_diagram.png')
                        plt.close()
                        ##2DPCA
                        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
                        # aux_lis=np.array([persistence_2d[i_band][0],persistence_2d[i_band][1],persistence_2d[i_band][2]], dtype=object)
                        # x_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][0],y))),aux_lis)))+0.05
                        # y_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][1] if x[1][1]!=np.inf  else 0 ,y))),aux_lis)))*1.2
                        # for j in range(3):
                        #     a=plot_func(persistence_2d[i_band][j],axes=axes[j])
                        #     a.set_title('{0} persistence diagrams of \n motivational state {1} and band {2}'.format(space,j,band_dic[i_band]))
                        #     a.set_xlim(-0.05,x_max)
                        #     a.set_ylim(0,y_max)
                        # fig.suptitle('Persistence diagrams of the {0} for\n frequency band {1} and motivational state PCA 2d'.format(space,band_dic[i_band]),fontsize=24)
                        # fig.tight_layout(pad=0.5)
                        # fig.subplots_adjust(top=0.8)
                        # plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/pca_persistence_diagram_2d.png')
                        # plt.close()
                        
                        '''
                        matrix=cdist(PC,PC)
                        Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                        #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                        Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                        persistence=Rips_simplex_tree_sample.persistence()
        
            
                        gd.plot_persistence_diagram(persistence)
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'_Global_Persistence_diagram.png')
                        plt.close()'''
                        
                        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 14))
                        vect0,vect1=[0,0,0],[0,0,0]
                        for i in range(3):
                            dim_list=np.array(list(map(lambda x: x[0], persistence[i_band][i])))
                            point_list=np.array(list(map(lambda x: x[1], persistence[i_band][i])))
                            zero_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)]
                            one_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)]
                            
    
                            descriptors_computer=TopologicalDescriptorsNocl()
                            descriptors_computer.fit((zero_dim,one_dim))
                            vect0[i],vect1[i]=descriptors_computer.transform((zero_dim,one_dim))
                        
                        

                        axes[0].boxplot([vect0[0][0],vect0[1][0],vect0[2][0]],showfliers=False)
                        axes[0].set_title('Life BoxPlot dimension 0')
                        axes[1].boxplot([vect1[0][0],vect1[1][0],vect1[2][0]],showfliers=False)
                        axes[1].set_title('Life BoxPlot dimension 1')
                        
                        axes[2].boxplot([vect1[0][2],vect1[1][2],vect1[2][2]],showfliers=False)
                        axes[2].set_title('Midlife BoxPlot dimension 1')
                        axes[3].boxplot([vect1[0][3],vect1[1][3],vect1[2][3]],showfliers=False)
                        axes[3].set_title('Birth BoxPlot dimension 1')
    
                        axes[4].boxplot([vect1[0][4],vect1[1][4],vect1[2][4]],showfliers=False)
                        axes[4].set_title('Death BoxPlot dimension')
                                #a.set_xlim(-0.05,x_max)
                                #a.set_ylim(0,y_max)
                        fig.suptitle('Descriptors Boxplots of the {0} for\n frequency band {1} and different motivational state'.format(space,band_dic[i_band]),fontsize=24)
                        fig.tight_layout(pad=1.00)
                        fig.subplots_adjust(top=0.8)
                        plt.savefig(subj_dir+space+'/PCA/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_descriptors.png')
                        plt.close()
                        bloc_i+=1
                    subj_t=subj_t+1
                
                subject_table=pd.DataFrame(subject_table[:,1:],index=subject_table[:,0],columns=['Clean Channels','N. trials','M0','M1','M2','N. trials w/o OL','M0 w/o OL','M1 w/o OL','M2 w/o OL','N. trials w/o OL 2','M0 w/o OL 2','M1 w/o OL 2','M2 w/o OL 2'])
                subject_table.to_csv('subj_dir+space+/PCA/data_table.csv')
            

            print('======TIME======')    
            print((time.time()-t_pca)/60, 'minuts for pca')
            resolut=1000
            if exploratory:
                t_expl=time.time()
                labels=labels_original
                for i_band in bands:
                    for i_measure in range(n_measure):
                        print('plotting topological descriptors with ', measures[i_measure])   
                        
                        fig, axes = plt.subplots(nrows=n_band, ncols=3, figsize=(90, 36))
                        fig4, axes4 = plt.subplots(nrows=n_band, ncols=9, figsize=(36, 36))
                        fig2, axes2 = plt.subplots(nrows=n_band*n_dim, ncols=3, figsize=(36, 36))
                        fig3, axes3 = plt.subplots(nrows=n_band*n_dim, ncols=3, figsize=(36, 36))
                        j=0
                        for i_band in bands: 
                            print('band',band_dic[i_band])
                            exploratory_pipe = skppl.Pipeline([("band_election", Band_election(i_band)),("persistence", PH_computer(measure=measures[i_measure]))])
                            persistence=exploratory_pipe.fit_transform(ts_band)
                            for i_dim in range(n_dim):
                                dimension_scaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                                dimension_scaler.fit(persistence)
                                dim_vect=dimension_scaler.transform(persistence)
          
                                descriptors_computer=TopologicalDescriptors()
                                descriptors_computer.fit(dim_vect)
                                vect=descriptors_computer.transform(dim_vect)
                                vect=np.array(vect)
                                
                                vect0=vect[labels==0]
                                vect1=vect[labels==1]
                                vect2=vect[labels==2]
                                
                                if i_dim==0:
                                
                                    axes[i_band][0].boxplot([vect0[:,0],vect1[:,0],vect2[:,0]],showfliers=False)
                
                                    axes[i_band][0].set_title('{0} average life of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                
                                    axes[i_band][1].boxplot([vect0[:,1],vect1[:,1],vect2[:,1]],showfliers=False)
                
                                    axes[i_band][1].set_title('{0} std life of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes[i_band][2].boxplot([vect0[:,2],vect1[:,2],vect2[:,2]],showfliers=False)
                
                                    axes[i_band][2].set_title('{0} persistent entropy of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                
                                else:
                                    
                                    axes4[i_band][0].boxplot([vect0[:,0],vect1[:,0],vect2[:,0]],showfliers=False)
            
                                    axes4[i_band][0].set_title('{0} average life of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                
                                    axes4[i_band][1].boxplot([vect0[:,1],vect1[:,1],vect2[:,1]],showfliers=False)
                
                                    axes4[i_band][1].set_title('{0} std life of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][2].boxplot([vect0[:,2],vect1[:,2],vect2[:,2]],showfliers=False)
                
                                    axes4[i_band][2].set_title('{0} persistent entropy of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                
                                    axes4[i_band][3].boxplot([vect0[:,3],vect1[:,3],vect2[:,3]],showfliers=False)
                
                                    axes4[i_band][3].set_title('{0} average midlife of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][4].boxplot([vect0[:,4],vect1[:,4],vect2[:,4]],showfliers=False)
                
                                    axes4[i_band][4].set_title('{0} std midlife of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][5].boxplot([vect0[:,5],vect1[:,5],vect2[:,5]],showfliers=False)
                
                                    axes4[i_band][5].set_title('{0} average birth of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][6].boxplot([vect0[:,6],vect1[:,6],vect2[:,6]],showfliers=False)
                
                                    axes4[i_band][6].set_title('{0} std birth of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][7].boxplot([vect0[:,7],vect1[:,7],vect2[:,7]],showfliers=False)
                
                                    axes4[i_band][7].set_title('{0} average death of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    axes4[i_band][8].boxplot([vect0[:,8],vect1[:,8],vect2[:,8]],showfliers=False)
                
                                    axes4[i_band][8].set_title('{0} std death of band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                    
                                    
                                silhouette_computer=DimensionSilhouette()
                                silhouette_computer.fit(dim_vect)
                                vect=silhouette_computer.transform(dim_vect)
                                vect=np.array(vect)
                                mean0=vect[labels==0].mean(axis=0)
                                mean1=vect[labels==1].mean(axis=0)
                                mean2=vect[labels==2].mean(axis=0)
                                y_max=np.max([mean0,mean1,mean2])
                                axes2[j][0].plot(mean0[:resolut])
            
                                
                                axes2[j][0].set_title('{0} persistence Silhouette of \n motivational state 0 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes2[j][0].set_xlim(-2,resolut)
                                axes2[j][0].set_ylim(0,y_max*1.1)
                                
                                
                                axes2[j][1].plot(mean1[:resolut])
            
                                
                                axes2[j][1].set_title('{0} persistence Silhouette of \n motivational state 1 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes2[j][1].set_xlim(-2,resolut)
                                axes2[j][1].set_ylim(0,y_max*1.1)
                                
                                axes2[j][2].plot(mean2[:resolut])
            
                                
                                axes2[j][2].set_title('{0} persistence Silhouette of \n motivational state 2 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes2[j][2].set_xlim(-2,resolut)
                                axes2[j][2].set_ylim(0,y_max*1.1)
                                
                                
                                
                                landscape_computer=DimensionLandScape()
                                landscape_computer.fit(dim_vect)
                                vect=landscape_computer.transform(dim_vect)
                                vect=np.array(vect)
                                mean0=vect[labels==0].mean(axis=0)
                                mean1=vect[labels==1].mean(axis=0)
                                mean2=vect[labels==2].mean(axis=0)
                                y_max=np.max([mean0,mean1,mean2])
                                axes3[j][0].plot(mean0[:resolut])
                                axes3[j][0].plot(mean0[resolut:2*resolut])
                                axes3[j][0].plot(mean0[2*resolut:3*resolut])
                                axes3[j][0].plot(mean0[3*resolut:4*resolut])
                                axes3[j][0].plot(mean0[4*resolut:5*resolut])
                                
                                axes3[j][0].set_title('{0} persistence Landscapes of \n motivational state 0 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes3[j][0].set_xlim(-2,resolut)
                                axes3[j][0].set_ylim(0,y_max*1.1)
                                
                                
                                axes3[j][1].plot(mean1[:resolut])
                                axes3[j][1].plot(mean1[resolut:2*resolut])
                                axes3[j][1].plot(mean1[2*resolut:3*resolut])
                                axes3[j][1].plot(mean1[3*resolut:4*resolut])
                                axes3[j][1].plot(mean1[4*resolut:5*resolut])
                                
                                axes3[j][1].set_title('{0} persistence Landscapes of \n motivational state 1 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes3[j][1].set_xlim(-2,resolut)
                                axes3[j][1].set_ylim(0,y_max*1.1)
                                
                                axes3[j][2].plot(mean2[:resolut])
                                axes3[j][2].plot(mean2[resolut:2*resolut])
                                axes3[j][2].plot(mean2[2*resolut:3*resolut])
                                axes3[j][2].plot(mean2[3*resolut:4*resolut])
                                axes3[j][2].plot(mean2[4*resolut:5*resolut])
                                
                                axes3[j][2].set_title('{0} persistence Landscapes of \n motivational state 2 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                                axes3[j][2].set_xlim(-2,resolut)
                                axes3[j][2].set_ylim(0,y_max*1.1)
                                
                                
                                j=j+1
                        expl_path=subj_dir+space+'/exploratory/'+measures[i_measure]
                        if not os.path.exists(expl_path):
                            print("create directory(plot):",expl_path)
                            os.makedirs(expl_path)        
            
                        fig3.suptitle('Persistence Landscapes of the {0} for\n different frequency bands and motivational state of {1}'.format(space,measures[i_measure]),fontsize=24)
                        fig3.tight_layout(pad=0.5)
                        fig3.subplots_adjust(top=0.8)
                        plt.savefig(expl_path+'/Landscapes.png')
                        plt.close()
                        
                        fig2.suptitle('Persistence Silhouette of the {0} for\n different frequency bands and motivational state of {1}'.format(space,measures[i_measure]),fontsize=24)
                        fig2.tight_layout(pad=0.5)
                        fig2.subplots_adjust(top=0.8)
                        plt.savefig(expl_path+'/Silhouette.png')
                        plt.close()
                                
                                
            
                        fig4.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension {2}'.format(space,measures[i_measure],dimensions[i_dim]),fontsize=24)
                        fig4.tight_layout(pad=0.5)
                        fig4.subplots_adjust(top=0.8)
                        plt.savefig(expl_path+'/one_topological_descriptors.png')
                        plt.close()
                            
                    
                        fig.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension 0'.format(space,measures[i_measure]),fontsize=24)
                        fig.tight_layout(pad=0.5)
                        fig.subplots_adjust(top=0.8)
                        plt.savefig(expl_path+'/zero_topological_descriptors.png')
                        plt.close()
                
                
            
                    print('======TIME======')    
                    print((time.time()-t_expl)/60, 'minuts for exploration')
                dimensions.append('both')
                n_dim+=1
                classifiers=[skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=5000))]),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')  ]
                n_classifiers=len(classifiers)
                
                
                cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
                n_rep = 10 # number of repetitions
                
            if classification:
        
                t_clf=time.time()
                perf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers]) # (last index: MLR/1NN)
                perf_shuf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers])# (last index: MLR/1NN)
                conf_matrix = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers,3,3]) # (fourthindex: MLR/1NN)
            
            
        
                for i_band in bands:
                    band_selector=Band_election(i_band)
                    band_selector.fit(ts_band)
                    band_data=band_selector.transform(ts_band)
                    for i_measure in range(n_measure):
                        t_mes=time.time()
                        ph_computer=PH_computer(measure=measures[i_measure])
                        ph_computer.fit(band_data)
                        persistence=ph_computer.transform(band_data)
                        for i_dim in range(n_dim):
                            dimensionscaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                            dimensionscaler.fit(persistence)
                            dimensional_persistence=dimensionscaler.transform(persistence)
                            for i_vector in range(n_vectors):
                                tda_compt=feat_vect[i_vector]
                                tda_compt.fit(dimensional_persistence)
                                tda_vect=tda_compt.transform(dimensional_persistence)
                                for i_classifier in range(n_classifiers):
                                    clf=classifiers[i_classifier]
                                    print('band',band_dic[i_band],'measure',measures[i_measure],'dim',dimensions[i_dim],'vector',i_vector,'classifier',i_classifier)
                                    for i_rep in range(n_rep):
                                        for ind_train, ind_test in cv_schem.split(ts_band,labels): # false loop, just 1 
                                            #print('band',band_dic[i_band],'measure',measures[i_measure],'dim',dimensions[i_dim],'vector',i_vector,'classifier',i_classifier,'repetition:',i_rep)
        
                                            clf.fit(tda_vect[ind_train,:], labels[ind_train])
                                        
                                            perf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier] = clf.score(tda_vect[ind_test,:], labels[ind_test])
                                            conf_matrix[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=clf.predict(tda_vect[ind_test,:]))
                                            
                                            
                                            shuf_labels = np.random.permutation(labels)
                    
                                            clf.fit(tda_vect[ind_train,:], shuf_labels[ind_train])
                                            perf_shuf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier]= clf.score(tda_vect[ind_test,:], shuf_labels[ind_test])
                                    
                        print((time.time()-t_mes)/60, 'minuts for',measures[i_measure])
        
                                                                        
                                    
                                            
        
            
                # save results       
                np.save(subj_dir+space+'/perf.npy',perf)
                np.save(subj_dir+space+'/perf_shuf.npy',perf_shuf)
                np.save(subj_dir+space+'/conf_matrix.npy',conf_matrix)                     
                
                
                fmt_grph = 'png'
                cmapcolours = ['Blues','Greens','Oranges','Reds']
                
                fig, axes = plt.subplots(nrows=n_band, ncols=n_vectors*n_measure*n_dim, figsize=(120, 24))
                    
                for i_band in bands:
                    band = band_dic[i_band]
                    i=0
                    for i_measure in range(n_measure):
                        for i_vector in range(n_vectors):
                            for i_dim in range(n_dim):
                                
                                # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
                                chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
                            
                                # plot performance and surrogate
                                #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                                axes[i_band][i].violinplot(perf[i_band,i_measure,i_dim,i_vector,:,0],positions=[-0.2],widths=[0.3])
                                axes[i_band][i].violinplot(perf[i_band,i_measure,i_dim,i_vector,:,1],positions=[0.2],widths=[0.3])
        
                                axes[i_band][i].violinplot(perf_shuf[i_band,i_measure,i_dim,i_vector,:,0],positions=[0.8],widths=[0.3])
                                axes[i_band][i].violinplot(perf_shuf[i_band,i_measure,i_dim,i_vector,:,1],positions=[1.2],widths=[0.3])
                                
                                axes[i_band][i].plot([-1,2],[chance_level]*2,'--k')
                                axes[i_band][i].axis(xmin=-0.6,xmax=2.4,ymin=0,ymax=1.05)
                                #axes[i_band][i].set_xticks([0,1,2,3],['MLR','1NN','control1','control2'])##Provar
                                axes[i_band][i].set_ylabel('accuracy '+band_dic[i_band]+' '+str(i_vector),fontsize=8)
                                axes[i_band][i].set_title(band_dic[i_band]+' '+measures[i_measure]+dimensions[i_dim]+str(i_vector))
                                i=1+i
                plt.savefig(subj_dir+space+'/accuracies.png', format=fmt_grph)
                plt.close()
                
        
                
                fig2, axes2 = plt.subplots(nrows=n_band, ncols=n_vectors*n_measure*n_dim, figsize=(96, 24))
            
                for i_band in bands:
                    band = band_dic[i_band]
                    i=0
                    for i_measure in range(n_measure):
                        for i_vector in range(n_vectors):
                            for i_dim in range(n_dim):
                                
                                
                                # plot performance and surrogate
                                #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                                
                                axes2[i_band][i].imshow(conf_matrix[i_band,i_measure,i_dim,i_vector,:,0,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
                                #plt.colorbar()
                                axes2[i_band][i].set_xlabel('true label',fontsize=8)
                                axes2[i_band][i].set_ylabel('predicted label',fontsize=8)
                                axes2[i_band][i].set_title(band_dic[i_band]+measures[i_measure]+dimensions[i_dim]+str(i_vector))
                                i=1+i
                fig.tight_layout(pad=0.5)
                plt.savefig(subj_dir+space+'/confusion_matrix.png', format=fmt_grph)
                plt.close()
                print('======TIME======') 
                print((time.time()-t_clf)/60, 'minuts for classification')
            if last:
                t_last=time.time()
                from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
                classifiers = [skppl.Pipeline([("persistence", PH_computer()),("scaler",DimensionDiagramScaler("zero")),("TDA",gd.representations.BottleneckDistance(epsilon=0.1)),("Estimator",sklnn.KNeighborsClassifier(metric="precomputed"))]),skppl.Pipeline([("persistence", PH_computer()),("scaler",DimensionDiagramScaler("one")),("TDA",gd.representations.BottleneckDistance(epsilon=0.1)),("Estimator",sklnn.KNeighborsClassifier(metric="precomputed"))]) ] 
                
                
                perf = np.zeros([n_band,3,n_rep]) # (last index: MLR/1NN)
                perf_shuf = np.zeros([n_band,3,n_rep])# (last index: MLR/1NN)
                conf_matrix = np.zeros([n_band,3,n_rep,3,3]) # (fourthindex: MLR/1NN)
                
        
                for i_band in bands:
                    for i_classifier in range(2):
                        clf=classifiers[i_classifier]
                        t_nn=time.time()
                        print('band',band_dic[i_band],'classifier',i_classifier)
                        for i_rep in range(n_rep):
                            for ind_train, ind_test in cv_schem.split(ts_band,labels): 
                                #print('band',band_dic[i_band],'classifier',i_classifier,'repetition',i_rep)
                                clf.fit(ts_band[ind_train,i_band,:,:], labels[ind_train])
                                pred=clf.predict(ts_band[ind_test,i_band,:,:])
                                perf[i_band,i_classifier,i_rep] = skm.accuracy_score(pred, labels[ind_test])
                                conf_matrix[i_band,i_classifier,i_rep,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=pred) 
                                
                                
                                shuf_labels = np.random.permutation(labels)
                                clf.fit(ts_band[ind_train,i_band,:,:], shuf_labels[ind_train])
                                pred=clf.predict(ts_band[ind_test,i_band,:,:])
                                perf_shuf[i_band,i_classifier,i_rep] = skm.accuracy_score(pred, shuf_labels[ind_test])
                        print((time.time()-t_nn)/60, 'minuts for classifier', i_classifier)         
                            # save results       
                np.save(subj_dir+space+'/perf2.npy',perf)
                np.save(subj_dir+space+'/perf_shuf2.npy',perf_shuf)
                np.save(subj_dir+space+'/conf_matrix2.npy',conf_matrix) 
        
                fmt_grph = 'png'
                cmapcolours = ['Blues','Greens','Oranges','Reds']
                    
                fig, axes = plt.subplots(nrows=1, ncols=n_band, figsize=(48, 24))
                        
                for i_band in bands:
                    band = band_dic[i_band]
                                
                    # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
                    chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
                
                    # plot performance and surrogate
                    #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                    axes[i_band].violinplot(perf[i_band,0,:],positions=[-0.2],widths=[0.3])
                    axes[i_band].violinplot(perf[i_band,1,:],positions=[0.2],widths=[0.3])
                    #axes[i_band].violinplot(perf[i_band,2,:],positions=[0.6],widths=[0.3])
        
                    axes[i_band].violinplot(perf_shuf[i_band,0,:],positions=[1.2],widths=[0.3])
                    axes[i_band].violinplot(perf_shuf[i_band,1,:],positions=[1.6],widths=[0.3])
                    #axes[i_band].violinplot(perf_shuf[i_band,2,:],positions=[2],widths=[0.3])
                    
                    axes[i_band].plot([-1,2],[chance_level]*2,'--k')
                    axes[i_band].axis(xmin=-0.6,xmax=2.4,ymin=0,ymax=1.05)
                    #axes[i_band][i].set_xticks([0,1,2,3],['MLR','1NN','control1','control2'])##Provar
                    axes[i_band].set_ylabel('accuracy '+band,fontsize=8)
                    axes[i_band].set_title(band)
                plt.savefig(subj_dir+space+'/accuracies2.png', format=fmt_grph)
                plt.close()
                    
            
                    
                fig2, axes2 = plt.subplots(nrows=n_band, ncols=2, figsize=(24, 24))
            
                for i_band in bands:
                    band = band_dic[i_band]
                    for i_clf in range (2):
                        # plot performance and surrogate
                        #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                        axes2[i_band][i_clf].imshow(conf_matrix[i_band,i_clf,:,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
                        axes2[i_band][i_clf].colorbar()
                        #plt.colorbar()
                        axes2[i_band][i_clf].set_xlabel('true label',fontsize=8)
                        axes2[i_band][i_clf].set_ylabel('predicted label',fontsize=8)
                        axes2[i_band][i_clf].set_title(band+str(i_clf))
                    
                fig.tight_layout(pad=0.5)
                plt.savefig(subj_dir+space+'/confusion_matrix2.png', format=fmt_grph)
                plt.close()
                print('======TIME======') 
                print((time.time()-t_last)/60, 'minuts for last') 
                                       
            print('======TIME======')
            print('======TIME======')

            print((time.time()-t)/60, 'minuts for', intensities,space,exploratory,classification,last) 
    
    
    subjects_index=[]
    for subject in subjects:
        subjects_index.append('Subject ' +str(subject)+ ' ElectrodeSpace')
        subjects_index.append('Subject ' +str(subject)+ ' FontSpace')
    #data_table=pd.DataFrame(#data_table,index=subjects_index,columns=['clean electrodes','Trials', 'Trials w/o OL NF', 'Trials w/o OL Alpha', 'Trials w/o OL Betta', 'Trials w/o OL Gamma', 'M0', 'M1', 'M2','M0 w /o OL NF', 'M1 w /o OL NF', 'M2 w /o OL NF','M0 w /o OL Alpha', 'M1 w /o OL Alpha', 'M2 w /o OL Alpha','M0 w /o OL Betta', 'M1 w /o OL Betta', 'M2 w /o OL Betta','M0 w /o OL Gamma', 'M1 w /o OL Gamma', 'M2 w /o OL Gamma','Variance with 3 components NF','Variance with 3 components Alpha','Variance with 3 components Betta','Variance with 3 components Gamma','Variance with 2 components NF','Variance with 2 components Alpha','Variance with 2 components Betta','Variance with 2 components Gamma'])
    #data_table.to_csv('results/#data_table.csv')
                                       
