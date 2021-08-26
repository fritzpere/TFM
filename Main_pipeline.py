 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:51:44 2021

@author: fritz
"""

from preprocess_data import *
from TDApipeline import *
from intensities_pipeline import *
from knn_pipeline import *
import scipy.io as sio
import os
import time  
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as la


def define_subject_dir(i_sub):
    """
    Creates the directory if it doesn't exist
    :param i_sub: subject id
    :return: directory path
    """
    res_dir = "results/intensities/subject_" + str(i_sub) +'/'
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

    subjects=list(range(25,36 )) 

    bloc_dic={}
    bloc_subj_dic={}
    ##We define which blocs correspond to which sessions. This is information we need to get handed from the experiment.
    bloc_subj_dic[25]=np.array([[1, 2, 8, 3, 5, 4],[6, 7, 2, 10, 1, 9]])
    bloc_subj_dic[26]=np.array([[1, 2, 10, 6, 3, 7],[1, 2, 4, 5, 8, 9]])
    bloc_subj_dic[27]=np.array([[1, 2, 10, 6, 7, 3],[5, 2, 4, 1, 8, 9]])
    bloc_subj_dic[28]=np.array([[1, 2, 9, 7, 4, 6],[5, 3, 2, 8, 1, 10]])
    bloc_subj_dic[29]=np.array([[1, 2, 8, 5, 4, 7],[3, 6, 2, 10, 9, 1]])
    bloc_subj_dic[30]=np.array([[1, 2, 10, 8, 7, 3],[5, 6, 2, 9, 4, 1]])
    bloc_subj_dic[31]=np.array([[1, 3, 2, 5, 8, 10],[4, 5, 1, 6, 2, 7]])
    bloc_subj_dic[32]=np.array([[1, 2, 5, 9, 8, 10],[2, 4, 6, 1, 7, 3]])
    bloc_subj_dic[33]=np.array([[1, 3, 5, 4, 2 ,6],[8, 2, 10, 9, 1, 7]])
    bloc_subj_dic[34]=np.array([[2, 6, 4, 3, 1, 5],[7, 1, 9, 10, 2, 8]])
    bloc_subj_dic[35]=np.array([[1, 3, 5, 4, 2, 6],[8, 10, 2, 9, 1, 7]])

    ##  We define the bands, dimensions and TOpological Feature Vectors that we will use
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'} 
    bands=[2,1,0,-1] 
    n_band=len(bands)
    dimensions=["zero","one"]
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    n_vectors=len(feat_vect)
    n_subj=len(subjects)
    data_table=np.zeros((2*n_subj,11))
    subj_t=0              
    random_predictions_matrix=np.zeros((n_dim,n_vectors+1))
    ## For each subject we load the data
    for subject in subjects:

        space='both'
        data_space,subj_dir,index=load_data(subject,space=space)

        spaces=['electrodeSpace','fontSpace']
        index=index[0]
        ## We reoganize the blocks betwee sessions to make it easier to work with
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

        #For both Electrode Space and Font Space we will preprocess the data. (we remove Nans, organize the data into Time Series, filter the data into 3 different frequancy bands)
        for sp in range(2):
            t=time.time()
            space=spaces[sp]

            subject_table=np.zeros((8,11))
            max_acc=np.zeros((2,4))

            if not os.path.exists(subj_dir+space):
                print("create directory(plot):",subj_dir+space)
                os.makedirs(subj_dir+'/'+space)
            print('cleaning and filtering data of',space,'of subject',subject)
            preprocessor=Preprocessor(data_space[sp])
            #filtered_ts_dic=preprocessor.get_filtered_ts_dic()
            ts_band,labels_original,invalid_ch=preprocessor.get_trials_and_labels()
            if sp==0:
                np.save(subj_dir+'/silent-channels-'+str(subject)+'.npy',invalid_ch)
            
            ## We fill up a table with the number of clean electrodes for each subject.(A table for each subject) (general table for all subjects)
            subject_table[:,0]=preprocessor.N
             ## We fill up a table with the number of trials in total and for each motivational state. (general table for all subjects)
            data_table[subj_t+n_subj*sp,0]=preprocessor.N
            data_table[subj_t+n_subj*sp,1]=ts_band.shape[0]
            data_table[subj_t+n_subj*sp,2]=(labels_original==0).sum()
            data_table[subj_t+n_subj*sp,3]=(labels_original==1).sum()
            data_table[subj_t+n_subj*sp,4]=(labels_original==2).sum()

            #We defina which trials correspond to which Session
            sessions=[]
            sessions.append(np.array(list(range(12)))[bloc_session==1])
            sessions.append(np.array(list(range(12)))[bloc_session==2])

            t_pca=time.time()
            N=ts_band.shape[-1]
            persistence={}
            subject_table_index=[]
            table_i=-1
            for i_band in bands:
                persistence[i_band]={}
                bloc_i=1
                #We compute the mean intensity of each Time Series
                PC_all=np.abs(ts_band[:,i_band,:,:]).mean(axis=1)
                PC_all=PC_all.reshape((-1,N))
                labels_all=labels_original
                tr2bl=preprocessor.tr2bl_ol
                #For each Session we compute select the Point Cloud, remove outliers, realize a PCA, compute the topology and fill up the table
                for ses in sessions:
                    table_i+=1
                    subject_table_index.append(band_dic[i_band]+str(bloc_i))
                    temp=[tr_bl in ses for tr_bl in tr2bl]
                    PC=PC_all[temp]
                    labels=labels_all[temp]

                    subject_table[table_i,1]=len(labels)
                    subject_table[table_i,2]=len(labels[labels==0])
                    subject_table[table_i,3]=len(labels[labels==1])
                    subject_table[table_i,4]=len(labels[labels==2])

                    data_table[subj_t+n_subj*sp,4+bloc_i]=PC.shape[0]
                    #We Apply PCA to our Point Cloud to reduce the dimensionality
                    mean=np.mean(PC, axis=0)
                    X =(PC - mean).T #X.shape: (42,632)
                    n = X.shape[1]
                    Y =  X.T/np.sqrt(n-1)

                    u, s, vh = la.svd(Y, full_matrices=False)
                    r=np.sum(np.where(s>1e-12,1,0))
                    #pca = vh[:r,:] @ X[:,:] # Principal components
                    variance_prop = s[:r]**2/np.sum(s[:r]**2) # Variance captured
                    acc_variance = np.cumsum(variance_prop)
                    std = s[:r]

                    #Let us plot the accumulated variance that we have for each dimension
                    fig= plt.figure( figsize=(18, 4))

                    # 3/4 of the total variance rule
                    plt.scatter(range(len(acc_variance)),acc_variance*100)
                    plt.xticks(range(len(acc_variance)))
                    plt.hlines(75, xmin=0, xmax=len(std), colors='r', linestyles='dashdot')
                    plt.title('3/4 of the total variance rule')
                    plt.xlabel('PCA coordinates')
                    plt.ylabel('accumulated variance')

                    if not os.path.exists(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)):
                        print("create directory(plot):",subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i) )
                        os.makedirs(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i) )
                    plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/accumulated_variance.png')
                    plt.close(fig)
                    #print('acumulated variance:',acc_variance)
                    #We save on our subject table the accumulated variance within the 3 most important dimensions.
                    subject_table[table_i,5]=acc_variance[3]
                    #Let us work with this 3-dimensional Point Cloud. 
                    pca = vh[:3,:] @ X[:,:]

                    pca=pca.T



                    pca,labels,PC=preprocessor.reject_outliers(pca,labels,PC,m=2) 
                    
                    
                    #We fill up the table again since we have removed outliers
                    subject_table[table_i,6]=len(labels)
                    subject_table[table_i,7]=len(labels[labels==0])
                    subject_table[table_i,8]=len(labels[labels==1])
                    subject_table[table_i,9]=len(labels[labels==2])
                    
                    #We reproject the PCA to the original coordinates and save the reprojected and the originals to compare them laters
                    reproj= vh[:3,:].T @ pca.T + mean.reshape((-1,1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m0.npy',reproj[:,labels==0].mean(axis=1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m1.npy',reproj[:,labels==1].mean(axis=1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m2.npy',reproj[:,labels==2].mean(axis=1))
                            
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m0.npy',PC[labels==0].mean(axis=0))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m1.npy',PC[labels==1].mean(axis=0))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m2.npy',PC[labels==2].mean(axis=0))


                    #Now we can use Topology om order to classify trials depending on how much they change the topology of each Point Cloud of motivational States
                    print('intensities for band ', band_dic[i_band], 'and session', bloc_i)
                    subject_table[table_i,10],random_predictions_matrix,max_acc[bloc_i-1,i_band]=tda_intensity_classifier(subj_dir,space+'/'+band_dic[i_band]+'/session'+str(bloc_i),pca,labels,i_band)
                    

                    #Now we weill plot the Point Cloud in 3 different Perspectives and also the point cloud of each motivational State
                    plt.rcParams['xtick.labelsize']=16
                    plt.rcParams['ytick.labelsize']=16
                    plt.rcParams.update({'font.size': 16})

                    fig = plt.figure(figsize=[36,16])
                    ax =fig.add_subplot(2, 3, 1, projection='3d')
                    fig.add_axes(ax)
                    #fig.add_subplot(projection='3d')

                    pca_M0=pca[labels==0]
                    pca_M1=pca[labels==1]
                    pca_M2=pca[labels==2]

                    ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='z')
                    ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='z')
                    ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='z')
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud direction z')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$X$')
                    ax.set_ylabel('$Y$')
                    ax.set_zlabel('$Z$')
                    
                    ax = fig.add_subplot(2, 3, 2, projection='3d')
                    fig.add_axes(ax)
                    ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='y')
                    ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='y')
                    ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='y')
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud direction y')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$X$')
                    ax.set_ylabel('$Z$')
                    ax.set_zlabel('$Y$')

                    ax = fig.add_subplot(2, 3, 3, projection='3d')
                    fig.add_axes(ax)
                    ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',c='r',alpha=0.5,zdir='x')
                    ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',c='g',alpha=0.5,zdir='x')
                    ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',c='b',alpha=0.5,zdir='x')
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud direction x')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$Z$')
                    ax.set_ylabel('$Y$')
                    ax.set_zlabel('$X$')

                    ax = fig.add_subplot(2, 3, 4, projection='3d')
                    fig.add_axes(ax)
                    ax.scatter(pca_M0[:,0],pca_M0[:,1],pca_M0[:,2],label='M0',alpha=0.5,c='r',zdir='z')
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud motivation 0')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$X$')
                    ax.set_ylabel('$Y$')
                    ax.set_zlabel('$Z$')

                    ax = fig.add_subplot(2, 3, 5, projection='3d')
                    fig.add_axes(ax)
                    ax.scatter(pca_M1[:,0],pca_M1[:,1],pca_M1[:,2],label='M1',alpha=0.5,c='g',zdir='z')
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud motivation 1')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$X$')
                    ax.set_ylabel('$Y$')
                    ax.set_zlabel('$Z$')

                    ax = fig.add_subplot(2, 3, 6, projection='3d')
                    fig.add_axes(ax)
                    ax.scatter(pca_M2[:,0],pca_M2[:,1],pca_M2[:,2],label='M2',alpha=0.5,c='b',zdir='z')    
                    ax.legend()
                    ax.set_title(band_dic[i_band]+' pca projection Point Cloud motivation 2')
                    ax.set_xlim3d(-1, 1)
                    ax.set_ylim3d(-1, 1)
                    ax.set_zlim3d(-1, 1)
                    ax.set_xlabel('$X$')
                    ax.set_ylabel('$Y$')
                    ax.set_zlabel('$Z$')
                    
                    fig.suptitle('Point Clouds of Principal Components of band '+band_dic[i_band] ,fontsize=48)
                    plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca projection_PC.png')
                    plt.close(fig)
                    ##let us compute the persistence Diagram for each Motavational state and plot it
                    pca_list=[pca_M0,pca_M1,pca_M2]
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
                        if persistence[i_band][i]==[]:
                            persistence[i_band][i]=[(0,(0.0,0.0))]
                            
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
                    plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_persistence_diagram.png')
                    plt.close(fig)
                    ##let us compute the persistence Silhouettes for each Motavational state and plot it
                    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
                    for i_dim in range(2):
                        silhouettes=[]
                        for i_motiv in range(3):
                            dim_list=np.array(list(map(lambda x: x[0], persistence[i_band][i_motiv])))
                            point_list=np.array(list(map(lambda x: x[1], persistence[i_band][i_motiv])))
                            zero_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)]
                            one_dim=point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)]
                            dim_persistence=(zero_dim,one_dim)
                            silhouettes.append(dim_persistence[i_dim])
                        silhouette_computer=DimensionSilhouette()
                        silhouette_computer.fit([silhouettes[0],silhouettes[1],silhouettes[2]])
                        vect=silhouette_computer.transform([silhouettes[0],silhouettes[1],silhouettes[2]])
                        y_max=np.max(vect)
                        axes[i_dim][0].plot(vect[0])
                        axes[i_dim][0].set_title('{0} persistence Silhouette of \n motivational state 0 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                        axes[i_dim][0].set_xlim(-2,1000)
                        axes[i_dim][0].set_ylim(0,y_max*1.1)
                        
                        axes[i_dim][1].plot(vect[1])
                        axes[i_dim][1].set_title('{0} persistence Silhouette of \n motivational state 1 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                        axes[i_dim][1].set_xlim(-2,1000)
                        axes[i_dim][1].set_ylim(0,y_max*1.1)
                        
                        axes[i_dim][2].plot(vect[2])
                        axes[i_dim][2].set_title('{0} persistence Silhouette of \n motivational state 2 and band {1} of dimension {2}'.format(space,band_dic[i_band],dimensions[i_dim]))
                        axes[i_dim][2].set_xlim(-2,1000)
                        axes[i_dim][2].set_ylim(0,y_max*1.1)
                        
                    fig.suptitle('Persistence Silhouettes of the {0} for\n frequency band {1} and motivational state PCA'.format(space,band_dic[i_band]),fontsize=24)
                    fig.tight_layout(pad=0.5)
                    fig.subplots_adjust(top=0.8)
                    plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_persistence_silhouettes.png')
                    plt.close(fig)
                    ##let us compute the Topological Descriptors for each Motavational state and plot it
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
                    fig.suptitle('Descriptors Boxplots of the {0} for\n frequency band {1} and different motivational states'.format(space,band_dic[i_band]),fontsize=24)
                    fig.tight_layout(pad=1.00)
                    fig.subplots_adjust(top=0.8)
                    plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/pca_descriptors.png')
                    plt.close(fig)
                    ##We save the number of random classifications we have made for each dimension and Feature vector
                    random_predictions_matrix=pd.DataFrame(random_predictions_matrix,columns=['dimension 0','dimension 1'],index=['Landscapes','Silhouettes','Descriptors','Bottleneck'])
                    random_predictions_matrix.to_csv(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/random_preds.csv')
                    
                    bloc_i+=1
            ## We select the band with highet mean accuracy from the silhouette feature vector       
            max_bloc1=np.argmax(max_acc[0,:])
            max_bloc2=np.argmax(max_acc[1,:])
            data_table[subj_t+n_subj*sp,8]=band_dic[max_bloc1]
            if max_bloc1==3:
                max_bloc1=-1
            data_table[subj_t+n_subj*sp,7]=max_acc[0,max_bloc1]
            data_table[subj_t+n_subj*sp,10]=band_dic[max_bloc2]
            if max_bloc2==3:
                max_bloc2=-1
            data_table[subj_t+n_subj*sp,9]=max_acc[1,max_bloc2]
            
            ## We finish complete table for the subject
            subject_table=pd.DataFrame(subject_table,index=subject_table_index,columns=['Clean Channels','N. trials','M0','M1','M2','captured variance after PCA','N. trials w/o Outliers ','M0 w/o Outliers ','M1 w/o Outliers ','M2 w/o Outliers ','test size'])
            
            subject_table.to_csv(subj_dir+space+'/'+'/subject_table.csv')

            print('======TIME======')    
            #print((time.time()-t_pca)/60, 'minuts for pca')
        subj_t=subj_t+1
    #Finishing the general table
    subjects_index=[]
    for subject in subjects:
        subjects_index.append('Subject ' +str(subject)+ ' ElectrodeSpace')
    for subject in subjects:
        subjects_index.append('Subject ' +str(subject)+ ' FontSpace')
    data_table=pd.DataFrame(data_table,index=subjects_index,columns=['Channels','Trials', 'Trials M0', 'Trials M1', 'Trials M2', 'Trials Session 1', 'Trials Session 2','maximum accuracy w/ Silhouette achieved in session 1','frequency band of maximum accuracy in session1','maximum accuracy w/ Silhouette achieved in session 2','frequency band of maximum accuracy in session2'])
    data_table.to_csv('results/intensities/data_table.csv')
                                       
