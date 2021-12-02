 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:51:44 2021

@author: fritz
"""

from utils.preprocess_data import *
from utils.TDApipeline import *
from utils.intensities_pipeline import *
import scipy.io as sio
import os
import dataframe_image as dfi
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

    subjects=list(range(25,36)) 
    subjects=[33]

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

        #spaces=['electrodeSpace','fontSpace']
        spaces=['electrodeSpace']
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
        for sp in range(1):
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
            subject_table[:,0]=int(preprocessor.N)
             ## We fill up a table with the number of trials in total and for each motivational state. (general table for all subjects)
            data_table[subj_t+n_subj*sp,0]=int(preprocessor.N)
            data_table[subj_t+n_subj*sp,1]=int(ts_band.shape[0])
            data_table[subj_t+n_subj*sp,2]=int((labels_original==0).sum())
            data_table[subj_t+n_subj*sp,3]=int((labels_original==1).sum())
            data_table[subj_t+n_subj*sp,4]=int((labels_original==2).sum())

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

                    subject_table[table_i,1]=int(len(labels))
                    subject_table[table_i,2]=int(len(labels[labels==0]))
                    subject_table[table_i,3]=int(len(labels[labels==1]))
                    subject_table[table_i,4]=int(len(labels[labels==2]))

                    data_table[subj_t+n_subj*sp,4+bloc_i]=int(PC.shape[0])
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
                    '''     
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
                    plt.close(fig)'''
                    #print('acumulated variance:',acc_variance)
                    #We save on our subject table the accumulated variance within the 3 most important dimensions.
                    subject_table[table_i,5]=acc_variance[3]
                    #Let us work with this 3-dimensional Point Cloud. 
                    pca = vh[:3,:] @ X[:,:]

                    pca=pca.T



                    pca,labels,PC=preprocessor.reject_outliers(pca,labels,PC,m=2) 
                    
                    
                    #We fill up the table again since we have removed outliers
                    subject_table[table_i,6]=int(len(labels))
                    subject_table[table_i,7]=int(len(labels[labels==0]))
                    subject_table[table_i,8]=int(len(labels[labels==1]))
                    subject_table[table_i,9]=int(len(labels[labels==2]))
                    '''
                    #We reproject the PCA to the original coordinates and save the reprojected and the originals to compare them laters
                    reproj= vh[:3,:].T @ pca.T + mean.reshape((-1,1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m0.npy',reproj[:,labels==0].mean(axis=1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m1.npy',reproj[:,labels==1].mean(axis=1))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/reprojected_means_m2.npy',reproj[:,labels==2].mean(axis=1))
                            
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m0.npy',PC[labels==0].mean(axis=0))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m1.npy',PC[labels==1].mean(axis=0))
                    np.save(subj_dir+space+'/'+band_dic[i_band]+'/session'+str(bloc_i)+'/original_means_m2.npy',PC[labels==2].mean(axis=0))
                    '''

                    #Now we can use Topology om order to classify trials depending on how much they change the topology of each Point Cloud of motivational States
                    print('intensities for band ', band_dic[i_band], 'and session', bloc_i)
                    subject_table[table_i,10],random_predictions_matrix,max_acc[bloc_i-1,i_band]=tda_intensity_classifier(subj_dir,space+'/'+band_dic[i_band]+'/session'+str(bloc_i),pca,labels,i_band)
                    bloc_i=bloc_i+1
