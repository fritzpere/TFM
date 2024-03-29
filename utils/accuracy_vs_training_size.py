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
import numpy as np
import sklearn.model_selection as skms
from collections import defaultdict
import sklearn.metrics as skm
import sklearn.neighbors as sklnn
import matplotlib.pyplot as plt
from scipy.io import savemat



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

                    #We save on our subject table the accumulated variance within the 3 most important dimensions.
                    subject_table[table_i,5]=acc_variance[3]
                    #Let us work with this 3-dimensional Point Cloud. 
                    pca = vh[:3,:] @ X[:,:]

                    pca=pca.T



                    pca,labels,PC=preprocessor.reject_outliers(pca,labels,PC,m=2) 
                    


                    #Now we can use Topology om order to classify trials depending on how much they change the topology of each Point Cloud of motivational States
                    print('intensities for band ', band_dic[i_band], 'and session', bloc_i)
                    tda_intensity_classifier(subj_dir,space+'/'+band_dic[i_band]+'/session'+str(bloc_i),pca,labels,i_band)
                    bloc_i=bloc_i+1
                    
                    




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



def tda_intensity_classifier(subj_dir,space,PC,labels,i_band):
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
    #feat_vect_names=['Landscapes','Silhouettes','Descriptors','Bottleneck']
    feat_vect_names=['Silhouettes']
    n_vectors=len(feat_vect)
    n_rep=10 
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'}
    band = band_dic[i_band]
    #Initiialize matrices where we will save several information (accuracies distribution, confusion matrix, random predictions matrix)
    rand_n=np.zeros((n_rep,n_vectors,n_dim))
    test_size=np.zeros(n_rep)
    topo_perf = np.zeros([n_dim,n_vectors,n_rep])
    
    knn_perf = np.zeros(n_rep)
    knn_conf_matrix =np.zeros((n_rep,3,3))
    #Initialize 1 Nearest Neighbor classifier
    clf= sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    if not os.path.exists(subj_dir+space+'/1nn_clf'):
        print("create directory(plot):",subj_dir+space+'/1nn_clf')
        os.makedirs(subj_dir+space+'/1nn_clf')
    #perf_shuf = np.zeros([n_dim,n_vectors+1,n_rep])
    topo_conf_matrix = np.zeros([n_dim,n_vectors,n_rep,3,3])

    if not os.path.exists(subj_dir+space+'/topological_clf'):
        print("create directory(plot):",subj_dir+space+'/topological_clf')
        os.makedirs(subj_dir+space+'/topological_clf')
    
    t_int=time.time()
    #We lool which motivational state has less points
    trials_per_m=min((labels==0).sum(),(labels==1).sum(),(labels==2).sum())
    if trials_per_m==0: #If there is a motivational state without a point we will not classify
        np.save(subj_dir+space+'/topological_clf/'+band+'perf_intensity.npy',topo_perf)
        np.save(subj_dir+space+'/1nn_clf/'+band+'perf_intensity.npy',knn_perf) 
        return -1,np.zeros((n_vectors,n_dim)),-1
    
    ##NEW
    final_size=trials_per_m
    sizes=[]
    matlab_matrix=[]
    for trials_per_m in range(50,final_size,10):
        train_size=int(trials_per_m*3*0.8)
        #We balabce the dataset by downsampling
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
                topo_pred_array=np.zeros((len(ind_test),n_vectors,n_dim,3))
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
                        #tda_vect[i_motiv][n_vectors][i_dim]=dim_persistence #Saving directly the persistence in one dimension (later we will compute Bottleneck distance)
                        
                '''#We normalize the descriptor Vector        
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
                    tda_vect[m][2][1]=descriptors1[m]'''
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
                                
                                topo_pred_array[i,i_vector,i_dim,i_motiv]=np.linalg.norm(tda_compt.transform([dimensional_persistence])-tda_vect[i_motiv][i_vector][i_dim])
                            '''
                            tda_compt=feat_vect[n_vectors-1]
                            tda_compt.fit([dimensional_persistence])
                            
                            topo_pred_array[i,n_vectors-1,i_dim,i_motiv]=np.linalg.norm(((tda_compt.transform([dimensional_persistence])-mins[i_dim])/(maxs[i_dim]-mins[i_dim]))-tda_vect[i_motiv][n_vectors-1][i_dim])
    
                            topo_pred_array[i,n_vectors,i_dim,i_motiv]=gd.bottleneck_distance(dimensional_persistence,tda_vect[i_motiv][n_vectors][i_dim],0.01)'''
                    i=i+1
            
                #We predict and compute accuracy and confusion martix
                for i_vector in range(n_vectors):
                    for i_dim in range(n_dim):
                        topo_pred,rand_n[i_rep,i_vector,i_dim]=topological_clf(topo_pred_array[:,i_vector,i_dim,:])
    
                        topo_perf[i_dim,i_vector,i_rep] = skm.accuracy_score(topo_pred, labels_dwnsamp[ind_test])
                        #topo_conf_matrix[i_dim,i_vector,i_rep,:,:] += skm.confusion_matrix(y_true=labels_dwnsamp[ind_test], y_pred=topo_pred,normalize='true')               
        print((time.time()-t_int)/60, 'minuts for classification')
        
        sizes.append(train_size)
        matlab_matrix.append(topo_perf[0,0,:].copy())
    if not os.path.exists('results/intensities/fig10'):
        print("create directory(plot):",'results/intensities/fig10')
        os.makedirs('results/intensities/fig10')
    matlab_matrix=np.array(matlab_matrix)
    sizes=np.array(sizes)
    savemat('results/intensities/fig10/'+subj_dir[-3:-1]+band+space[-1]+'accuracies_per_test_size.mat', {'accuracies' :matlab_matrix})
    savemat('results/intensities/fig10/'+subj_dir[-3:-1]+band+space[-1]+'sizes.mat', {'train_sizes':sizes})
    np.save('results/intensities/fig10/'+subj_dir[-3:-1]+band+space[-1]+'accuracies_per_test_size.npy', matlab_matrix)
    np.save('results/intensities/fig10/'+subj_dir[-3:-1]+band+space[-1]+'sizes.npy', sizes)
    #We plot accuracies and confusion matrices for 1nn
        
        
    return



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
    