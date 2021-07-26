#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:42:02 2021

@author: fritz
"""
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

import time  
import sklearn.pipeline as skppl
import sklearn.linear_model as skllm
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprp
import pandas as pd
import sklearn.neighbors as sklnn


def define_subject_dir(i_sub):
    """
    Creates the directory if it doesn't exist
    :param i_sub: subject id
    :return: directory path
    """
    res_dir = "results/conncectivity/subject_" + str(i_sub) + '/'
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
        return (elec_space,font_space),subj_dir
    


if __name__ == "__main__":
    subjects=[25,26,27]

    exploratory=True
    classification=True
    classification2=True
    band_dic={-1: 'noFilter', 0:'alpha',1:'beta',2:'gamma'}   
    bands=[2,1,0,-1] 
    n_band=len(bands)
    measures=["euclidean","correlation","quaf"]#,"dtw"]
    
    n_measure=len(measures)
    dimensions=["zero","one"]
    
    n_dim=len(dimensions)
    feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
    n_vectors=len(feat_vect)
    
    for subject in subjects:

        space='both'
        data_space,subj_dir=load_data(subject,space=space)
    
        spaces=['electrodeSpace','fontSpace']

        for sp in range(2):
            t=time.time()
            space=spaces[sp]
    
            if not os.path.exists(subj_dir+space):
                print("create directory(plot):",subj_dir+space)
                os.makedirs(subj_dir+'/'+space)
            print('cleaning and filtering data of',space,'of subject',subject)
            preprocessor=Preprocessor(data_space[sp])
            #filtered_ts_dic=preprocessor.get_filtered_ts_dic()
            ts_band,labels_original=preprocessor.get_trials_and_labels()
    
            
            
            resolut=1000
            if exploratory:
                t_expl=time.time()
                labels=labels_original

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
                    plt.close(fig3)
                    
                    fig2.suptitle('Persistence Silhouette of the {0} for\n different frequency bands and motivational state of {1}'.format(space,measures[i_measure]),fontsize=24)
                    fig2.tight_layout(pad=0.5)
                    fig2.subplots_adjust(top=0.8)
                    plt.savefig(expl_path+'/Silhouette.png')
                    plt.close(fig2)
                            
                            
        
                    fig4.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension {2}'.format(space,measures[i_measure],dimensions[i_dim]),fontsize=24)
                    fig4.tight_layout(pad=0.5)
                    fig4.subplots_adjust(top=0.8)
                    plt.savefig(expl_path+'/one_topological_descriptors.png')
                    plt.close(fig4)
                        
                
                    fig.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension 0'.format(space,measures[i_measure]),fontsize=24)
                    fig.tight_layout(pad=0.5)
                    fig.subplots_adjust(top=0.8)
                    plt.savefig(expl_path+'/zero_topological_descriptors.png')
                    plt.close(fig)
            
            
        
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
                                    
                        print((time.time()-t_mes)/60, 'minuts for',measures[i_measure], 'in band '+band_dic[i_band])
        
                                                                        
                                    
                                            
        
            
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
                                axes[i_band][i].set_ylabel('accuracy '+band_dic[i_band]+' '+str(i_vector),fontsize=8)
                                axes[i_band][i].set_title(band_dic[i_band]+' '+measures[i_measure]+dimensions[i_dim]+str(i_vector))
                                i=1+i
                plt.savefig(subj_dir+space+'/accuracies.png', format=fmt_grph)
                plt.close(fig)
                
        
                
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
                fig2.tight_layout(pad=0.5)
                plt.savefig(subj_dir+space+'/confusion_matrix.png', format=fmt_grph)
                plt.close(fig2)
                print('======TIME======') 
                print((time.time()-t_clf)/60, 'minuts for classification')
            if classification2:
                t_last=time.time()
                #from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
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
                plt.close(fig)
                    
            
                    
                fig2, axes2 = plt.subplots(nrows=n_band, ncols=2, figsize=(24, 24))
            
                for i_band in bands:
                    band = band_dic[i_band]
                    for i_clf in range (2):
                        # plot performance and surrogate
                        #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                        axes2[i_band][i_clf].imshow(conf_matrix[i_band,i_clf,:,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
                        #plt.colorbar()
                        axes2[i_band][i_clf].set_xlabel('true label',fontsize=8)
                        axes2[i_band][i_clf].set_ylabel('predicted label',fontsize=8)
                        axes2[i_band][i_clf].set_title(band+str(i_clf))
                    
                fig2.tight_layout(pad=0.5)
                plt.savefig(subj_dir+space+'/confusion_matrix2.png', format=fmt_grph)
                plt.close(fig2)
                print('======TIME======') 
                print((time.time()-t_last)/60, 'minuts for last') 
                                       
            print('======TIME======')
            print('======TIME======')
                                       
