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


from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)

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
    
    if space=='electrode_space':
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        return elec_space,subj_dir
    elif space=='font_space':
        font_space=raw_data['ic_data3']
        return font_space,subj_dir
    else:
        elec_space=raw_data['dataSorted'] # [N,T,n_trials,motiv] 
        font_space=raw_data['ic_data3']
        return elec_space,font_space,subj_dir
    

if __name__ == "__main__":
    debug=True    
    if debug: 
        subject=25
        elec_space,subj_dir=load_data(subject,space='electrode_space')
        print('cleaning and filtering data of electrode space of subject',subject)
        elec_space_preprocessor=Preprocessor(elec_space)
        filtered_ts_dic=elec_space_preprocessor.get_filtered_ts_dic()
        ts_band,labels=elec_space_preprocessor.get_trials_and_labels()
        
        ##Exploratory analisis with filtered_ts_dic
        
        
        
        
        bands=[-1,0,1,2]
        n_band=len(bands)
        measures=["intensities"]
        n_measure=len(measures)
        dimensions=["both","zero","one"]
        n_dim=len(dimensions)
        feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]
        n_vectors=len(feat_vect)
        classifiers=[skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))]),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation'),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute')  ]
        n_classifiers=len(classifiers)
        
        
        cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        n_rep = 3 # number of repetitions
        
        perf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers]) # (last index: MLR/1NN)
        perf_shuf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers])# (last index: MLR/1NN)
        conf_matrix = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers,3,3]) # (fourthindex: MLR/1NN)
        
        for i_band in range(n_band):
            for i_measure in range(n_measure):
                for i_dim in range(n_dim):
                    for i_vector in range(n_vectors):
                        for i_classifier in range(n_classifiers):
                            for i_rep in range(n_rep):
                                for ind_train, ind_test in cv_schem.split(ts_band,labels): # false loop, just 1 
                                    print('band',bands[i_band],'measure',measures[i_measure],'dim',dimensions[i_dim],'vector',i_vector,'classifier',i_classifier,'repetition:',i_rep)

                                    pipe_MLR = skppl.Pipeline([("band_election", Band_election(bands[i_band])),("persistence", PH_computer(measure=measures[i_measure])),("scaler",DimensionDiagramScaler(dimensions=dimensions[i_dim])),("TDA",feat_vect[i_vector]),('clf',classifiers[i_classifier])])
                                    
                                    pipe_MLR.fit(ts_band[ind_train,:], labels[ind_train])
                                    
                                    perf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier] = pipe_MLR.score(ts_band[ind_test,:], labels[ind_test])
                                    conf_matrix[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=pipe_MLR.predict(ts_band[ind_test,:]))  
                                    
                                    
                                    shuf_labels = np.random.permutation(labels)
            
                                    pipe_MLR.fit(ts_band[ind_train,:], shuf_labels[ind_train])
                                    perf_shuf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier]= pipe_MLR.score(ts_band[ind_test,:], shuf_labels[ind_test])
                            
                                    
                            

    
    
        # save results       
        np.save(subj_dir+'/perf.npy',perf)
        np.save(subj_dir+'/perf_shuf.npy',perf_shuf)
        np.save(subj_dir+'/conf_matrix.npy',conf_matrix)                     
        
        band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
        fmt_grph = 'png'
        cmapcolours = ['Blues','Greens','Oranges','Reds']
        
        fig, axes = plt.subplots(nrows=n_band, ncols=n_vectors*n_measure*n_dim, figsize=(24, 24))
            
        for i_band in bands:
            band = band_dic[i_band]
            i=0
            for i_measure in range(n_measure):
                for i_dim in range(n_dim):
                    for i_vector in range(n_vectors):
                        
                        # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
                        chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
                    
                        # plot performance and surrogate
                        #axes[i_band][i_vector].axes([0.2,0.2,0.7,0.7])
                        axes[i_band][i].violinplot(perf[i_band,i_measure,i_dim,i_vector,:,0],positions=[-0.2],widths=[0.3])
                        axes[i_band][i].violinplot(perf[i_band,i_measure,i_dim,i_vector,:,1],positions=[0.2],widths=[0.3])
                        axes[i_band][i].violinplot(perf[i_band,i_measure,i_dim,i_vector,:,2],positions=[0.6],widths=[0.3])
                        
                        axes[i_band][i].violinplot(perf_shuf[i_band,i_measure,i_dim,i_vector,:,0],positions=[1.2],widths=[0.3])
                        axes[i_band][i].violinplot(perf_shuf[i_band,i_measure,i_dim,i_vector,:,1],positions=[1.6],widths=[0.3])
                        axes[i_band][i].violinplot(perf_shuf[i_band,i_measure,i_dim,i_vector,:,2],positions=[2],widths=[0.3])
                        
                        axes[i_band][i].plot([-1,2],[chance_level]*2,'--k')
                        axes[i_band][i].axis(xmin=-0.6,xmax=2.4,ymin=0,ymax=1.05)
                        #axes[i_band][i].set_xticks([0,1,2,3],['MLR','1NN','control1','control2'])##Provar
                        axes[i_band][i].set_ylabel('accuracy_'+band+'_'+str(i_vector),fontsize=8)
                        axes[i_band][i].set_title(band+', '+measures[i_measure]+dimensions[i_dim]+i_vector)
                        i=1+i
        plt.savefig(subj_dir+'accuracies.png', format=fmt_grph)
        plt.close()
        '''
        fig2, axes2 = plt.subplots(nrows=4, ncols=n_vector, figsize=(24, 24))
    
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
        plt.savefig(subj_dir+space+'/'+measure+'/conf_matrix/'+name+'confusion_matrix_MLR.png', format=fmt_grph)
        plt.close()'''
        
        
        
        
        '''
        parameters = {
            'band_election__band': (-1,0,1,2),
            'persistence__measure': ("intensities","correlation"), 
            'scaler__use': (True,False),
            'scaler__dimension': ("both","zero","one"),
            'TDA__num_landscapes': (2,3,5),
            'TDA__resolution': (100,1000),}
        
        grid_search = skms.GridSearchCV(pipe, parameters, n_jobs=8, verbose=1)
        print('got it')
        t=time.time()
        grid_search.fit(ts_band, labels)
        print((time.time()-t)/60, "on the gridsearch")
        print("Best score: %0.3f" % grid_search.best_score_)
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
        pd.DataFrame(grid_search.cv_results_).to_csv('prova.png')'''






        
      
            
           
       
    
    else: 
        subjects=list(range(26,36))
        for subject in subjects:
            t=time.time()
            
            #elec_space,subj_dir=load_data(subject,space='electrode_space')
            elec_space,font_space,subj_dir=load_data(subject)
            
            print('computing persistence of electrode space of subject',subject)
            elec_space_pers=compute_persistence_from_EEG(elec_space,measure='intensities',subj_dir=subj_dir,space='electrode_space',save=True) #pointcloud shape (432, 50)
            
            print('computing persistence of electrode space (with correlations) of subject',subject)
            elec_space_pers_corr=compute_persistence_from_EEG(elec_space,measure='correlation',subj_dir=subj_dir,space='electrode_space',save=True) #pointcloud shape (432, 50)
            print('computing persistence of font space of subject',subject)
            font_space_pers=compute_persistence_from_EEG(font_space,measure='intensities',subj_dir=subj_dir,space='font_space',save=True)
            print('computing persistence of font space (with correlations) of subject',subject)
            font_space_pers_corr=compute_persistence_from_EEG(font_space,measure='correlation',subj_dir=subj_dir,space='font_space',save=True)
            
            print('plotting and saving data of subject',subject)
            elec_space_descriptor_vector_dic,labels=compute_topological_descriptors(elec_space_pers,subj_dir,space='electrode_space',measure='intensities')
            
            font_space_descriptor_vector_dic,labels=compute_topological_descriptors(font_space_pers,subj_dir,space='font_space',measure='intensities')
            elec_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(elec_space_pers_corr,subj_dir,space='electrode_space',measure='correlation')
            font_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(font_space_pers_corr,subj_dir,space='font_space',measure='correlation')
            
            get_accuracies_per_band(elec_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='electrode_space',measure='intensities')
            
            get_accuracies_per_band(font_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='font_space',measure='intensities')
            get_accuracies_per_band(elec_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='electrode_space',measure='correlation')
            get_accuracies_per_band(font_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='font_space',measure='correlation')
        
               
            print((time.time()-t)/60, 'minuts for subject',subject)
