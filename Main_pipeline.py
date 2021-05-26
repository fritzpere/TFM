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
from joblib import Memory
from shutil import rmtree

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



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
        space='electrodeSpace'

        elec_space,subj_dir=load_data(subject,space=space)
        
        if not os.path.exists(subj_dir+space):
            print("create directory(plot):",subj_dir+space)
            os.makedirs(subj_dir+'/'+space)
        
        print('cleaning and filtering data of electrode space of subject',subject)
        elec_space_preprocessor=Preprocessor(elec_space)
        filtered_ts_dic=elec_space_preprocessor.get_filtered_ts_dic()
        ts_band,labels=elec_space_preprocessor.get_trials_and_labels()
        
        band_dic={-1: 'noFilter', 0:'alpha',1:'betta',2:'gamma'}
        
        
        
        
        bands=[-1,0,1,2]
        n_band=len(bands)
        #measures=["intensities","correlation","quaf","dtw"]
        measures=["dtw"]
        n_measure=len(measures)
        dimensions=["zero","one"]
        n_dim=len(dimensions)
        feat_vect=[DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()]

        n_vectors=len(feat_vect)
        
        
        ##NOOOOU
        '''
        N=ts_band.shape[-1]
        for i_band in bands:
            print('global picture of band',band_dic[i_band] )
            PC=np.abs(ts_band[:,i_band,:,:]).mean(axis=1)
            PC=PC.reshape((-1,N))##fer absolut abans de fer la mitja??
        
            matrix=cdist(PC,PC)
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence=(Rips_simplex_tree_sample.persistence())
        

            gd.plot_persistence_diagram(persistence)
            plt.savefig(subj_dir+space+'/'+band_dic[i_band]+'_Global_Persistence_diagram.png')'''

        
        resolut=1000
        '''
        for i_measure in range(n_measure):
            print('plotting topological descriptors with ', measures[i_measure])   
            
            fig, axes = plt.subplots(nrows=n_band, ncols=3, figsize=(90, 36))
            fig4, axes4 = plt.subplots(nrows=n_band, ncols=9, figsize=(36, 36))
            fig2, axes2 = plt.subplots(nrows=n_band*n_dim, ncols=3, figsize=(36, 36))
            fig3, axes3 = plt.subplots(nrows=n_band*n_dim, ncols=3, figsize=(36, 36))
            j=0
            for i_band in bands: 
                print('band',band_dic[i_band])
                exploratory_pipe = skppl.Pipeline([("band_election", Band_election(bands[i_band])),("persistence", PH_computer(measure=measures[i_measure]))])
                presistence=exploratory_pipe.fit_transform(ts_band)
                for i_dim in range(n_dim):
                    dimension_scaler=DimensionDiagramScaler(dimensions=dimensions[i_dim])
                    dim_vect=dimension_scaler.fit_transform(presistence)
                 

                    
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

            
            fig3.suptitle('Persistence Landscapes of the {0} for\n different frequency bands and motivational state of {1}'.format(space,measures[i_measure]),fontsize=24)
            fig3.tight_layout(pad=0.5)
            fig3.subplots_adjust(top=0.8)
            plt.savefig(subj_dir+space+'/'+measures[i_measure]+'Landscapes.png')
            plt.close()
            
            fig2.suptitle('Persistence Silhouette of the {0} for\n different frequency bands and motivational state of {1}'.format(space,measures[i_measure]),fontsize=24)
            fig2.tight_layout(pad=0.5)
            fig2.subplots_adjust(top=0.8)
            plt.savefig(subj_dir+space+'/'+measures[i_measure]+'Silhouette.png')
            plt.close()
                    
                    

            fig4.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension {2}'.format(space,measures[i_measure],dimensions[i_dim]),fontsize=24)
            fig4.tight_layout(pad=0.5)
            fig4.subplots_adjust(top=0.8)
            plt.savefig(subj_dir+space+'/'+measures[i_measure]+'one_topological_descriptors.png')
            plt.close()
                
        
            fig.suptitle('Topological descriptors of the {0} for\n different frequency bands and motivational state of {1} dimension 0'.format(space,measures[i_measure]),fontsize=24)
            fig.tight_layout(pad=0.5)
            fig.subplots_adjust(top=0.8)
            plt.savefig(subj_dir+space+'/'+measures[i_measure]+'zero_topological_descriptors.png')
            plt.close()'''
        
        
#descomentar lo d-adalt
        
        
    
            


        
        '''
        dimensions.append('both')
        n_dim+=1
        classifiers=[skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000))]),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')  ]
        n_classifiers=len(classifiers)
        
        
        cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        n_rep = 10 # number of repetitions
        
        perf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers]) # (last index: MLR/1NN)
        perf_shuf = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers])# (last index: MLR/1NN)
        conf_matrix = np.zeros([n_band,n_measure,n_dim,n_vectors,n_rep,n_classifiers,3,3]) # (fourthindex: MLR/1NN)
        
        
        pooling_list=[]
        for i_band in range(n_band):
            for i_measure in range(n_measure):
                for i_dim in range(n_dim):
                    for i_vector in range(n_vectors):
                        for i_classifier in range(n_classifiers):
                            topo_pipe= skppl.Pipeline([("band_election", Band_election(bands[i_band])),("persistence", PH_computer(measure=measures[i_measure])),("scaler",DimensionDiagramScaler(dimensions=dimensions[i_dim])),("TDA",feat_vect[i_vector]),('clf',classifiers[i_classifier])])
                            for i_rep in range(n_rep):
                                ind_tr ,ind_te=cv_schem.split(ts_band,labels)
                                pooling_list.append((ind_tr,ind_te,i_rep))
                            pool.map(evaluate, pooling_list)
                            
                                    
                            
                                    
        from multiprocessing import Pool                           
        def evaluate(inpu):
            ind_train,ind_test,i_rep=inpu
            print('band',bands[i_band],'measure',measures[i_measure],'dim',dimensions[i_dim],'vector',i_vector,'classifier',i_classifier,'repetition:',i_rep)
            
            topo_pipe.fit(ts_band[ind_train,:], labels[ind_train])
            perf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier] = topo_pipe.score(ts_band[ind_test,:], labels[ind_test])
            conf_matrix[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=topo_pipe.predict(ts_band[ind_test,:]))  
            
            
            shuf_labels = np.random.permutation(labels)
        
            topo_pipe.fit(ts_band[ind_train,:], shuf_labels[ind_train])
            perf_shuf[i_band,i_measure,i_dim,i_vector,i_rep,i_classifier]= topo_pipe.score(ts_band[ind_test,:], shuf_labels[ind_test])

    
        # save results       
        np.save(subj_dir+space+'/perf.npy',perf)
        np.save(subj_dir+space+'/perf_shuf.npy',perf_shuf)
        np.save(subj_dir+space+'/conf_matrix.npy',conf_matrix)                     
        
        
        fmt_grph = 'png'
        cmapcolours = ['Blues','Greens','Oranges','Reds']
        
        fig, axes = plt.subplots(nrows=n_band, ncols=n_vectors*n_measure*n_dim, figsize=(48, 24))
            
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
                        axes[i_band][i].set_ylabel('accuracy_'+str(band)+'_'+str(i_vector),fontsize=8)
                        axes[i_band][i].set_title(str(band)+', '+measures[i_measure]+dimensions[i_dim]+str(i_vector))
                        i=1+i
        fig.tight_layout(pad=0.5)
        plt.savefig(subj_dir+space+'/accuracies.png', format=fmt_grph)
        plt.close()
        

        
        fig2, axes2 = plt.subplots(nrows=n_band, ncols=n_vectors*n_measure*n_dim, figsize=(48, 24))
    
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
                        axes2[i_band][i].set_title(str(band)+', '+measures[i_measure]+dimensions[i_dim]+str(i_vector))
                        i=1+i
        fig.tight_layout(pad=0.5)
        plt.savefig(subj_dir+space+'/confusion_matrix.png', format=fmt_grph)
        plt.close()
        '''
        from joblib import Memory
        from shutil import rmtree
        
        location = 'cachedir'
        memory = Memory(location=location, verbose=10)
        topo_pipe= skppl.Pipeline([("band_election", Band_election()),("persistence", PH_computer()),("scaler",DimensionDiagramScaler()),("TDA",DimensionLandScape()),('clf',skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000))]))],memory=memory)
        
        parameters = {
            'band_election__band': [-1,0,1,2],
            'persistence__measure': ["dtw"],
            'scaler__dimension': ["zero","one",'both'],
            'TDA': [DimensionLandScape(),DimensionSilhouette(),TopologicalDescriptors()],
            'clf':[skppl.Pipeline([('Std_scal',skprp.StandardScaler()),('Clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000))]),sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation') ],
            }
        
        grid_search = skms.GridSearchCV(topo_pipe, parameters, n_jobs=22, verbose=1)
        print('got it')
        t=time.time()
        
        grid_search.fit(ts_band, labels)
        print((time.time()-t)/60, "on the gridsearch")
        print("Best score: %0.3f" % grid_search.best_score_)
        
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
        pd.DataFrame(grid_search.cv_results_).to_csv('prova.csv')
        '''
        memory.clear(warn=False)
        rmtree(location)'''

    
    else: 
        subjects=list(range(26,36))
        for subject in subjects:
            t=time.time()
            
            #elec_space,subj_dir=load_data(subject,space='electrodeSpace')
            elec_space,font_space,subj_dir=load_data(subject)
            
            print('computing persistence of electrode space of subject',subject)
            elec_space_pers=compute_persistence_from_EEG(elec_space,measure='intensities',subj_dir=subj_dir,space='electrodeSpace',save=True) #pointcloud shape (432, 50)
            
            print('computing persistence of electrode space (with correlations) of subject',subject)
            elec_space_pers_corr=compute_persistence_from_EEG(elec_space,measure='correlation',subj_dir=subj_dir,space='electrodeSpace',save=True) #pointcloud shape (432, 50)
            print('computing persistence of font space of subject',subject)
            font_space_pers=compute_persistence_from_EEG(font_space,measure='intensities',subj_dir=subj_dir,space='font_space',save=True)
            print('computing persistence of font space (with correlations) of subject',subject)
            font_space_pers_corr=compute_persistence_from_EEG(font_space,measure='correlation',subj_dir=subj_dir,space='font_space',save=True)
            
            print('plotting and saving data of subject',subject)
            elec_space_descriptor_vector_dic,labels=compute_topological_descriptors(elec_space_pers,subj_dir,space='electrodeSpace',measure='intensities')
            
            font_space_descriptor_vector_dic,labels=compute_topological_descriptors(font_space_pers,subj_dir,space='font_space',measure='intensities')
            elec_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(elec_space_pers_corr,subj_dir,space='electrodeSpace',measure='correlation')
            font_space_descriptor_vector_dic_corr,labels=compute_topological_descriptors(font_space_pers_corr,subj_dir,space='font_space',measure='correlation')
            
            get_accuracies_per_band(elec_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='electrodeSpace',measure='intensities')
            
            get_accuracies_per_band(font_space_descriptor_vector_dic,labels,subj_dir=subj_dir,space='font_space',measure='intensities')
            get_accuracies_per_band(elec_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='electrodeSpace',measure='correlation')
            get_accuracies_per_band(font_space_descriptor_vector_dic_corr,labels,subj_dir=subj_dir,space='font_space',measure='correlation')
        
               
            print((time.time()-t)/60, 'minuts for subject',subject)
