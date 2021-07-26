#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:36:48 2021

@author: fritzpere
"""

'''
Set of classes with all the functions needed to put on a sklearn pipeline
in order to do a topological analisis of EEGs and feed topological descriptors to classifiers
'''
import numpy as np
import gudhi as gd
from sklearn.preprocessing   import MinMaxScaler
from scipy.spatial.distance import cdist
from tslearn.metrics import dtw
from scipy import stats
from scipy.spatial.distance import pdist
import gudhi.representations as gdr


class Band_election:
    '''
    Band_election selects a given band and returns the corresponding TimeSeries
    '''
    def __init__(self,band=-1):
        self.band=band
        
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        return X[:,self.band,:,:]
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
          setattr(self, parameter, value)
        return self
    def get_params(self, deep=True):
        return {"band": self.band}
    
class PH_computer:
    '''
    PH_computer selects a given measure and 
    returns the corresponding Persistent Diagrams of dimensions 0 and 1'''

    def __init__(self,measure='intensities'):
        self.measure=measure

        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, plot=False):
        
        #First we need to compute a distance or semi-distance matrix 
    
        persistence=[]
        zero_dim=[]
        one_dim=[]
        for k in range(X.shape[0]):
            band_tensor = X[k,:,:].T
            #print('nans?',np.isnan(band_tensor[i]).any())
            #print('cloud shape:',band_tensor.shape)
            
            if self.measure=='euclidean':
                band_tensor = np.abs(band_tensor[:,:])
                matrix=cdist(band_tensor,band_tensor)
                '''
                n_coor = band_tensor.shape[0]
                matrix = np.zeros((n_coor, n_coor))
                row,col = np.triu_indices(n_coor,1)
                matrix[row,col] = pdist(band_tensor)'''
            elif self.measure=='correlation':

                T=1200
                ts_tmp = band_tensor.copy()
                ts_tmp -= np.outer(ts_tmp.mean(1),np.ones(T))
                matrix= np.tensordot(ts_tmp,ts_tmp,axes=(1,1)) / float(T-1)
                
                matrix/= np.sqrt(np.outer(matrix.diagonal(),matrix.diagonal())) ##Nomes falta això 
                matrix=np.arccos(matrix)
                
            elif self.measure=='dtw':
                fun=lambda x,y: dtw(x,y,global_constraint="sakoe_chiba", sakoe_chiba_radius=100)#motius biologics
                band_tensor = np.abs(band_tensor[:,:])
                #matrix=cdist(band_tensor,band_tensor,fun)
                
                n_coor = band_tensor.shape[0]
                matrix = np.zeros((n_coor, n_coor))
                row,col = np.triu_indices(n_coor,1)
                distancies=pdist(band_tensor,fun)
                matrix[row,col] = distancies
                matrix[col,row] = distancies
                
            else:
                fun=lambda x,y: stats.linregress(x,y)[2]**2 #quaf deter
                #matrix=cdist(band_tensor,band_tensor,fun)
                
                n_coor = band_tensor.shape[0]
                matrix = np.ones((n_coor, n_coor))
                row,col = np.triu_indices(n_coor,1)
                distancies=pdist(band_tensor,fun)
                matrix[row,col] = distancies
                matrix[col,row] = distancies
                matrix=1-matrix
                
                
            '''
            Then we compute the persistence of a 
            filtration of Rips Complex given the 
            before computed Distance Matrix
            '''
            #max_edge=np.max(matrix)
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence.append(Rips_simplex_tree_sample.persistence())
            #We separate it into dimensions and remove the feature that persist to infinity to be able to calculate descriptors later on
            dim_list=np.array(list(map(lambda x: x[0], persistence[k])))
            point_list=np.array(list(map(lambda x: x[1], persistence[k])))
            zero_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)])
            one_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)])
        
        return (zero_dim,one_dim)


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
          setattr(self, parameter, value)
        return self
    
    def get_params(self, deep=True):
        return {"measure": self.measure}
    
    
    
    

class DimensionDiagramScaler(gdr.DiagramScaler):
    '''
    Just a MinMaxScaler that also select which homology dimensions to consider. 
    Dimension 0, Dimension 1, or both.
    '''
    def __init__(self,dimensions='both',use=False):
        super().__init__(use,scalers=[([0,1], MinMaxScaler())])
        self.dimensions=dimensions
        
    def fit(self,X,y=None):
        super().fit(X,y)
        return self
    
    def transform(self, X):
        if self.dimensions=='both':
            return super().transform(X[0]),super().transform(X[1])
        elif self.dimensions=='zero':
            return super().transform(X[0])
        else:
            return super().transform(X[1])
   
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
          setattr(self, parameter, value)
        return self
    
    def get_params(self, deep=True):
        dic=super().get_params()
        dic["dimensions"]=self.dimensions
        return dic
        
class DimensionLandScape:
    '''
    It computes the Persistent Landscapes and discretizes them to vectors
    '''
    def __init__(self,num_landscapes=5,resolution=1000,L0=None,L1=None):
        self.n=num_landscapes
        self.resol=resolution
        if L0:
            self.L0=L0
        else:
            self.L0=gdr.Landscape(num_landscapes, resolution)
        if L1:
            self.L1=L1
        else:
            self.L1=gdr.Landscape(num_landscapes, resolution)
    
    def fit(self,X,y=None):
        if type(X)==tuple:
            self.L0.fit(X[0],y)
            self.L1.fit(X[1],y)
        else:
            self.L0.fit(X,y)
        return self
    
    def transform(self, X):
        if type(X)==tuple:
            landsc0=self.L0.transform(X[0])
            landsc1=self.L1.transform(X[1])

            return np.concatenate((landsc0,landsc1),axis=1)
        else:
            return self.L0.transform(X)
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
          setattr(self, parameter, value)
        return self
    
    def get_params(self, deep=True):
        return {"num_landscapes":self.n,"resolution":self.resol,"L0": self.L0,"L1": self.L1, }
        
class DimensionSilhouette:
    '''
    It computes the Persistent Silhouettes and discretizes them to vectors
    '''
    def __init__(self,resolut=1000,p=1,S0=None,S1=None):##canviar
        self.resol=resolut
        self.p=p
        if S0:
            self.S0=S0
        else:
            self.S0=gdr.Silhouette(resolution=resolut, weight=lambda x: np.power(x[1]-x[0],p))
        if S1:
            self.S1=S1
        else:
            self.S1=gdr.Silhouette(resolution=resolut, weight=lambda x: np.power(x[1]-x[0],p))
    
    def fit(self,X,y=None):
        if type(X)==tuple:
            self.S0.fit(X[0],y)
            self.S1.fit(X[1],y)
        else:
            self.S0.fit(X)
        return self
    def transform(self, X):
        if type(X)==tuple:
            sil0=self.S0.transform(X[0])
            sil1=self.S1.transform(X[1])
            return np.concatenate((sil0,sil1),axis=1)
        else:
             return self.S0.transform(X)

    def get_params(self, deep=True):
        return {"resolut":self.resol,"p":self.p,"S0": self.S0,"S1": self.S1,}

class TopologicalDescriptors:
    '''
    It computes different topological descriptors depending on the dimension
    ''' 

    def fit(self,X,y=None):
        return self
    
    
    def transform(self, X):
        
        if type(X)==tuple:

            vec0=get_feat_dim0(X[0])
            vec1=get_feat_dim1(X[1])
            return np.concatenate((vec0,vec1),axis=1)
        elif all([all(X[i][:,0]==0) for i in range(len(X))]):
            return self.get_feat_dim0(X)
        else:
           return self.get_feat_dim1(X) 


    def get_params(self, deep=True):
        return dict()    

    def get_feat_dim0(self,X):
        '''
        get the 0-dimensional features (Avarage Life, Standard Desviation of Life, and topological Entropy)
        '''
        n=len(X)
        avg_life,std_life,entropy=[],[],[]
        for k in range (n):
            life=np.array(list(map(lambda x: x[1]-x[0],X[k])))
            L=life.sum()
            n_lifes=life.shape[0]
            if n_lifes!=0:
               avg_life.append(L/n_lifes)
               std_life.append(life.std())
            else:
                avg_life.append(-1)
                std_life.append(-1)
            entropy.append(-np.array(list(map(lambda d: (d/L)*np.log(d/L) if L!=0 else -1,life))).sum())
        avg_life,std_life,entropy=np.array(avg_life).reshape(-1,1),np.array(std_life).reshape(-1,1),np.array(entropy).reshape(-1,1)    
        return np.concatenate(( avg_life,std_life,entropy),axis=1)
            
    def get_feat_dim1(self,X):
        '''
        get the 1-dimensional features (Same as 0-dimensional, plus Midlife, Birth and Death (avarages and STDs))
        '''
        n=len(X)
        avg_life,std_life,avg_midlife,std_midlife,avg_birth,std_birth,avg_death,std_death,entropy=[],[],[],[],[],[],[],[],[]
        for k in range(n):
            birth=X[k][:,0]
            death=X[k][:,1]
            life=np.array(list(map(lambda x: x[1]-x[0],X[k])))
            midlife=np.array(list(map(lambda x: (x[1]+x[0])/2,X[k])))
            L=life.sum()
            n_lifes=life.shape[0]
            if n_lifes!=0:
               avg_life.append(L/n_lifes)
               std_life.append(life.std())
               avg_midlife.append(midlife.mean())
               std_midlife.append(midlife.std())
               avg_birth.append(birth.mean())
               std_birth.append(birth.std())
               avg_death.append(death.mean())
               std_death.append(death.std())
            else:
                avg_life.append(-1)
                std_life.append(-1)
                avg_midlife.append(-1)
                std_midlife.append(-1)
                avg_birth.append(-1)
                std_birth.append(-1)
                avg_death.append(-1)
                std_death.append(-1)
        
            entropy.append(-np.array(list(map(lambda d: (d/L)*np.log2(d/L) if L!=0 else -1,life))).sum())
        avg_life,std_life,avg_midlife,std_midlife,entropy=np.array(avg_life).reshape(-1,1),np.array(std_life).reshape(-1,1),np.array(avg_midlife).reshape(-1,1),np.array(std_midlife).reshape(-1,1),np.array(entropy).reshape(-1,1)
        avg_birth,std_birth,avg_death,std_death=np.array(avg_birth).reshape(-1,1),np.array(std_birth).reshape(-1,1),np.array(avg_death).reshape(-1,1),np.array(std_death).reshape(-1,1)
        return np.concatenate(( avg_life,std_life,entropy,avg_midlife,std_midlife,avg_birth,std_birth,avg_death,std_death),axis=1)



class TopologicalDescriptorsNocl:
    '''
    It computes different topological descriptors depending 
    on the dimension but it doesn't return all the descriptors
    as a vector since this is for plotting the descriptors and not classfying'
    ''' 

    def fit(self,X,y=None):
        return self
    
    
    def transform(self, X):
        
        if type(X)==tuple:

            vec0=self.get_feat_dim0_Nocl(X[0])
            vec1=self.get_feat_dim1_Nocl(X[1])
            return vec0,vec1
        else:
            print('wrong class')

    def get_params(self, deep=True):
        return dict()    




    def get_feat_dim0_Nocl(self,X):
    
    
        life=np.array(list(map(lambda x: x[1]-x[0],X)))
        L=life.sum()
        n_lifes=life.shape[0]
    
        entropy=-np.array(list(map(lambda d: (d/L)*np.log(d/L) if L!=0 else -1,life))).sum()
    
        return life,entropy
            
    def get_feat_dim1_Nocl(self,X):
    
        birth=X[:,0]
        death=X[:,1]
        life=np.array(list(map(lambda x: x[1]-x[0],X)))
        midlife=np.array(list(map(lambda x: (x[1]+x[0])/2,X)))
        L=life.sum()
        n_lifes=life.shape[0]
    
        
        entropy=-np.array(list(map(lambda d: (d/L)*np.log2(d/L) if L!=0 else -1,life))).sum()
        return life,entropy,midlife,birth,death