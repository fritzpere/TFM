#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:36:48 2021

@author: fritzpere
"""
import numpy as np
import gudhi as gd
from sklearn.preprocessing   import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import gudhi.representations as gdr
class Band_election:
    def __init__(self,band=-1):
        self.band=band
        
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        return X[self.band]
    
class PH_computer:
    def __init__(self,measure='intensities'):
        self.measure=measure

        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        persistence=[]
        zero_dim=[]
        one_dim=[]
        for k in range(X.shape[0]):
            band_tensor = X[k,:,:].T
            #print('nans?',np.isnan(band_tensor[i]).any())
            #print('cloud shape:',band_tensor.shape)
            
            if self.measure=='intensities':
                band_tensor = np.abs(band_tensor[:,:])
                matrix=distance_matrix(band_tensor,band_tensor)

            else:
                '''
                points=band_tensor.copy()
                normalized_p=normalize(points-np.mean(points,axis=0),axis=1)
                matrix= normalized_p @ normalized_p.T
                matrix=1-matrix
                '''
                T=1200
                ts_tmp = band_tensor.copy()
                ts_tmp -= np.outer(np.ones(T),ts_tmp.mean(0))
                matrix= np.tensordot(ts_tmp,ts_tmp,axes=(0,0)) / float(T-1)
                
                matrix/= np.sqrt(np.outer(matrix.diagonal(),matrix.diagonal()))
                matrix=np.arccos(matrix)
            #max_edge=np.max(matrix)
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence.append(Rips_simplex_tree_sample.persistence())
        
            dim_list=np.array(list(map(lambda x: x[0], persistence[k])))
            point_list=np.array(list(map(lambda x: x[1], persistence[k])))
            zero_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)])
            one_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)])
        
        return zero_dim,one_dim
    
    
    
    

class DimensionDiagramScaler(gdr.DiagramScaler):
    def __init__(self,dimensions='both'):
        super().__init__(scalers=[([0,1], MinMaxScaler())])
        self.dimensions=dimensions
        
    def fit(self,X,y=None):
        #super().fit(X,y)
        return self
    
    def transform(self, X):
        if self.dimensions=='both':
            return super().transform(X[0]),super().transform(X[1])
        elif self.dimensions=='zero':
            return super().transform(X[0])
        else:
            return super().transform(X[1])
        
        
class DimensionLandScape:
    def __init__(self,num_landscapes=2,resolution=100):
        self.L0=gdr.Landscape(num_landscapes, resolution)
        self.L1=gdr.Landscape(num_landscapes, resolution)

        
        
    def fit(self,X,y=None):
        if type(X)==tuple:
            self.L0.fit(X[0],y)
            self.L1.fit(X[1],y)
        else:
            self.L0.fit(X)
    
    
    def transform(self, X):
        if type(X)==tuple:
            landsc0=self.L0.transform(X[0])
            landsc1=self.L1.transform(X[1])
            return (landsc0,landsc1)
        else:
             return L0.transform(X)
        
class DimensionSilhouette:
    def __init__(self,num_landscapes=2,resolution=100,p=1):
        self.S0=gdr.Silhouette(num_landscapes, resolution,weight=lambda x: np.power(x[1]-x[0],p))
        self.S1=gdr.Silhouette(num_landscapes, resolution,weight=lambda x: np.power(x[1]-x[0],p))

    def fit(self,X,y=None):
        if type(X)==tuple:
            self.S0.fit(X[0],y)
            self.S1.fit(X[1],y)
        else:
            self.S0.fit(X)
    
    def transform(self, X):
        if type(X)==tuple:
            sil0=self.S0.transform(X[0])
            sil1=self.S1.transform(X[1])
            return (sil0,sil1)
        else:
             return S0.transform(X)
    