import numpy
# from fastdist import fastdist
import pandas
from numba import jit,njit, vectorize
from numba.typed import List

"""
Reimplmentation of silhouettescore calculated with numba
Distance matrix can be calculated with fastdist library or even our own numba_gower 
It is good only for 2 clusters...
"""

@njit(fastmath=True)
def apply_label_vector(x,y):
   return numpy.multiply(x,y)

@njit(fastmath=True)
def apply_label_vector_c(x,y,c0,c1, base_label):
    if(base_label == 0):
        a =  numpy.sum(numpy.multiply(x,1-y))/(c0) if c0 > 0 else 1  
        b = numpy.sum(numpy.multiply(x,y))/(c1+1) 
    else:
        a =  numpy.sum(numpy.multiply(x,y))/(c1) if c1 > 0 else 1  
        b = numpy.sum(numpy.multiply(x,1-y))/(c0+1)  
    
    # print("row " , a, " :: " , b)
    c = max(a,b)
    return (b-a)/c

# this will only get the silhouette score for two clusters
@njit(fastmath=True)
def silhouette_scoreA(X_distances,labels ):
   
    res = numpy.zeros( (X_distances.shape[0] , 3) ) # 3--- one for a, one for b, and one for the max
    c0  = numpy.sum(1-labels)-1
    c1  = numpy.sum(labels)-1
    for row in range(X_distances.shape[0]):
        if( labels[row] == 0 ):
            res[row][0] = numpy.sum( apply_label_vector(X_distances[row] , 1-labels ) )/(c0) if c0 > 0 else 1
            res[row][1] = numpy.sum( apply_label_vector(X_distances[row] , labels ) )/(c1+1)
        else:
            res[row][0] = numpy.sum( apply_label_vector(X_distances[row] , labels ) )/(c1) if c1 > 0 else 1
            res[row][1] = numpy.sum( apply_label_vector(X_distances[row] , 1-labels ) )/(c0+1)
        
        res[row][2] = max(res[row][1],res[row][0])
        res[row][2] = (res[row][1]-res[row][0])/res[row][2]

    return numpy.mean(res[:,2])

# this will only get the silhouette score for two clusters
@njit(fastmath=True)
def silhouette_scoreB(X_distances,labels ):
   
    res = numpy.zeros( X_distances.shape[0] ) # 3--- one for a, one for b, and one for the max
    c0  = numpy.sum(1-labels)-1
    c1  = numpy.sum(labels)-1
    for row in range(X_distances.shape[0]):
       res[row] = apply_label_vector_c(X_distances[row], labels, c0, c1, labels[row])


    return numpy.mean(res)