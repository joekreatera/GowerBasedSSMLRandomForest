import numpy
import uuid
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas
import math
import random
from time import time
from sklearn.manifold import TSNE 
from .utils import pairwise_distances

class SSClustering(BaseEstimator, ClassifierMixin):
    """
    This class generates clusters according to specific points. 
    Needs an array with the total set of instances and a matrix with the labels. All the unlabeled instances
    should have -1 in their row and 0|1 in the corresponding label. 

    The process will generate the n_clusters min the cosine distance of supervised instances lables and max the 
    distance between them. 
    The unsupervised instances will be added to their closed supervised instance. The n clusters are done by first
    comparing the N feature closest supervised instances and the N label furthest ones. 

    Algorithm:

    1. Determine M clusters (equal to the amount of supervised instances)
    2. Determine N final clusters according to feature and label space
    3. Merge M clusters with N determined centers according to their feature and label space. 
    
    On predict
    1. Add unsupervised instances to each cluster according to their distance on feature space to the cluster base supervised instance (this was changed from original idea because of performance, here we state the final impl)
    2. Return clusters = [{supervised base instance index, unsupervised instances indices}]

    By default the distance metric is euclidean taking every feature on X but a distance matrix can be set on constructor.
    and must have values between 0 and 1, being 0 the closest and 1 the furthest. 

    For distances on label space the default is cosine distance but a ditance matrix can be set on constructor. 

    If X is a pandas dataframe, and the parameter is confirmed, the return of transform will be an array of tuples 
    with the first part as the supervised indices of the clusters and the second as the unsupervised instances

    In any other case, the return of transform will be a numpy dataframe in the order that X was input with the following output>
    -i for unsupervised instances of cluster i
    i for supervised instances of cluster i

    So, for 3 clusters, the final numpy array will look like:
    [-1,-2,-1,-2,-3,1,2,3,1,2,3]

    If X is a pandas dataframe, the algorithm assumes that the feature distance and label distance will be also 
    pandas dataframes. 
    The label distance matrix will need also the distance between unsupervised and supervised instances. The distances between
    these types will not be considered. THis is a requirement to maintain the order of the matrices, which must be the same
    as the original X matrix.

    It is considered but not implemented to adapt a technique such as spectral clustering for this class. The idea would be to implement
    a graph inside and combine the minimization properties.

    On "y", the unsupervised instances must have -1 in all the row (for each label)


    This is the closest I found to the method, almost the same 
    Seed Clustering -> Constrained KMeans[1]. Like Seeded KMeans, it uses the seed clustering to
    initialize the centroids in KMeans. However, in subsequent steps, cluster labels
    of seed data are kept unchanged in the cluster assignment steps, and only the
    labels of the non-seed data are re-estimated. It is appropriate when the initial
    seed labeling is noise-free, or if the user does not want the labels of the seed data
    to change.


    """

    def __init__(self, n_clusters, feature_distance_matrix = None, label_distance_matrix = None, just_base_cluster=False, use_gower=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.just_base_cluster = just_base_cluster
        if( self.n_clusters > 2):
            print("Values different to 2, are not supported, clamping to 2.")
            self.n_clusters = 2

        self.feature_distance_matrix = feature_distance_matrix
        self.label_distance_matrix = label_distance_matrix
        self.original_index = None
        self.use_gower = use_gower

    

    def fit(self, X, y , far_distance_on_attribute_space = False, overwrite_label_distance = None):
        """
        X includes ONLY data , data frame 
        y includes only labels, dataframe with index as X
        """
        #t1 = time()

        if( type(X) is pandas.DataFrame):
            self.original_index = X.index.array
            X = X.to_numpy()
            if(self.feature_distance_matrix is not None and not type(self.feature_distance_matrix) is numpy.ndarray):
                raise Exception("All provided matrices should be the same type")
            if(self.label_distance_matrix is not None and not type(self.label_distance_matrix) is numpy.ndarray ):
                raise Exception("All provided matrices should be the same type")
        
        #print("SSCLustering init1 time:" , (time() - t1 ) )
        #t1 = time()

        if(type(y) is pandas.DataFrame ):
            y = y.to_numpy()

        if(self.feature_distance_matrix is None):
            if self.use_gower:
                # this is not tested with the new gower function
                self.feature_distance_matrix = pairwise_distances(X, metric='gower')
            else: # remember this is not normalized and we are not accounting for that in the next part
                self.feature_distance_matrix = pairwise_distances(X, metric='euclidean')
        
        if (overwrite_label_distance is not None):
            self.label_distance_matrix = overwrite_label_distance

        if(self.label_distance_matrix is None and overwrite_label_distance is None):
            self.label_distance_matrix = pairwise_distances( y , metric='cosine' )


        #print("SSCLustering init2 time:" , (time() - t1 ) )
        
        #t1 = time()

        supervised_indices = numpy.argwhere(y[:,0] > -1)
        
        # this will get repeated values, we just need the indices of the first columns
        supervised_indices = list(supervised_indices[:,0]) # very row, just the first column, and remove the repeated ones
        self.numpy_index_to_pandas_index_dict = {}
        if(self.original_index is not None):
            pandas_index = numpy.array(self.original_index)[ supervised_indices ]
            self.numpy_index_to_pandas_index = numpy.array([supervised_indices , pandas_index])
            
            #print("S Index " , self.numpy_index_to_pandas_index  )
        else:
            self.numpy_index_to_pandas_index = numpy.array([supervised_indices , supervised_indices])
        
        #print("SSCLustering second time:" , (time() - t1 ) )

        #t1 = time()

        # we need the value of the distance on attribute space, just between supervised
        a = 0.0 # this has been the best
        b = 1-a

        f_distances = self.feature_distance_matrix[ supervised_indices]
        f_distances = f_distances[:,supervised_indices]
        #print("f_distances  " , f_distances)
        l_distances = self.label_distance_matrix[supervised_indices]
        l_distances = l_distances[:,supervised_indices]
        #print("l_distances" , l_distances)
        # f_order = numpy.argsort(distances) # this is the order that is tractable to original index via supervised indices
        if( not far_distance_on_attribute_space): # then choose the closes in attribute space and furthest in label space
            f_distances = 1 - f_distances
        
        f_times_l = (1-f_distances)*a+l_distances*(1-a) # looking for the smallest f_distance with largest l_dist 
        
        #print("ftimes_l", f_times_l)
        order_on_row = numpy.argsort(f_times_l) # now we have, in the last column, the index of the largest value
        #print("sorting order" , order_on_row )
        #print("sorting" , numpy.sort(f_times_l))
        largest_by_row = numpy.sort(f_times_l)[:,-1] # value of the largest
        #print("row wise largest?" , largest_by_row)
        
        order_on_col = numpy.argsort(largest_by_row) # index of the largest

        index_a = order_on_col[-1]
        index_b = order_on_row[index_a, -1]

        #print(index_a, index_b) # indices of the instances that should be separated , is squered so any order is the same. 
        #orig_instance_index_1 = self.numpy_index_to_pandas_index[1,index_a]
        #orig_instance_index_2 = self.numpy_index_to_pandas_index[1,index_b]

        #print(orig_instance_index_1, orig_instance_index_2) # indices of the instances that should be separated , is squered so any order is the same. 
        
        a = 0.3 # this has been the best, but could be a hyperparameter
        b = 1-a
        inv_f_times_l = 1 - (1-f_distances)*a+l_distances*(1-a)
        # use this one if a == 0.0 
        # inv_f_times_l = 1 - f_times_l # greatest f fistance with smallest l distance , should be equal to the inverse of the previous one
        
        # now, the first colum has the closes one. 
        #print("inv ftimes_l\n", inv_f_times_l)
        order_on_row = numpy.argsort(inv_f_times_l) # now we have, in the last column, the index of the largest value
        cluster_0 = order_on_row[index_a,:]
        cluster_1 = order_on_row[index_b,:]
         
        #print("INV- sorting order" , order_on_row )
        # ordered instances for cluster 0 
        #print("ordered instances for cluster 0, without first and  last")

        cluster_0= cluster_0[1:]
        #print(cluster_0)
        #print("ordered instances for cluster 1, first is the other cluster, last is cluster base")
        cluster_1= cluster_1[1:]
        #print(cluster_1)

        if(random.random() > 0.5):
            final_cluster_0 = list(cluster_1.copy())
            final_cluster_1 = [cluster_0[-1]]
            size = len(cluster_1)
        else: # this was the original one
            final_cluster_0 = list(cluster_0.copy())
            final_cluster_1 = [cluster_1[-1]]
            size = len(cluster_0)


        #print("SSCLustering third time:" , (time() - t1) )

        # this cycle optimizes the balanced distance between the two
        # by updating the second cluster with the furthest one of the first

        # THIS IS NOT DONE for n_clusters > 2
        for instance_index in range(0,size-2 ): # do not do it with two or less elements on cluster 0
            to_transfer = final_cluster_0[0] # is always 0 as we are removing the first, every time
            final_cluster_0.remove(to_transfer)
            final_cluster_1.append(to_transfer)
            cluster_0_distances = numpy.average( inv_f_times_l[index_a, final_cluster_0] )
            cluster_1_distances = numpy.average( inv_f_times_l[index_b, final_cluster_1] )
            #print("cluster 0 distances: " , inv_f_times_l[index_a, cluster_0] , "with mean: " , cluster_0_distances , " set ",final_cluster_0 )
            #print("cluster 1 distances: " , inv_f_times_l[index_b, cluster_1] , "with mean: " , cluster_1_distances , " set ",final_cluster_1)
            if( cluster_0_distances > cluster_1_distances):
                break
        
        #print("final decision")
        #print("Supervised cluster 0" , final_cluster_0)
        #print("Supervised cluster 1" , final_cluster_1)

        self.clusters_index = []
        
        #print(self.numpy_index_to_pandas_index)
        #print(supervised_indices)
        self.clusters_index.append(  [ self.numpy_index_to_pandas_index[1][i] for i in final_cluster_0 ])
        self.clusters_index.append(  [ self.numpy_index_to_pandas_index[1][i] for i in final_cluster_1 ] )
        
        self.cluster_memory1 = [ self.numpy_index_to_pandas_index[1][i] for i in final_cluster_0 ]
        self.cluster_memory2 = [ self.numpy_index_to_pandas_index[1][i] for i in final_cluster_1 ]

        if(self.just_base_cluster):
            self.clusters_index = [ [ self.numpy_index_to_pandas_index[1][final_cluster_0[-1]] ] , [ self.numpy_index_to_pandas_index[1][final_cluster_1[-1]] ]  ]
            
            
        #print("Original index")
        # print(self.clusters_index)
        
        # distances from instances to each cluster 
        # this method could be enhaneced by taking the distance of the neighbors to the clusters owed instances

    def predict(self, X, distance_matrix = None, fixed_points = None):
        """
        X should include the supervised samples in the same original order
        if original X was a pandas dataframe, it will be looked up by index, so no need of the same order
        out put is a numpy array in the same order as input with just one column, 0 or 1, depending on the assigned cluster
        
        If distance matrix is provided, it must have the same rows and cols of X rows (squared matrix)

        If distance matrix is provided, everything should be in numpy arrays
        If distance matrix is provided, then fixed points should be here also. Is not tested for thing not pandas dataframe
        """
        if(type(X) is pandas.DataFrame):
            x_to_predict = X.to_numpy()
            fixed_points = pandas.DataFrame(index=X.index)
            fixed_points["base_cluster"] = 10
            fixed_points.loc[self.cluster_memory1,"base_cluster"] = -2 # corresponds to cluster 0
            fixed_points.loc[self.cluster_memory2,"base_cluster"] = -3 # corresponds to cluster 1
            fixed_points = fixed_points.to_numpy().ravel()
        else:
            x_to_predict = X
        final_prediction = numpy.zeros(shape=(x_to_predict.shape[0],self.n_clusters))
        
        #print("Cluster index " ,  self.clusters_index )
        for cluster in range(0,self.n_clusters): # the "2" should be the cluster quantity
            
            if( distance_matrix is None ): # use our own distance matrix
                #print(f"Indices of the cluster {cluster}" ,  self.clusters_index[cluster])
                #print(X)
                #print("cluster " , cluster)
                supervised_feature_space = X.loc[self.clusters_index[cluster]].to_numpy()
                #print("supervised_feature_space "  , supervised_feature_space)
                if(self.use_gower):
                    # yet to be answered:
                    # what about the min and max that should be considered form the while local dataset. 
                    # we should input them here
                    #
                    distance_matrix1 = pairwise_distances(supervised_feature_space,x_to_predict, metric="gower")
                    distance_matrix1 = distance_matrix1.reshape(-1,1) # it comes in one row
                else:
                    distance_matrix1 = pairwise_distances(x_to_predict, supervised_feature_space, metric="euclidean")
                #print(distance_matrix1)
            else:
                # the 2 should be the amount of clusters
                # this is NOT TESTED! 
                distance_matrix1 = distance_matrix[:,self.clusters_index[cluster]] 

            # print(distance_matrix1)
            closest_ones = numpy.sort(distance_matrix1) # from lowest to highest
            #print("closest_ones\n",closest_ones)
            final_prediction[:,cluster] = closest_ones[:,0]
        
        #print("Final prediction;: " , final_prediction)
        final_prediction = numpy.argsort(final_prediction)
        final_prediction = final_prediction[:,0]
        # print("************" , final_prediction)
        # print("++++++++++++++" , fixed_points)
        final_prediction = numpy.minimum(final_prediction, fixed_points)
        final_prediction = numpy.where( final_prediction == -2 , 0, final_prediction)
        final_prediction = numpy.where( final_prediction == -3 , 1, final_prediction)
        #print("--------------" , final_prediction)
        return final_prediction
    

if( __name__ == "__main__"):
    emotions = pandas.read_csv("../datasets/emotions.csv")
    print(emotions)
    sup = [0,3,5,7,9,11,13,15,17,19,21,23,27,29,31,33,35,37,41,43,45]
    unsup= [3,9,11,13,15,19,21,27,31,35,37]
    cols = ["col_0","col_1","col_2","col_3","col_4"]
    df = emotions[ cols + ["label_0", "label_1","label_2"] ]
    
    df = df.iloc[ sup ]

    print(df)
    def do_color(x):
        # print(x)
        a = x["label_0"]
        b = x["label_1"]
        c = x["label_2"]
        # print(a,b,c)
        a = 'F0' if a == 1 else '00'
        b = 'F0' if b == 1 else '00'
        c = 'F0' if c == 1 else '00'
        
        return '#'+a+b+c
    
    def do_cluster_color(x):
        # print(x)
        if(x["cluster"] == 0):
            return '#FF0000'
        
        return '#0000FF'
    

    tSNE = TSNE(n_components = 2, perplexity=16)
    transformed = tSNE.fit_transform(  df[ cols]  )
    import matplotlib.pyplot as plt
    df2 = pandas.DataFrame(transformed, columns=['x','y'], index=df.index)
    df2["label_0"] = df["label_0"]
    df2["label_1"] = df["label_1"]
    df2["label_2"] = df["label_2"]

    df2["color"] = df2.apply(do_color, axis=1) 
    print(df2)




    df.loc[ unsup, ["label_0","label_1","label_2"]] = -1
    c  = SSClustering(2)

    X = df[ cols ]
    y = df[ ["label_0","label_1","label_2"]]

    #print(X)
    #print(y)

    c.fit(X,y)
    clusters = c.predict(X)
    
    
    df["cluster"] = clusters
    df2["cluster"] = df.apply(do_cluster_color, axis=1) 
    print(df)

    plt.figure(figsize=(13,8))
    plt.scatter(df2["x"], df2['y'], c=df2["color"], s=200)
    plt.scatter(df2["x"], df2['y'], c=df2["cluster"], s=100)
    plt.show()
