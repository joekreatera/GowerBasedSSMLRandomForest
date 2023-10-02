import itertools
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .DTNode import DecisionTree, DecisionTreeNodeV2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
import pprint
import pandas as pd
import sys
from numpy.random import default_rng
import math
import threading
from multiprocessing import Process
import random
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel,check_pairwise_arrays, euclidean_distances, pairwise_distances
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from ksm.UD3_5Clustering import UD3_5Clustering
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
from  sklearn import svm

from joblib import Parallel, delayed
from multiprocessing import shared_memory
LEFT_SIMPLE_SUM = 0
RIGHT_SIMPLE_SUM = 1
LEFT_SQUARED_SUM = 2
RIGHT_SQUARED_SUM = 3
LEFT_TOTAL = 4
RIGHT_TOTAL = 5
TOTAL_VARIANCE = 6
LEFT_VARIANCE = 7
RIGHT_VARIANCE = 8
ITERATION_VALUE = 9
BEST_ITERATION_VALUE = 10
BEST_POSITION_INDEX = 11
BEST_ORIGINAL_INDEX = 12
SAME_VALUE_COLUMN = 13


class SSMLKVForestPredictor(BaseEstimator, ClassifierMixin):
    """
    Will not implement as an extension of MLSSVRPRedictor due to changes on main fit algorithm. They will look alike, but we need to separate some common funcionalities.
    """
    
    def __init__(self, confidence=0.2, leaf_relative_instance_quantity=0.05, unlabeledIndex=None, rec=5000, bfr = './', tag='vrssml', complete_ss = True,  
    trees_quantity = 1 ,  is_multiclass = False , do_random_attribute_selection = False,
    alpha = 10, C = 0.01,  gamma = 10, hyper_params_dict = None, do_ranking_split = False, p=[.5,0,0,.5], q=[1,0,0], use_complex_model=False, njobs=None):
        super().__init__()
        self.confidence = confidence
        self.trees = []
        self.random_generator = default_rng()
        self.tree = DecisionTree()
        self.tree_init = False
        self.output_file = None
        self.unlabeledIndex = unlabeledIndex
        self.ready = False
        self.max_recursion = rec
        self.dataset_ones_distribution = None
        
        if( hyper_params_dict is None):
            self.node_hyper_params = {
                'alpha' : alpha, # not working right now
                'C' : C, # not working right now
                'gamma' : gamma,
                'pA' : p[0],
                'pB' : p[1],
                'pC' : p[2],
                'pD' : p[3],
                'qA' : q[0],
                'qB' : q[1],
                'qC' : q[2],
                'do_ranking_split':do_ranking_split,
                'use_complex_model':use_complex_model,
                'purity_op':max
            }
        else: 
            self.node_hyper_params = hyper_params_dict
            
            self.node_hyper_params['pA'] = hyper_params_dict['p'][0]
            self.node_hyper_params['pB'] = hyper_params_dict['p'][1]
            self.node_hyper_params['pC'] = hyper_params_dict['p'][2]
            self.node_hyper_params['pD'] = hyper_params_dict['p'][3]
            self.node_hyper_params['qA'] = hyper_params_dict['q'][0]
            self.node_hyper_params['qB'] = hyper_params_dict['q'][1]
            self.node_hyper_params['qC'] = hyper_params_dict['q'][2]
            
            leaf_relative_instance_quantity = hyper_params_dict["leaf_relative_instance_quantity"]
            trees_quantity = hyper_params_dict["trees_quantity"] 
            self.do_ranking_split = hyper_params_dict["do_ranking_split"] 
            
        self.trees_quantity = trees_quantity # at least one! solution with 1 decision tree should be equals to MLSSVRPRedictor
        self.leaf_relative_instance_quantity = leaf_relative_instance_quantity
        
        self.base_file_root  = bfr
        self.tag = tag
        self.complete_ss = complete_ss
        self.is_multiclass = is_multiclass
        self.do_random_attribute_selection = do_random_attribute_selection
        
    def get_leaf_instance_quantity(self):
        return self.instances.shape[0]*self.leaf_relative_instance_quantity


    def get_supervised_relative_random_sampling(self, label_set,bagging_amount_pct = 1):
        """
        Return a sample  based on sampling code on 
        "A Consolidated Decision Tree-Based Intrusion Detection System for Binary and Multiclass Imbalanced Datasets"
        
        Wc[P]= 100  - [ sfC[p]/|stepSc| *100]
        desired sample weight for class P = 100 - frequency of class p divided by total amount of instances times 100 
        
        return the index of the instances to be selected
        """
        supervised_samples = label_set[ (label_set > 0).any(axis=1) ]
        unsupervised_samples = label_set[ (label_set == -1).any(axis=1) ]
        supervised_samples = supervised_samples.fillna(0)
        class_freq = supervised_samples.sum(axis = 0)/supervised_samples.shape[0]
        class_inv_freq = (1 - class_freq)*label_set.shape[0]    
        class_freq = class_freq*supervised_samples.shape[0]
        
        all_indices = set({})
        for index,value in class_inv_freq.iteritems() :
            original_c = class_freq.loc[index]
            indices = random.sample( list( supervised_samples[ supervised_samples[index] == 1 ].index.array  ) ,  int(min( int(value) ,  int(original_c)   )*bagging_amount_pct)  ) 
            all_indices = all_indices.union(set(indices))
        
    
        s_unsup = random.sample( list(unsupervised_samples.index.array) , int(len(unsupervised_samples.index)*bagging_amount_pct  ) ) # this was like 1 , not *bagging_amount_pct
        all_indices = all_indices.union(set( s_unsup  ) )


        # to test a really dumb selection
        #all_indices = random.sample( list( label_set.index.array  ) ,  int(len(label_set.index.array)*bagging_amount_pct)  ) 
        # print(all_indices)

        # print(len(all_indices))  
        # final_set = label_set.loc[pd.Index(list(all_indices))]
        
        # class_freq = final_set.sum(axis = 0)
        # print(class_freq)
        # summ = pd.concat( [ summ , class_freq ] , axis = 1, ignore_index = True)
        # print(summ)
        return pd.Index(list(all_indices))

    def get_random_columns(self, features_count, selectable_columns, feature_graph = None, linear_selection=False):
        
        random_columns = []
        if not ( feature_graph is None):
            if( features_count > 0):
                # get the indices 
                start_feature_index = np.random.randint(0, features_count )
                start_feature = selectable_columns[start_feature_index]
                # print(f'random feature index {start_feature_index} -> {start_feature}')
                feature_column = self.feature_graph[start_feature, :]
                # print(f'{feature_column} vs {selectable_columns}')
                indexes_to_pick_from_forward = np.argsort(feature_column)[::-1] # most related
                indexes_to_pick_from_reverse = indexes_to_pick_from_forward[::-1] # least related
                
                # print(indexes_to_pick_from)
                amount_to_pick = int(math.log( features_count,2 ))
                
                # this should select only the selectable columns 
                all_indices = set(indexes_to_pick_from_forward)
                selected_indices = set(selectable_columns)
                to_remove = all_indices - selected_indices 
                
                picking_index = []
                
                f_i = 0
                b_i = 0
                element = 0
                a = np.random.randint(0,10)
                    
                while element < len(indexes_to_pick_from_forward):
                    # if element%2==0: # some days select the most related and some days select the least related
                    if a > 5:
                        if( indexes_to_pick_from_forward[f_i] not in to_remove):
                            picking_index.append(indexes_to_pick_from_forward[f_i])
                            f_i += 1
                            element += 1
                        else:
                            f_i += 1
                    else:
                        if( indexes_to_pick_from_reverse[b_i] not in to_remove):
                            picking_index.append(indexes_to_pick_from_reverse[b_i])
                            b_i += 1
                            element += 1
                        else:
                            b_i += 1
                    if( b_i >= len(indexes_to_pick_from_reverse) or  f_i >= len(indexes_to_pick_from_forward) ):
                        element = len(indexes_to_pick_from_forward)
                        break;
                # picking_index = [elem for elem in indexes_to_pick_from if elem not in to_remove]

                random_columns = picking_index[0:amount_to_pick]
                random_columns2 = random.sample( selectable_columns ,  int(math.log( features_count,2 ))  )
                # print(f'------------------- [{start_feature}] {random_columns} vs {random_columns2} ({all_indices}) ({selected_indices}) was {indexes_to_pick_from_forward} removed {to_remove}')
            else:
                return [], True 
        else:
            #features_count = self.instances.shape[1]
            
            # random_columns = self.random_generator.integers(low=0, high=features_count, size= int(math.log( features_count,2 )) )
            
            # do select based on weights if feature_weights is provided
            if( features_count > 0):
                #random_columns = random.sample( selectable_columns ,  int(math.log( features_count,2 ))  )
                n_to_select = int(math.log( features_count,2 ))  if not linear_selection else features_count
                random_columns = random.sample( selectable_columns , n_to_select  )
            else:
                return [], True 
        
        return random_columns ,  False

    def get_average_distance(self, distance_matrix_df):
        
        # 0 is the same, 1 is the farthest
        half_matrix = np.tril(distance_matrix_df.to_numpy())
        n = distance_matrix_df.shape[0] - 1 # n its the amount of elements, n-1 would take out the distances of one to itself
        elements = n*(n+1)/2 # just get the diagonal of the matrix, originally n*(n+1)/2 from https://en.wikipedia.org/wiki/Summation
        if(elements==0):
            return   0.0 # is just the distance of one to itself
        sum = half_matrix.ravel().sum()/elements
        return sum
        # np.mean(half_matrix)
        
    def get_cluster_silhouette_score(self, X, labels, precomputed_distance = False):
        if(precomputed_distance):
            silhouette_precomputed = 1 - X.to_numpy()
            return 1-(silhouette_score(silhouette_precomputed, labels.to_numpy().ravel(), metric='precomputed')+1)/2
        
        return 1-(silhouette_score(X, labels)+1)/2
    
    def get_split_stats(self, cell, ordered_col, labelset):
        less_equal_than_index = ordered_col[ordered_col<=cell].index
        great_than_index = ordered_col[ordered_col>cell].index
        
        labels = labelset.loc[less_equal_than_index]
        # print(type(labels))
        less_equal_zeros = labels[ labels[labels.columns[0]] == 0].index
        less_equal_ones = labels[ labels[labels.columns[0]] == 1].index
        labels = labelset.loc[great_than_index]
        great_zeros = labels[ labels[labels.columns[0]] == 0].index
        great_ones = labels[ labels[labels.columns[0]] == 1].index
        
        left_total = len(less_equal_than_index)+0.000001
        right_total  = len(great_than_index)+0.000001
        
        
        left_purity = abs(len(less_equal_zeros) - len(less_equal_ones))/left_total # if this goes to 0, its totaally un-pure
        right_purity = abs(len(great_zeros) - len(great_ones))/right_total # if this goes to 0, its totaally un-pure
        total_purity = (left_purity + right_purity)*.5 # average of purity 
        
        #print(len(less_equal_than_index), "  --   " , len(great_than_index) , "     ", len(less_equal_zeros) , " __ " , len(great_zeros) , ">>" , left_purity , " & " ,  right_purity )
        return total_purity
                
        
    def get_best_split_by_column(self, col, labelset):
        # print(col)
        #print(labelset)
        #labelset = kwargs[0]
        o_col = col.sort_values(ascending=True)
        results = col.apply(self.get_split_stats, ordered_col= o_col,labelset=labelset ) # we are looking for the total purity
        results = results.sort_values(ascending=False)
        #print(results.head() )
        index = results.index[0]
        value = col.loc[index]
        purity = results.loc[0]
        #print(col.name,":",index,"<->",value,"<->",purity)
        #print("-")
        return [index, value,purity]


    def get_best_split_by_column_discretized_by_index(self, col, labelset , operation = max):
        #print(labelset)
        #labelset = kwargs[0]
        steps = 10
        o_col = col.sort_values(ascending=True)
        step = int(len(o_col.index)/steps)
        
        best_purity = -1 if operation  == max else 1
        best_value = -1
        best_index = o_col.index[0]
        simmetry_index = -1
        for i in range(0, len(o_col.index), step ):
            val = o_col.loc[o_col.index[ int(i) ]]
            less_equal_than_index = o_col[o_col<=val].index
            great_than_index = o_col[o_col>val].index
            # last index 
            
            
            
            # print(type(labels))
            total_purity = 0
            
            # and the rest of the labels!!! 
            for label_s in range(0, len(labelset.columns) ):
                labels = labelset.loc[less_equal_than_index]
                less_equal_zeros = labels[ labels[labels.columns[label_s]] == 0].index
                less_equal_ones = labels[ labels[labels.columns[label_s]] == 1].index
                labels = labelset.loc[great_than_index]
                great_zeros = labels[ labels[labels.columns[label_s]] == 0].index
                great_ones = labels[ labels[labels.columns[label_s]] == 1].index
                
                left_total = len(less_equal_than_index)+0.000001
                right_total  = len(great_than_index)+0.000001
            
            
                left_purity = abs(len(less_equal_zeros) - len(less_equal_ones))/left_total # if this goes to 0, its totaally un-pure
                right_purity = abs(len(great_zeros) - len(great_ones))/right_total # if this goes to 0, its totaally un-pure
                total_purity += (left_purity + right_purity)*.5 # average of purity 
                
            total_purity /= len(labels.columns)
            # greater or equal in order to update to a greater step
            
            if( operation(total_purity,best_purity) == total_purity  and len(less_equal_than_index)>0 and len(great_than_index) > 0 ):
                best_purity = total_purity
                best_value = val
                best_index = list(less_equal_than_index.array)
                best_index = best_index[-1]
                simmetry_index = abs(left_total-right_total)/o_col.shape[0]
            
        #if( best_purity == -1):
        #    print(f"--> Column size {o_col.shape}")
        return [best_index, best_value, best_purity, simmetry_index]





    def get_best_split_by_column_discretized(self, col, labelset, operation = max):
        
        #labelset = kwargs[0]
        steps = 20
        
        o_col = col.sort_values(ascending=True)
        
        best_purity = -1 if operation == max else 1
        best_value = -1
        best_index = o_col.index[0]
        simmetry_index = -1
        min_value = o_col.min()
        max_value = o_col.max()
        step = (max_value-min_value)/steps
        for i in range(1,steps):
            
            less_equal_than_index = o_col[o_col<=min_value + i*step].index
            great_than_index = o_col[o_col>min_value + i*step].index
            # last index 
            
            # print(type(labels))
            
            total_purity = 0
            
            # and the rest of the labels!!! 
            for label_s in range(0, len(labelset.columns) ):
                labels = labelset.loc[less_equal_than_index]
                less_equal_zeros = labels[ labels[labels.columns[label_s]] == 0].index
                less_equal_ones = labels[ labels[labels.columns[label_s]] == 1].index
                labels = labelset.loc[great_than_index]
                great_zeros = labels[ labels[labels.columns[label_s]] == 0].index
                great_ones = labels[ labels[labels.columns[label_s]] == 1].index
                
                left_total = len(less_equal_than_index)+0.000001
                right_total  = len(great_than_index)+0.000001
                
                # this tries to measure a balance in the sides
                sides_diff = (1 -  (left_total - right_total)/(left_total + right_total))*.5
                # this should penalize the purity, scale down.   
                
                left_purity = abs(len(less_equal_zeros) - len(less_equal_ones))/left_total # if this goes to 0, its totaally un-pure
                right_purity = abs(len(great_zeros) - len(great_ones))/right_total # if this goes to 0, its totaally un-pure
                total_purity += sides_diff*(left_purity + right_purity)*.5 # average of purity 
            
            total_purity /= len(labels.columns)
            
            # print(total_purity , "_" , len(less_equal_than_index) , " ", len(great_than_index) )
            # greater or equal in order to update to a greater step
            if( operation(total_purity,best_purity) == total_purity  and len(less_equal_than_index)>0 and len(great_than_index) > 0 ):
                best_purity = total_purity
                best_value = i*step
                best_index = list(less_equal_than_index.array)
                best_index = best_index[-1]
                simmetry_index = abs(left_total-right_total)/o_col.shape[0]
            
        #if( best_purity == -1):
        #    print(f"--> Column size {o_col.shape}")
        return [best_index, best_value, best_purity, simmetry_index]
        
    def get_best_split(self, distances_df, label_df, is_squared = True):
        
        # res = distances_df.apply(self.get_best_split_by_column, axis=0 ,result_type='expand', labelset=label_df ).T
        # just select a sub space of all the columns, just like random forest...
        d_df = distances_df.sample(n=int(math.log( distances_df.shape[1],2 )) ,axis='columns')

        
        if( is_squared ):
            
            res = d_df.apply(self.get_best_split_by_column_discretized, axis=0 ,result_type='expand', labelset=label_df ).T   
            res.sort_values(2, ascending=False, inplace=True)
            # print(res)
            r = res.to_numpy()
            r = r[0]
            # r = r.to_numpy()
            # thiw will generate an error towards best_split withoput discretization
            return int(r[0]),r[1],r[2],r[3] # 2 is impurity, 0 is node, 1 is distance value, 3 is simetry
                     
        else:
            res = d_df.apply(self.get_best_split_by_column_discretized_by_index, axis=0 ,result_type='expand', labelset=label_df ).T
            res.sort_values(2, ascending=False, inplace=True)
            best_col = res.index[0]
            r = res.to_numpy()
            r = r[0]
            # print(res)
            # r = r.to_numpy()
            # thiw will generate an error towards best_split withoput discretization
            # print(r)
            return best_col,r[1],r[2],r[3] # 2 is impurity, 0 is node, 1 is distance value

    def get_cluster_distance_based_best_split(self, distances_df, cluster_labels_df, original_label_df = None, is_squared = True, operation = max):
        
        label_df = cluster_labels_df
        silhouette_precomputed = 1 - distances_df.to_numpy()
        scores = silhouette_samples( silhouette_precomputed , label_df.to_numpy().ravel() , metric='precomputed')
        # with these scores, we should select with more probability the ones in the middle of the two clusters. 

        
        
        abs_scores = np.abs(scores) # let's assume that the order is the same as distances_df and label_df
        scores_df = pd.DataFrame(data = abs_scores , index = label_df.index )
        scores_df.sort_values(by=[0], ascending=True, inplace=True) # get all those close to 0 either left or right
        n_cols_to_select = int(math.log( scores_df.shape[0],2 ))
        index_to_select = scores_df.index.array[0:n_cols_to_select]
        # two options, leave it like this, or select just one of the best 
        ####### this is something to determine in other datasets
        # from these options is almost the same selecting any of these rather than a subset. Let's stay with the complex one, as it has better f1
        # index_to_select = random.sample( list(index_to_select) , 1) # this is one way of doing, the other is taking the total n_cols_to_select.. Just comment this line
        d_df = distances_df[ index_to_select ]
        
        # another idea: if these instances are the ones that divide the best, then test just them, not every instance??? 
        
        
        # res = distances_df.apply(self.get_best_split_by_column, axis=0 ,result_type='expand', labelset=label_df ).T
        # just select a sub space of all the columns, just like random forest...
        
        # d_df = distances_df.sample(n=int(math.log( distances_df.shape[1],2 )) ,axis='columns')
        label_df = cluster_labels_df if original_label_df is None else original_label_df
        
        if( is_squared ):
            
            res = d_df.apply(self.get_best_split_by_column_discretized, axis=0 ,result_type='expand', labelset=label_df, operation=operation ).T   
            res.sort_values(2, ascending=False, inplace=True)
            # print(res)
            r = res.to_numpy()
            r = r[0]
            # r = r.to_numpy()
            # thiw will generate an error towards best_split withoput discretization
            return int(r[0]),r[1],r[2],r[3] # 2 is impurity, 0 is node, 1 is distance value, 3 is simetry
                     
        else:
            res = d_df.apply(self.get_best_split_by_column_discretized_by_index, axis=0 ,result_type='expand', labelset=label_df, operation=operation  ).T
            res.sort_values(2, ascending=False, inplace=True)
            best_col = res.index[0]
            r = res.to_numpy()
            r = r[0]
            # print(res)
            # r = r.to_numpy()
            # thiw will generate an error towards best_split withoput discretization
            # print(r)
            return best_col,r[1],r[2],r[3] # 2 is impurity, 0 is node, 1 is distance value

    def get_label_balance_score(self, labels_df):
        label_count = labels_df.sum(axis='rows')
        mx = label_count.max()
        label_count = label_count/mx
        avg = label_count.mean()
        return avg
        
    # , tree , tree_id , instances, labels  ---> previously
    def generate_tree(self):
        # check if root node is None
        # the on should 
        tree = DecisionTree()
        tree_id=0
        instances = self.instances
        labels = self.labels
        # sample_size = int(self.instances.shape[0]*0.9)
        #selected_instances_index = self.instances.index
        # the baggind amount should also be a hyperparam!!!
        print("------------------")
        selected_instances_index = self.get_supervised_relative_random_sampling(labels, bagging_amount_pct = 0.84 )
        root_node = DecisionTreeNodeV2(None, selected_instances_index, labels , dataset=instances, hyper_params_dict=self.node_hyper_params )
        tree.add_root_node(root_node)
        # maybe we could generate parallel jobs on the children as well, no by tree, but by node
        self.generate_children(root_node ,0, tree_id = tree_id ,full_instances=instances, full_labels=labels )    

        return tree
    
    def generate_children(self, node,recursion, tree_id = 0 ,full_instances=None, full_labels=None ):
        # proxy
        return self.generate_childrenV1_1(node,recursion, tree_id, full_instances=full_instances, full_labels=full_labels  )
        
    def generate_childrenV1_1(self, node,recursion, tree_id = 0 , full_instances=None, full_labels=None):
        
        if(node.is_leaf):
            return 
        
        indices = node.get_instance_index()  # get the indices to work
        # indices = pd.Index([2,3,4,5,9,10,11,12,15,17])
        # get labels combinations
        labels = full_labels.loc[indices,:] # was self.labels
        instances = full_instances.loc[indices,:] # was self.instances
        
        # select N features... prepare for selecting M groups of N features 
        n_features = self.node_hyper_params["N_attr"]
        m_groups = self.node_hyper_params["M_groups"]
        
        unique_values_count = instances.nunique().to_numpy() # different values on each column 
        selectable_columns = list( (np.argwhere( unique_values_count > 1 )).ravel() )
        features_count = len(selectable_columns) # selectable columns has the column index, not the name
        
        label_index = labels[ labels[labels.columns[0]] != -1].index 
        supervised_examples = pairwise_distances( labels.loc[ label_index ].to_numpy() , metric='cosine' )
        df_labels_distances = pd.DataFrame(data=supervised_examples, columns=label_index)
        df_labels_distances.set_index(label_index, inplace=True)
        
        #print(df_labels_distances)
        """
        When classying use:
            cols index to get the cols of the row to be classified
            left cluster centroid 
            right cluster centroid 
            
            the closer it is by  gaussian/RBF kernel with euclidean distance to A or B, will determine if it goes right or left. 
            this could change to identify a certain spectrum of the cluster!! 
        """
        best_partition_coeff = np.PINF
        
        left_final_cluster_index = None
        right_final_cluster_index = None
        m = 0
        
        max_iterations = 8
        iteration = 0
        
        min_supervised_per_leaf = 1 # new nodes have to have more that this amount 
        left_supervised_objects = min_supervised_per_leaf
        right_supervised_objects = min_supervised_per_leaf
        print(f"Doing :{tree_id} , {node.id} {instances.shape}")
        columns_to_select = None
        key_node = -1
        decision_value = -1
        while m < m_groups and iteration < max_iterations:
            m+=1
            iteration += 1
            cols_index = self.get_random_columns(n_features, selectable_columns, linear_selection=True)
            ls = cols_index[0]
            x = instances.to_numpy()
            selected_cols = [instances.columns[i] for i in ls]
            
            do_spectral_clustering = True
            tree_div_calculated = False
            best_split_column = ""
            best_split_val = 0
            best_purity = 0
            best_simmetry = 0
            best_split_node = ""
            d_df =  None
            
            if( do_spectral_clustering ):
                
                """
                A = euclidean_distances( instances[ selected_cols ].to_numpy()  ,squared=True)
                A = A*-1/(2*self.node_hyper_params["gamma"]**2)
                np.exp(A,A)
                """
                 # spectral needs the distances to go the other way!!! it might be the same as we only need two clusters
                 
                #  affinity='precomputed', go for default with rbf
                print("spectral model")
                spectral_model = SpectralClustering(n_clusters = 2, eigen_solver='amg' , assign_labels='cluster_qr', gamma = self.node_hyper_params["gamma"] , n_jobs=-1)
                
                # kernelized_distances = rbf_kernel(X=x[ : ,  list(ls) ] , gamma = self.node_hyper_params["gamma"])
                # labels_rbf = spectral_model.fit_predict(instances[ selected_cols ])
                A = None
                labels_rbf = None
                with warnings.catch_warnings():
                    try:
                        labels_rbf = spectral_model.fit_predict( instances[ selected_cols ].to_numpy() )
                        print("out spectral model")
                        A = spectral_model.affinity_matrix_
                        if( A.shape[0] == 0):
                            m -= 1
                            continue
                    except Warning:
                        print("Caught warning")
                        r1 = int(random.random()*1000)
                        np.savetxt(f"affinity_matrix_{r1}.csv", A, delimiter=",")
                        m -= 1
                        continue
                # print(spectral_model.affinity_matrix_)
            else:
                # kmeans get the original columns, from them we will take distances
                #kmeans_model = KMeans(n_clusters = 2)
                #kmeans_model.fit( instances[selected_cols] )
                # covariance_type='full' si,tied no, diag si, spherical mmm....
                
                # gaussian mixture can do the distance stuff
                
                # gmodel = GaussianMixture(n_components=2, covariance_type='full')
                
                # when applying this model, it will only count the attribute space, no searching on the best split point
                
                gmodel  = UD3_5Clustering() #.fit_predict(train_set[instance_columns])
                
                
                """
                # ucomment to try with distances instead of original data
                # ----------------------------------------
                A = euclidean_distances( instances[ selected_cols ].to_numpy()  ,squared=True)
                A = A*-1/(2*self.node_hyper_params["gamma"]**2)
                np.exp(A,A)
                do_spectral_clustering = True
                d_df =  pd.DataFrame(data=A, columns=instances.index, index=instances.index)
                labels_rbf = gmodel.fit_predict(  d_df )
                tree_div_calculated = True    
                # is not purity but a measure of how compact is this 
                best_split_column,best_split_val,best_purity = gmodel.get_tree_params()
                best_split_node = best_split_column
                best_simmetry = 1 # do not count for this
                # --------------------------------------------------
                """
                
                # ucomment preivous to try with distances instead of original data
                # ----------------------------------------
                labels_rbf = gmodel.fit_predict(  instances[ selected_cols ] )
                tree_div_calculated = True    
                # is not purity but a measure of how compact is this 
                best_split_column,best_split_val,best_purity = gmodel.get_tree_params()
                best_simmetry = 1 # do not count for this
                # --------------------------------------------------
                
                
                
                # when doing directly the attributes, remember to tree_div_calculated = True and  do_spectral_clustering = False
                #labels_rbf = gmodel.fit_predict(instances[ selected_cols ])
                
                #labels_rbf = kmeans_model.labels_
                
        
            clustered_df = pd.DataFrame(data = labels_rbf)
            clustered_df.set_index(instances.index, inplace=True)
            
            
            left_cluster = clustered_df[clustered_df[0] == 0].index 
            right_cluster = clustered_df[clustered_df[0] == 1].index
            #print(left_cluster)
            #print("---")
            #print(right_cluster)
            
            left_labels_index = set(left_cluster.array).intersection( set(label_index.array) ) 
            right_labels_index = set(right_cluster.array).intersection( set(label_index.array) ) 
            
            
            """
            example = instances.copy()
            print(example)
            example["category"] = 0
            example.loc[ right_cluster, "category" ] = 1
            example = example.reset_index()
            xvars = example.columns[1:20] #  just a subset
            print(example)
            yvars=['index']
            DecisionTreeNodeV2.function_to_draw(example,xvars, yvars,f'output_nonleaf_division_node_{node.id}_{m}_.png')
            
            
    
            example = pd.DataFrame(data=A, index=instances.index, columns=instances.index)
            example["category"] = 0
            example.loc[ right_cluster , "category" ] = 1
            example = example.reset_index()
            xvars = example.columns[1:50] #  just a subset
            yvars=['index']
            example = example.loc[1:50] # to see only the same subset as cols
            DecisionTreeNodeV2.function_to_draw(example,xvars, yvars,f'output_NON_leaf_node_{node.id}_{m}_.png')
            
            example = pd.DataFrame(data=A, index=instances.index, columns=instances.index)
            example["category"] = labels[labels.columns[0]]
            #example.loc[ right_cluster , "category" ] = 1
            example = example.reset_index()
            xvars = example.columns[1:50] #  just a subset
            yvars=['index']
            example = example.loc[1:50] # to see only the same subset as cols
            DecisionTreeNodeV2.function_to_draw(example,xvars, yvars,f'output_NON_leaf_node_{node.id}_{m}_alt.png')
            """
            # print( df_labels_distances )
            if( len(left_labels_index) == 0 or len(right_labels_index) == 0 ):
                #print(f"{node.id} ERROR; this is not a partition as there are no labeled samples on one side {len(left_labels_index)} { len(right_labels_index)} {instances.shape}")
                m -= 1
                continue 
            if( not do_spectral_clustering):
                
                d_df =  instances[selected_cols]
                if not tree_div_calculated:
                    best_split_column, best_split_val, best_purity, best_simmetry = self.get_best_split( d_df, clustered_df, is_squared = False) # returns the node (index of the node) and the value on to we can split
                
                supervised_index = set(label_index.array) 
                best_col = d_df[best_split_column]
                
                left_supervised_instances = set(best_col[ best_col <= best_split_val ].index.array).intersection(supervised_index)
                right_supervised_instances = set(best_col[ best_col > best_split_val ].index.array).intersection(supervised_index)    
                supervised_instances_balance = 1-abs(len(left_supervised_instances)-len(right_supervised_instances))/(len(left_supervised_instances)+len(right_supervised_instances))
                # c_score = supervised_instances_balance
                
                # this will get the average imbalance ratio between all labels. 
                label_balance_score_left = self.get_label_balance_score(labels.loc[left_supervised_instances])
                label_balance_score_right = self.get_label_balance_score(labels.loc[right_supervised_instances])
                label_balance_score  = (label_balance_score_left +  label_balance_score_right)*0.5 # bigger balance , bigger numbers, 
                label_balance_score = 1 - label_balance_score # but we are optimizing towards 0!
                c_score = label_balance_score
                    
                # we could test here with a precomputed distance matrix, not necessarily the original instance feature space, but the kernelized one    
               
                
                # would be ideal to have a measure that implied 
                s_score = self.get_cluster_silhouette_score( instances[selected_cols] , clustered_df )
                
                
                left_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[left_labels_index , left_labels_index ]  )
                right_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[right_labels_index , right_labels_index ]  )
                distance_avg = (left_cluster_label_distance+right_cluster_label_distance)/2.0
            
                actual_coeff = (1-best_purity)*self.node_hyper_params["pD"] +  distance_avg*self.node_hyper_params["pA"] + s_score*self.node_hyper_params["pB"] +  c_score*self.node_hyper_params["pC"] 
                
                if( len(left_supervised_instances)  <= 3 or len(right_supervised_instances) <= 3 ): # 4 and up
                    #print("try again")
                    m -= 1
                    continue
                    
                if( actual_coeff < best_partition_coeff):
                    best_partition_coeff = actual_coeff
                    left_final_cluster_index = best_col[ best_col <= best_split_val ].index
                    right_final_cluster_index = best_col[ best_col > best_split_val ].index
                    key_node  = best_split_column
                    decision_value = best_split_val
                    columns_to_select  = selected_cols
                
            elif( self.do_ranking_split):
                
                d_df =  pd.DataFrame(data=A, index=instances.index, columns=instances.index)
                
                # best_soplit node is the index of the column that was selected as thge best 
                # best split value is the distance value to divide 
                # best purity is the purity index of the left and right branch combined 
                # best simmetry is the balance betweeen left and right branch amounts
                
                if not tree_div_calculated:
                    # Ttried already but  did not work as expected >... we should try to measure the opposite, the impurity!!! to try to tackle the imbalance ratio. Change the actual_coeff equuattion and the ordering of the results from get_best_split
                    # what we have not tested is actually doiing the division between the two clusters!!! not generating a split point. As we already have one.
                    
                    # this one works better with birds
                    #best_split_node, best_split_val, best_purity, best_simmetry = self.get_best_split( d_df, clustered_df) # returns the node (index of the node) and the value on to we can split
                    # this one works better with emotions

                    # DONT change so the d_df is the original dataset, as the distances are already calculated
                    best_split_node, best_split_val, best_purity, best_simmetry = self.get_cluster_distance_based_best_split( d_df, clustered_df, original_label_df = clustered_df,  operation=self.node_hyper_params["purity_op"] ) # min is because we want to minimize the purity, keep complicated splits    # returns the node (index of the node) and the value on to we can split    
                
                best_col = d_df[best_split_node]
                
                ndf = pd.DataFrame(index=instances.index)
                ndf[f"distance_{best_split_node}"] = best_col               
                ndf["cluster_label"] = clustered_df[0]
                for pc in range(0, len(labels.columns) ) :
                    ndf[f"label_{pc}"] = labels[ labels.columns[pc] ]
                
                DecisionTreeNodeV2.function_to_draw(ndf, name=f"chart_{tree_id}_{node.id}_{recursion}_{m}_{best_purity}_val_{best_split_val}.png")
                
                supervised_index = set(label_index.array) 
                #print(best_col)
                
                
                left_supervised_instances = set(best_col[ best_col <= best_split_val ].index.array).intersection(supervised_index)
                right_supervised_instances = set(best_col[ best_col > best_split_val ].index.array).intersection(supervised_index)
                # this lets the more balanced nodes with the same amount of supervised data be kept- Balances the supervised instances
                supervised_instances_balance = 1-abs(len(left_supervised_instances)-len(right_supervised_instances))/(len(left_supervised_instances)+len(right_supervised_instances))
                
                # c_score = supervised_instances_balance
                
                # this will get the average imbalance ratio between all labels. 
                # think about balancing them weighting by the amount of instance. Penalty the few supervised nodes. 
                label_balance_score_left = self.get_label_balance_score(labels.loc[left_supervised_instances])
                label_balance_score_right = self.get_label_balance_score(labels.loc[right_supervised_instances])
                label_balance_score  = (label_balance_score_left +  label_balance_score_right)*0.5 # bigger balance , bigger numbers, 
                label_balance_score = 1 - label_balance_score # but we are optimizing towards 0!
                c_score = label_balance_score
                
                # is it necessary ? as we are already modifying this on the tree_div_calculated... Seems an over estimation, or calculate with the new clusters, the ones after the division. change the labels rbf
                # s_score = self.get_cluster_silhouette_score( instances[selected_cols].to_numpy() , labels_rbf )
                # should be the same as the other one. 
                clustered_df.loc[ best_col[ best_col <= best_split_val ].index ] = 0
                clustered_df.loc[ best_col[ best_col > best_split_val ].index ] = 1

                if( 
                    len(best_col[ best_col <= best_split_val ].index) == 0 or 
                    len(best_col[ best_col > best_split_val ].index) == 0
                    ):
                    m -= 1
                    continue
                
                s_score = self.get_cluster_silhouette_score( d_df , clustered_df  , precomputed_distance= True)
                

                # homogeneity on label space with cosine similarity. As opposed to variance. We might test different measurement
                left_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[left_labels_index , left_labels_index ]  )
                right_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[right_labels_index , right_labels_index ]  )
                distance_avg = (left_cluster_label_distance+right_cluster_label_distance)/2.0
                    
                
                #qA pA = 0.50 # coefficient of label concordance, distance on supervised  
                #qB pB = 0.30 # coefficient of simetric amount on both leaves,  simetry index 
                #qC pC = 0.2 # coefficient of supervised samples on both leaves, supervised samples on bot leaves 
                #pD= 0.1 #purity importance
                
                #actual_coeff = best_purity*0.3 +  distance_avg*.3 + s_score*.1 +  supervised_instances_balance*.3, lets get rid of simmetry in favour of the cluster shape  -> silhouette score 
                # actual_coeff = (1-best_purity)*self.node_hyper_params["pD"] +  distance_avg*self.node_hyper_params["pA"] + s_score*self.node_hyper_params["pB"] +  c_score*self.node_hyper_params["pC"] 
                # test minimizing purity 
                actual_coeff = (best_purity)*self.node_hyper_params["pD"] +  distance_avg*self.node_hyper_params["pA"] + s_score*self.node_hyper_params["pB"] +  c_score*self.node_hyper_params["pC"] 
 
 
                # print(f"Best split [id:{node.id},m:{m},i:{iteration}]:>{actual_coeff} {best_purity} {distance_avg}|{score} {best_simmetry} ")
                # 
                # print(f"Best split [id:{node.id},m:{m},i:{iteration}]:>{actual_coeff} {best_split_node} {best_split_val} {best_purity} {best_simmetry} {len(left_supervised_instances)} {len(right_supervised_instances)}")
                # amount of supervised instances should be a parameter according to the dataset , or even a percentage
                if( len(left_supervised_instances)  <= 3 or len(right_supervised_instances) <= 3 ): # 4 and up
                    #print("try again")
                    # this will be a left node as the best partition found, leaves a leaf with only unsupervised nodes
                    # let's try again 
                    m -= 1
                    continue
                print( actual_coeff ,"<", best_partition_coeff)
                if( actual_coeff < best_partition_coeff):
                    best_partition_coeff = actual_coeff
                    left_final_cluster_index = best_col[ best_col <= best_split_val ].index
                    right_final_cluster_index = best_col[ best_col > best_split_val ].index
                    key_node  = best_split_node
                    decision_value = best_split_val
                    columns_to_select  = selected_cols
                
            else:
                left_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[left_labels_index , left_labels_index ]  )
                right_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[right_labels_index , right_labels_index ]  )
                
                left_size = len(left_cluster)
                right_size = len(right_cluster)
                total_size = left_size/(left_size+right_size)
                size_coeff = abs(0.5 - total_size) # if close to 0.5 the number would be lower 
                
                sup_index_left = 1 - len(left_labels_index)/left_size # if it is 0, then this partition will not be enough to do the transductive part.  
                sup_index_right = 1 - len(right_labels_index)/right_size # if it is 0, then this partition will not be enough to do the transductive part.  
                sup_index_avg = (sup_index_left+sup_index_right)/2
                # the inverse is to maintain everything on minimization
                
                #qA pA = 0.50 # coefficient of label concordance, distance on supervised  
                #qB pB = 0.30 # coefficient of simetric amount on both leaves,  simetry index 
                #qC pC = 0.2 # coefficient of supervised samples on both leaves, supervised samples on bot leaves 
                #pD= 0.1 #purity importance
                
                actual_coeff = ((left_cluster_label_distance+right_cluster_label_distance)/2*self.node_hyper_params["qA"] + size_coeff*self.node_hyper_params["qB"] +  sup_index_avg*self.node_hyper_params["qC"])*1.0 
                #  print(f"Distances :{node.id} {left_cluster_label_distance}  {right_cluster_label_distance}  {size_coeff:>0.4} {actual_coeff:>0.4} supervised l{sup_index_left} r{sup_index_right}")
                
                if(sup_index_left  >= 0.99 or sup_index_right >= 0.99):
                    # this will be a left node as the best partition found, leaves a leaf with only unsupervised nodes
                    # let's try again 
                    m -= 1
                    continue
                
                if( actual_coeff < best_partition_coeff):
                    best_partition_coeff = actual_coeff
                    left_final_cluster_index = left_cluster
                    left_supervised_objects = len(left_labels_index)
                    right_final_cluster_index = right_cluster
                    right_supervised_objects =  len(right_labels_index)
                    columns_to_select  = selected_cols
        

        if( self.do_ranking_split ):
            if(iteration > max_iterations or decision_value == -1 ):
                # no way jose
                #print(f'{iteration} {left_final_cluster_index} {right_final_cluster_index} {left_supervised_objects} {right_supervised_objects}')
                node.set_leaf(True)
                return
            
        else:
            if(iteration > max_iterations or left_final_cluster_index is None or right_final_cluster_index is None or  left_supervised_objects <= min_supervised_per_leaf or right_supervised_objects <= min_supervised_per_leaf ):
                # no way jose
                #print(f'{iteration} {left_final_cluster_index} {right_final_cluster_index} {left_supervised_objects} {right_supervised_objects}')
                node.set_leaf(True)
                return
        
        if( not do_spectral_clustering ):
            # key node has column information although is named node
            node.set_decision_column_and_value(key_node,decision_value)
        elif( self.do_ranking_split):
            node.set_decision_value( decision_value  )
            #print(key_node)
            #print(columns_to_select)
            #print(decision_value)
            
            node.set_decision_instance_values( instances.loc[key_node, columns_to_select].to_numpy() )
            node.set_decision_columns(columns_to_select)
            
        else:
            left_centroid =  instances.loc[left_final_cluster_index, columns_to_select].mean(axis=0)
            right_centroid = instances.loc[right_final_cluster_index, columns_to_select].mean( axis=0)
            node.set_decision_columns(columns_to_select)
            node.set_left_centroid(left_centroid)
            node.set_right_centroid(right_centroid)
    
        left_child = DecisionTreeNodeV2(node, left_final_cluster_index,  full_labels, node.level+1, 
            not ( len(left_final_cluster_index) >= self.get_leaf_instance_quantity()) , dataset=full_instances, hyper_params_dict=self.node_hyper_params )
        # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
        node.set_left(left_child)
        #print(f"{node.id} L-> {left_child.id} {len(left_final_cluster_index)} // {left_supervised_objects} ")
        self.generate_children(left_child,recursion+1, tree_id = tree_id,full_instances=full_instances, full_labels=full_labels )

        right_child = DecisionTreeNodeV2(node, right_final_cluster_index,  full_labels, node.level+1, 
            not (len(right_final_cluster_index) >= self.get_leaf_instance_quantity()) , dataset=full_instances, hyper_params_dict=self.node_hyper_params )
        # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
        node.set_right(right_child)
        #print(f"{node.id} R-> {right_child.id} {len(right_final_cluster_index)} // {right_supervised_objects}")
        self.generate_children(right_child,recursion+1, tree_id = tree_id,full_instances=full_instances, full_labels=full_labels )

        
    def save_txt(self):
        tree_file = open('trees.txt' , 'w')
        for tree in self.trees:
            tree_file.write( tree.root.get_draw_repr() )
            tree_file.write("end_tree")
        tree_file.close()

    def report_final_data_distribution(self , tree_ones_distrib):
        # will receive for each node, the final class distribution. 
        self.average_class_distribution += tree_ones_distrib/self.trees_quantity/self.labels.shape[0]
    


    def get_params(self, deep=True):
        return {'leaf_relative_instance_quantity': self.leaf_relative_instance_quantity}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def calculate_labels_distribution(self):

        # is the same for every tree in the forest, each one will update according to its own tree
        distribution = (self.labels.replace(-1, 0)).sum()
        labeled_data_count = (self.labels.replace({0: 1, -1: 0})).sum()
        ones_distribution_total = distribution/labeled_data_count
        self.dataset_ones_distribution = ones_distribution_total
        # this won't change! all the trees need the same dataset_ones_distribution

        # print(self.dataset_ones_distribution)

    def fit(self, X, y):
        y = y.copy()
        #rint(self.unlabeledIndex)
        #print(self.compatibilityMatrix)
        #print(self.unlabeledIndex)
        self.dataset_assigned_label_count = None
        toSelect = None
        cantDo = (self.unlabeledIndex is None)
        

        if not (self.unlabeledIndex is None):
            toSelect = y.index.intersection(self.unlabeledIndex)
            print(f"Doing semi superv sim {X.shape} {self.leaf_relative_instance_quantity}  {self.base_file_root}")
            y.loc[toSelect, :] = -1
            self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
        else:
            print(f"Doing supervised sim {self.leaf_relative_instance_quantity} {self.base_file_root}")
            self.unlabeledIndex = pd.Index([0])
            toSelect = y.index.intersection(self.unlabeledIndex)
            y.loc[toSelect, :] = -1
            self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
            # just mimic with one the semisupervised setup
            # add unlabeled columns
            
            """
            for i in self.unlabeledIndex :
                self.compatibilityMatrix[i] = -1 #per column
    
            # add rows
            for i in self.unlabeledIndex :
                self.compatibilityMatrix.loc[i] = -1 # per row
            """        

        # if(noCompat):
        #    return self
        self.instances = X
        self.labels = y
        
        
        self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
        self.average_class_distribution = np.zeros( shape=[1 ,self.labels.shape[1] ])
        
        #self.compatibilityMatrix = compatibilityMatrix
        self.tree_log = open(self.base_file_root+'tree_log.txt', 'w')
        print(f"Doing tree generation {self.leaf_relative_instance_quantity} {self.base_file_root} {self.trees_quantity}")
        # previous to anything all the columns should be normalized, to be able to compare variances
        # testing with heuristic as division, so this should not be done.
        #self.instances = (self.instances-self.instances.min())/(self.instances.max()-self.instances.min())

        # for the amount of trees needed, generate N decision trees and train them all. Remember to select a subset of the columns
        self.trees = []
        threads = []
        self.calculate_labels_distribution()
        

        #print( ordered_attributes_indices )
        for i in range(0,self.trees_quantity):
            # this is parallelized
            # self.trees.append(DecisionTree())
            # self.generate_tree(self.trees[i], i)
            
            x = threading.Thread(target=self.generate_tree, args=(self.trees[i],i ) )
            # x = Process(target=self.generate_tree, args=(self.trees[i],i ) )
            x.start()

        for t in threads:
            t.join()
        threads.clear()        
        
        self.tree_log.close()
        print(f"Ended training {self.leaf_relative_instance_quantity} {self.base_file_root}")
        # get 1's and 0's distribution
        # self.calculate_labels_distribution()
        # if( self.complete_ss ):
        #self.fill_semisupervised() # changed this to do it inside generate tree method
        # updates internal ones distribution.
        #else:
        #    self.fill_ones_distribution()
        self.ready = True
        tree_index = 0


        # this should change to another explainable relation with the forest
        """
        for tree in self.trees :
            file = open(self.base_file_root+f'explanation_tree_{tree_index}.txt', 'w')
            tree.root.printRules(file, 0)
            file.close()
            tree_index += 1
        """
        return self

    def predict(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return pred

    def predict_with_proba(self, X, print_prediction_log=False ,y_true = None):
        # check_is_fitted(self)
        #predictions = np.zeros(shape=[self.labels.shape[1]])
        #probabilities = np.zeros(shape=[self.labels.shape[1]])
        predictions = []
        probabilities = []
        y_counter = 0
        for index, row in X.iterrows():
            print(f"I'm trying with this one {index}")
            tree_counter = 0
            instance_prediction = np.zeros(shape=[self.labels.shape[1]])
            instance_probability= np.zeros(shape=[self.labels.shape[1]])
            for tree in self.trees:
                # print(f"tree {tree_counter}")
                tree_counter += 1
                prd, prb = tree.root.predict_with_proba(row, original_labels=y_true[y_counter] if y_true is not None else None)
                #print(prd)
                instance_prediction +=  np.array( prd)/len(self.trees)
                instance_probability += np.array( prb)/len(self.trees)
            predictions.append(instance_prediction)
            probabilities.append(instance_probability)
            y_counter += 1
            
            # 0.4 could be moved... based on calibration,  this is joining all the trees
        
        return np.where( np.array(predictions)>=0.5,1,0 ) , np.array(probabilities)


    def predict_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return prob
