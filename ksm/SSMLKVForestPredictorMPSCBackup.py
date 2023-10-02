import itertools
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import uuid
from .DTNodeMPSC import DecisionTree, DecisionTreeNodeV2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
import pprint
import pandas as pd
import sys
from numpy.random import default_rng
import math
import threading
from multiprocessing import Process
import random
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel,check_pairwise_arrays
from .utils import pairwise_distances
from .SSClustering import SSClustering

from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
from  sklearn import svm

from joblib import Parallel, delayed
from multiprocessing import Manager
import multiprocessing as mp 
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
    gamma = 10, hyper_params_dict = None, do_ranking_split = False, p=[.5,0,0,.5], q=[1,0,0], use_complex_model=False, njobs=None):
        super().__init__()
        self.njobs = njobs
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
                'gamma' : gamma,
                'pA' : p[0],
                'pB' : p[1],
                'pC' : p[2],
                'pD' : p[3],
                'pE' : p[4],
                'pF' : p[5],
                'pG' : p[6],
                'do_ranking_split':do_ranking_split,
                'use_complex_model':use_complex_model,
            }
        else: 
            self.node_hyper_params = hyper_params_dict
            
            self.node_hyper_params['pA'] = hyper_params_dict['p'][0]
            self.node_hyper_params['pB'] = hyper_params_dict['p'][1]
            self.node_hyper_params['pC'] = hyper_params_dict['p'][2]
            self.node_hyper_params['pD'] = hyper_params_dict['p'][3]
            self.node_hyper_params['pE'] = hyper_params_dict['p'][4]
            self.node_hyper_params['pF'] = hyper_params_dict['p'][5]
            self.node_hyper_params['pG'] = hyper_params_dict['p'][6]


            leaf_relative_instance_quantity = hyper_params_dict["leaf_relative_instance_quantity"]
            trees_quantity = hyper_params_dict["trees_quantity"] 
            self.do_ranking_split = hyper_params_dict["do_ranking_split"] 
            
        self.node_hyper_params["output_quality"] = self.node_hyper_params.get("output_quality",False)
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
        # supervised_samples = label_set[ (label_set >= 0).any(axis=1) ]
        supervised_samples = label_set[ (label_set > 0).any(axis=1) ]
        unsupervised_samples = label_set[ (label_set == -1).any(axis=1) ]
        supervised_samples = supervised_samples.fillna(0)
        class_freq = supervised_samples.sum(axis = 0)/supervised_samples.shape[0]
        class_inv_freq = (1 - class_freq)*label_set.shape[0]    
        class_freq = class_freq*supervised_samples.shape[0]
        
        all_indices = set({})
        # changed to see if it really helped... and yes, it really helps
        for index,value in class_inv_freq.items() :
            original_c = class_freq.loc[index]
            indices = random.sample( list( supervised_samples[ supervised_samples[index] == 1 ].index.array  ) ,  int(min( int(value) ,  int(original_c)   )*bagging_amount_pct)  ) 
            all_indices = all_indices.union(set(indices))
        # print(int(len(supervised_samples.index)))
        
        # indices = random.sample( list( supervised_samples.index.array  ) , int(len(supervised_samples.index)*bagging_amount_pct))
        # all_indices = all_indices.union(set(indices))
        
        
        supervised_samples_1 = label_set[ (label_set >= 0).any(axis=1) ]
        #all_indices =set(supervised_samples_1.index.array)
        
        indices = random.sample( list( supervised_samples_1.index.array  ) , int(len(supervised_samples_1.index)*bagging_amount_pct))
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
    

    def get_random_columns(self, features_count, selectable_columns, linear_selection=False):
        
        random_columns = []
        
        if( features_count > 0):
            #random_columns = random.sample( selectable_columns ,  int(math.log( features_count,2 ))  )
            #n_to_select = int(math.log( features_count,2 ))*(random.random()*0.5+0.5)  if not linear_selection else features_count
            n_to_select = int(math.log( features_count,2 ))  if not linear_selection else features_count
            
            #print("selecting " , n_to_select )
            random_columns = random.sample( selectable_columns , k=int(math.ceil(n_to_select ))  )
            #print(random_columns)
        else:
            return [], True 
        
        return random_columns ,  False

    def get_average_distance(self, distance_matrix_df):
        
        if(type(distance_matrix_df) is pd.DataFrame):
            distance_matrix_df = distance_matrix_df.to_numpy()
        # 0 is the same, 1 is the farthest
        half_matrix = np.tril(distance_matrix_df)
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

    def get_label_balance_score(self, labels_df):
        return 1 - self.get_label_example_balance_score(labels_df) # minus 1 'cause this will be negated on main thread
        label_count = labels_df.sum(axis='rows')
        mx = label_count.max()
        label_count = label_count/mx
        avg = label_count.mean()
        return avg
    
    def get_label_example_balance_score(self, labels_df):
        """
        This function, instead on analyzing the imbalance between the top label and tail label
        will get a higher score when every positive and negative case of each label is represented. 
        It should take a higher score if there is more than one sample. It is better to have at least one example
        than to have many of just one type.
        The objective is to "distribute" the few labeled ones in all the tree clusters
        """
        avg = 0

       
        for name,col in labels_df.items():
            
            value_counts = col.value_counts()
            
            if( (0 in value_counts.index.array) and (1 in value_counts.index.array) ):
                neg = value_counts.loc[0]
                pos = value_counts.loc[1]
                min_obs = min(neg,pos)
                max_obs = max(neg,pos) # worst case, both are 0
                p = (min_obs+max_obs)/(col.count()) # worst case they are equal, we cannot have 0 on the count
                g = min_obs/max_obs
                avg += p*g # is the labeling proportion times labeling degree between them. We are looking for the bigger balanced one
            else:
                avg += 0
                continue

        res = avg/len(labels_df.columns.array)
        
        return res
    
    def get_features_distance(self, X):
        # dist = pairwise_distances(X.to_numpy(),metric='euclidean')
        dist = pairwise_distances(X.to_numpy(),metric=self.node_hyper_params['distance_function'])
        mn = dist.min()
        mx = dist.max()
        div = (mx - mn)+0.00001
        dist = (dist-mn)/div
        return self.get_average_distance(dist)


    def get_option(self):
        pass

    # , tree , tree_id , instances, labels  ---> previously
    def generate_tree(self,ns, tid):
        #ns = tup[0]
        #tid = tup[1]
        
        # check if root node is None
        # the on should 
        tree = DecisionTree()

        
        tree_id=tid
        
        #instances = shared_memory.SharedMemory(name="shared_instances")
        #labels = shared_memory.SharedMemory(name="shared_labels")
        instances = ns.instances
        labels = ns.labels
        # tq_bar = ns.pbar
        # print(ns.pbar)
        # sample_size = int(self.instances.shape[0]*0.9)
        #selected_instances_index = self.instances.index
        # the baggind amount should also be a hyperparam!!!
        
        selected_instances_index = self.get_supervised_relative_random_sampling(labels, bagging_amount_pct = self.node_hyper_params["bagging_pct"] )
        root_node = DecisionTreeNodeV2(None, selected_instances_index, labels , dataset=instances, hyper_params_dict=self.node_hyper_params, tree_id=tree_id)
        tree.add_root_node(root_node)
        # maybe we could generate parallel jobs on the children as well, no by tree, but by node
        self.generate_children(root_node ,0, tree_id = tree_id ,full_instances=instances, full_labels=labels )    

        # tq_bar.update(1)
        return tree #f"hello {i}"
    
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
        #print(str(tree_id) + " __ " + str(node.id) + "  sup index " + str(label_index) )
        supervised_examples = pairwise_distances( labels.loc[ label_index ].to_numpy() , metric='cosine' )
        df_labels_distances = pd.DataFrame(data=supervised_examples, columns=label_index)
        df_labels_distances.set_index(label_index, inplace=True)

        best_partition_coeff = -1
        
        left_final_cluster_index = None
        right_final_cluster_index = None
        m = 0
        
        max_iterations = self.node_hyper_params["m_iterations"]
        iteration = 0
        
        min_supervised_per_leaf = 2 # new nodes have to have more that this amount 
        left_supervised_objects = min_supervised_per_leaf
        right_supervised_objects = min_supervised_per_leaf
        
        # print(f"Doing :{tree_id} , {node.id} {instances.shape}")
        
        columns_to_select = None
        key_node = -1
        decision_value = -1

        cluster_centers = None

        if(node.level >= self.node_hyper_params["depth_limit"] ):
            node.set_leaf(True)
            return

        columns_selected_history = None

        while m < m_groups and iteration < max_iterations:
            m+=1
            iteration += 1
            cols_index = self.get_random_columns(n_features, selectable_columns, linear_selection=False)
            

            # print("cols index " , columns_selected_history)
            ls = cols_index[0]
            x = instances.to_numpy()
            selected_cols = [instances.columns[i] for i in ls]
            
            #res = get_option(selected_cols)
            
            do_spectral_clustering = True
            tree_div_calculated = False
            best_split_column = ""
            best_split_val = 0
            best_purity = 0
            best_simmetry = 0
            best_split_node = ""
            d_df =  None
            
            correlation_data = np.abs(instances[ selected_cols ].corr().to_numpy()).sum(axis=0)
            correlation_data = np.mean(correlation_data/correlation_data.shape[0])
            #print("Correlation data " , correlation_data)

            do_gower= True if self.node_hyper_params["distance_function"]=='gower' else False
            sscluster = SSClustering(2, just_base_cluster=True, use_gower=do_gower)
            sscluster.fit( instances[ selected_cols ] , labels , far_distance_on_attribute_space = True )
            labels_rbf = sscluster.predict(instances[ selected_cols ])
                
        
            clustered_df = pd.DataFrame(data = labels_rbf)
            clustered_df.set_index(instances.index, inplace=True)

            
           
           
            left_cluster = clustered_df[clustered_df[0] == 0].index 
            right_cluster = clustered_df[clustered_df[0] == 1].index
            #print(f"-{tree_id} ",left_cluster)
            #+print("---")
            #print(f"+{tree_id} ",right_cluster)
            
            left_labels_index = list(set(left_cluster.array).intersection( set(label_index.array) ) )
            right_labels_index = list(set(right_cluster.array).intersection( set(label_index.array) )) 
            
            #print(f"-**{tree_id} ",left_labels_index)
            #print(f"-**{tree_id} ",right_labels_index)

            # print( df_labels_distances )
            if( len(left_labels_index) == 0 or len(right_labels_index) == 0 ):
                # print(f"ZERO! Label index err {left_labels_index} {right_labels_index}")
                # print(f"{node.id} ERROR; this is not a partition as there are no labeled samples on one side {len(left_labels_index)} { len(right_labels_index)} {instances.shape}")
                m -= 1
                continue 
            
            label_balance_score_left = self.get_label_balance_score(labels)
            label_balance_score_right = self.get_label_balance_score(labels)
            
            label_balance_score  = (label_balance_score_left +  label_balance_score_right)*0.5 # bigger balance , bigger numbers, 
            c_score = label_balance_score
            
            left_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[left_labels_index , left_labels_index ]  )
            right_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[right_labels_index , right_labels_index ]  )
            distance_avg = (left_cluster_label_distance+right_cluster_label_distance)/2.0

            if (self.node_hyper_params["pE"] <= 0):
                s_score = 0 # the same for everyone, but do not calculate
            else:
                s_score = self.get_cluster_silhouette_score( instances[selected_cols] , clustered_df )
            
            
            attr_cluster_distance_A = self.get_features_distance(instances.loc[left_cluster,selected_cols])
            attr_cluster_distance_B = self.get_features_distance(instances.loc[right_cluster,selected_cols])
            attr_cluster_distance =(attr_cluster_distance_A + attr_cluster_distance_B)*0.5 # lets try to find the clusters with the most space between them.
            
            if( len(left_labels_index)  <= min_supervised_per_leaf or 
               len(right_labels_index) <= min_supervised_per_leaf 
                or len(left_cluster) < self.get_leaf_instance_quantity()
                or len(right_cluster) < self.get_leaf_instance_quantity()
               ): # 2 and up
                # print(f"{tree_id} Label index err {iteration} {left_labels_index} {right_labels_index}")
                m -= 1
                continue



            supervised_imbalance = min(len(left_labels_index), len(right_labels_index))/max(len(left_labels_index), len(right_labels_index))
            left_sup_unsup_ratio = len(left_labels_index)/len(left_cluster) 
            right_sup_unsup_ratio = len(right_labels_index)/len(right_cluster)
            sup_unsup_ratio_balance = min( left_sup_unsup_ratio , right_sup_unsup_ratio) / max( left_sup_unsup_ratio , right_sup_unsup_ratio)
            best_purity = 0
            
            # actual_coeff = (1-attr_cluster_distance)*self.node_hyper_params["pD"] +  (1-distance_avg)*self.node_hyper_params["pA"] + (1-s_score)*self.node_hyper_params["pB"] +  (c_score)*self.node_hyper_params["pC"] # c_score is already 
            actual_coeff = (1-distance_avg)*self.node_hyper_params["pA"] + supervised_imbalance*self.node_hyper_params["pB"] +  (sup_unsup_ratio_balance)*self.node_hyper_params["pC"] + (c_score)*self.node_hyper_params["pD"] +(1-s_score)*self.node_hyper_params["pE"]  + correlation_data*self.node_hyper_params["pF"] + (1-attr_cluster_distance)*self.node_hyper_params["pG"]
 
            new_best = False
            if( actual_coeff > best_partition_coeff):
                    # print(f"{tree_id} {actual_coeff}     {best_partition_coeff}      {supervised_imbalance}_ {sup_unsup_ratio_balance}       {attr_cluster_distance} {distance_avg} {1-s_score} {c_score} ")
                    
                    # we need to make sure that the partition will be fit enough after the iteration
                    left_supervised_objects = len(left_labels_index)
                    right_supervised_objects = len(right_labels_index)
                    best_partition_coeff = actual_coeff
                    left_final_cluster_index = left_cluster
                    right_final_cluster_index = right_cluster
                    columns_to_select  = selected_cols
                    cluster_centers = sscluster.clusters_index
                    new_best = True
            
            if( columns_selected_history is None):
                columns_selected_history = set()
            columns_selected_history.add( (tree_id, new_best,iteration,actual_coeff, correlation_data ,frozenset( cols_index[0]) ) )
            
        # print(f"Done {(left_final_cluster_index)} {(right_final_cluster_index)} {columns_to_select}")

        if(iteration > max_iterations or 
           left_final_cluster_index is None 
           or right_final_cluster_index is None 
           or  left_supervised_objects <= min_supervised_per_leaf 
           or right_supervised_objects <= min_supervised_per_leaf 
           ):
            # no way jose
            # print(f'Ended {iteration} {(left_final_cluster_index)} {(right_final_cluster_index)}  s.o. {left_supervised_objects} {right_supervised_objects}')
            node.set_leaf(True)
            return
    
        node.set_decision_columns(columns_to_select)

        left_child = DecisionTreeNodeV2(node, left_final_cluster_index,  full_labels, node.level+1, 
            False , dataset=full_instances, hyper_params_dict=self.node_hyper_params )
        # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
        left_child.tree_id = tree_id
        node.set_left(left_child)
        #print(f"{node.id} L-> {left_child.id} {len(left_final_cluster_index)} ")
        self.generate_children(left_child,recursion+1, tree_id = tree_id,full_instances=full_instances, full_labels=full_labels)

        if(self.node_hyper_params["output_tree_sets"]):
            merged = pd.concat( [instances.loc[left_final_cluster_index,columns_to_select], labels.loc[left_final_cluster_index, :] ] , axis=1)
            merged["cluster"] = 0
            merged["center"] = 0
            if( cluster_centers is not None):
                merged.loc[cluster_centers[0] , "center"] = 1
            merged.to_csv(f"emotions_nodes_story/{tree_id}_{node.level+1}_left_{uuid.uuid4()}.csv") 

        right_child = DecisionTreeNodeV2(node, right_final_cluster_index,  full_labels, node.level+1, 
            False , dataset=full_instances, hyper_params_dict=self.node_hyper_params )
        # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
        right_child.tree_id = tree_id
        node.set_right(right_child)
        # print(f"{node.id} R-> {right_child.id} {len(right_final_cluster_index)} // ")
        self.generate_children(right_child,recursion+1, tree_id = tree_id,full_instances=full_instances, full_labels=full_labels)
        
        #print("-------------")
        #for csh in columns_selected_history:
        #    print("++\n" ,  csh)
        #print("*****************" , len(columns_selected_history) )
        #print(columns_selected_history)
        
        if( self.node_hyper_params["output_tree_sets"] ):
            merged = pd.concat( [instances.loc[right_final_cluster_index,columns_to_select], labels.loc[right_final_cluster_index, :] ] , axis=1)
            merged["cluster"] = 1
            merged["center"] = 0
            if(cluster_centers is not None):
                merged.loc[cluster_centers[1] , "center"] = 1
            merged.to_csv(f"emotions_nodes_story/{tree_id}_{node.level+1}_right_{uuid.uuid4()}.csv") 


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
        

        # self.tree_log = open(self.base_file_root+'tree_log.txt', 'w') # this is not going well for multiprocess
        print(f"Doing tree generation {self.leaf_relative_instance_quantity} {self.base_file_root} {self.trees_quantity}")
        # previous to anything all the columns should be normalized, to be able to compare variances
        # testing with heuristic as division, so this should not be done.
        #self.instances = (self.instances-self.instances.min())/(self.instances.max()-self.instances.min())

        # for the amount of trees needed, generate N decision trees and train them all. Remember to select a subset of the columns
        self.trees = []
        self.calculate_labels_distribution()

        
        mgr = Manager()
        ns = mgr.Namespace()
        ns.instances = X
        ns.labels = y
        #ns.pbar = pbar
        pool = mp.Pool(processes = self.njobs) # this can grow with processors. 

        jobs = []
        for i in range(0, self.trees_quantity):
            jobs.append(
                pool.apply_async( self.generate_tree, args=(ns,i )) 
            )
        # result = pool.map_async( self.generate_tree, [(ns,i) for i in range(0,self.trees_quantity)] ) # it takes 22 secs to end them all
        pool.close()
        
        # pool.join()
        pbar = tqdm(total = self.trees_quantity)
        for job in jobs:
            self.trees.append(job.get())
            pbar.update(1)

        #for tree in result.get():
        #    # print("returning " , tree)
        #    self.trees.append(tree)

        pbar.close()
        
        print(f"Ended training")

        self.ready = True
        tree_index = 0


        # this should change to another explainable relation with the forest
        # the fact of the labels_distribution should be passed to each tree
        labels_distribution_array = []
        for tree in self.trees :
            labels_distribution = self.labels.loc[tree.root.instance_index.array]
            # print(labels_distribution[ labels_distribution[labels_distribution.columns[0]]>-1 ].index.array)
            labels_distribution = labels_distribution[ labels_distribution[labels_distribution.columns[0]]>-1 ].mean(axis=0)
            tree.root.set_global_labels_distribution(labels_distribution)
            labels_distribution_array.append(labels_distribution.to_numpy())
            """
            file = open(self.base_file_root+f'explanation_tree_{tree_index}.txt', 'w')
            tree.root.printRules(file, 0)
            file.close()
            tree_index += 1
            """
        # np.savetxt("labels_distribution.csv", np.array(labels_distribution_array) ,delimiter=",")
        if( self.node_hyper_params["output_quality"] ):
            self.quality = self.get_model_quality()
            print(f"Model trees mean distance (the larger, the better): {self.quality} ")
        return self
 
    def get_model_quality(self):
        t_N = int(len( list(self.instances.index.array))*0.15)
        testing_instances_index = random.sample( list(self.instances.index.array) , t_N  ) 
        # print(testing_instances_index)
        test_instances  = (self.instances.loc[testing_instances_index])
        # print(test_instances)
        distance_matrix = np.zeros(shape=[len(self.trees),len(self.trees)])

        pb = tqdm(total = test_instances.shape[0] )

        for index,instance in test_instances.iterrows():
            result_matrix = np.zeros(shape=[len(self.trees),self.labels.shape[1] ])
            i = 0
            for tree in self.trees :
                prd, prb = tree.root.predict_with_proba(instance, None)
                result_matrix[i] = prb
                i+=1
            #print(result_matrix)
            instance_distances = pairwise_distances(result_matrix, metric='euclidean')
            #print(instance_distances)
            distance_matrix += instance_distances/math.sqrt(self.labels.shape[1]) # normalize distances
            pb.update(1)

        pb.close()
        
        distance_matrix = distance_matrix/t_N
        # np.savetxt("distance_matrix.csv",distance_matrix,delimiter=",")
        # print(distance_matrix)
        
        highly_correlated = np.sort(distance_matrix)
        # np.savetxt("highly_correlated.csv",highly_correlated,delimiter=",")
        highly_correlated = highly_correlated[:,1]
        # print(highly_correlated[:,1])
        return np.mean(highly_correlated)


    def get_tree_structure(self, true_y_df):

        print("Obtaining trees structure")
        pb = tqdm(total = len(self.trees) )
        structure = []
        for tree in self.trees :
            # print(tree.root.instance_index)
            structure.append( tree.root.get_structure(true_y_df) ) 
            pb.update(1)
        return structure
    
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
        pbar = tqdm(total = X.shape[0])
        for index, row in X.iterrows():
            # print(f"I'm trying with this one {index}")
            tree_counter = 0
            instance_prediction = np.zeros(shape=[self.labels.shape[1]])
            instance_probability= np.zeros(shape=[self.labels.shape[1]])
            for tree in self.trees:
                # print(f"tree {tree_counter}")
                tree_counter += 1
                prd, prb = tree.root.predict_with_proba(row, original_labels=y_true[y_counter] if y_true is not None else None)
                #print(prd)
                
                # turn off when testing probabiluty based majority voting
                instance_prediction +=  np.array( prd)/len(self.trees)
                instance_probability += np.array( prb)/len(self.trees)

                # testing probability based majority voting, it works, especially to generate a bigger precision
                #instance_prediction += np.array( prb)

            #testing probability based majority voting, it works, especially to generate a bigger precision
            #instance_prediction = np.rint( instance_prediction/len(self.trees) ) # normalize wrt the forest size. Is the same as adding the probs of being 1 or 0... 
            predictions.append(instance_prediction)
            probabilities.append(instance_probability)
            y_counter += 1
            
            # 0.4 could be moved... based on calibration,  this is joining all the trees
            pbar.update(1)
        
        pbar.close()
        return np.where( np.array(predictions)>=0.5,1,0 ) , np.array(probabilities)


    def predict_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return prob
            