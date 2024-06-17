
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import uuid
from .DTNodeMPSC import DecisionTree, DecisionTreeNodeV2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
import tracemalloc

import pandas as pd

from multiprocessing.shared_memory import SharedMemory

from numpy.random import default_rng
import math
import threading
from multiprocessing import Process
import random
from .utils import pairwise_distances, silhouette_score, print_numba_signatures
from .SSClustering import SSClustering

from sklearn.metrics import silhouette_score as sk_silhouette_score
import warnings
from time import time

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


class CategoricalDictionary():
    """
    if X is the dataset from emotions.csv

     C = CategoricalDictionary(X.columns, X.dtypes)
    self.node_hyper_params["categorical_dictionary"]  = C
    
    print(C.is_category)
    print(C.cat_map)

    cat_feats = C.get_categorical_features( [1,25,68] )
    print("Cat feats " ,  cat_feats ) # should be [2]

    cat_feats = C.get_categorical_features( ["col_1","col_25","col_68"], arr_name_based=True )
    print("Cat feats from names" ,  cat_feats ) # should be [2]

    """

    @classmethod
    def get_cat_codes(cls, df, col_array):
        mappings = dict()
        for col in col_array:
            mappings[col] = dict()
            # generate the mapping for the column
            for c in range(len(df[col].cat.categories)):
                mappings[col][ df[col].cat.categories[c] ] = c

            df[col] = df[col].cat.codes 
            df[col] = pd.to_numeric( df[col] , downcast="float" ) #the idea is to get a final float32 array
        return mappings

    def __init__(self, pandas_columns, pandas_dtypes) -> None:
        self.is_category = []
        self.cat_map = {}
        self.columns = pandas_columns
        i = 0
        for col in range(len(pandas_columns)):
            self.cat_map[ pandas_columns[col] ] = col
            if( pandas_dtypes.iloc[col] == 'object' ):
                self.is_category.append(True)
            else:
                self.is_category.append(False)
                


    def get_categorical_features(self, arr, arr_name_based = False):  
        """
        check if elements in arr are in categorical names or in the values of categorical names
        gather all the ones that are. 

        Process is like this:
        For each of the elements in arr, 
            if the index is true on is_category
                add the index to an array.
                add 1 to index 
            else
                add 1  to index
        """
        to_look = []
        if(arr_name_based):
            for i in arr:
                to_look.append( self.cat_map[i] ) 
        else:
            to_look = arr

        idx = 0
        cat_feats = []
        for i in to_look:
            if( self.is_category[i] ):
                cat_feats.append(idx)
            idx+=1
        return cat_feats

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
        self.cosine_distances = None
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
            
        params = ['A','B','C','D','E','F','G']
        hyperparams_opt_sum = sum([ self.node_hyper_params['p'+p] for p in params ])

        for p in params :
            self.node_hyper_params['p' + p ] = self.node_hyper_params['p' + p ]/hyperparams_opt_sum

        self.node_hyper_params["output_quality"] = self.node_hyper_params.get("output_quality",False)
        self.trees_quantity = trees_quantity # at least one! solution with 1 decision tree should be equals to MLSSVRPRedictor
        self.leaf_relative_instance_quantity = leaf_relative_instance_quantity
        
        self.base_file_root  = bfr
        self.tag = tag
        self.complete_ss = complete_ss
        self.is_multiclass = is_multiclass
        self.do_random_attribute_selection = do_random_attribute_selection

        self.node_hyper_params["categorical_dictionary"] = None
        
  
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
            
            n_to_select = min( len(selectable_columns), n_to_select )
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
            silhouette_precomputed = X.to_numpy()
            # return (silhouette_score(silhouette_precomputed, labels.to_numpy().ravel(), metric='precomputed')+1)/2
            return (silhouette_score(silhouette_precomputed, labels.to_numpy().ravel())+1)/2
        
        return (sk_silhouette_score(X, labels)+1)/2 # this works when we want euclidean distance, but we already jhave gower distance inclduing categorical values

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
    
    def get_features_distance(self, X , dist = None):
        """
        We are not uysing the dist calculation WITH gower. 
        """
        # dist = pairwise_distances(X.to_numpy(),metric='euclidean')
        if( dist is None):
            if(self.node_hyper_params['distance_function'] == 'gower'):
                raise Exception("Distance calc. with gower is no supported" , "Do set the dist input when using gower")
            dist = pairwise_distances(X.to_numpy(),metric=self.node_hyper_params['distance_function'])
        mn = dist.min()
        mx = dist.max()
        div = (mx - mn)+0.00001
        dist = (dist-mn)/div
        return self.get_average_distance(dist)


    def get_option(self, instances, selected_cols, labels, label_index, df_labels_distances, min_supervised_per_leaf):
        
        correlation_data = None
        if(False):
            correlation_data = np.abs(instances[ selected_cols ].corr().to_numpy()).sum(axis=0)
            correlation_data = 1 - np.mean(correlation_data/correlation_data.shape[0])
        else:
            correlation_data = np.abs(instances[ selected_cols ].corr().to_numpy())
            correlation_data = np.tril(correlation_data)*correlation_data
            els = correlation_data.shape[0] -1 # tril is not counting the main diagonal
            els = els*(els+1)*0.5
            correlation_data = 1 - correlation_data.ravel().sum()/els # invert the one
            
            
        #print("Correlation data " , correlation_data)
        do_gower= True if self.node_hyper_params["distance_function"]=='gower' else False
        
        # this is the most expensive call of the whole algorithm-> (u+s)^2*features

        
        # cat_feats = self.node_hyper_params["categorical_dictionary"].get_categorical_features( [1,25,68] )
        cat_feats = self.node_hyper_params["categorical_dictionary"].get_categorical_features( selected_cols, arr_name_based=True )
        f_distance_matrix = pairwise_distances( instances[ selected_cols ].to_numpy(),metric=self.node_hyper_params['distance_function'], cat_features=cat_feats)
        
        f_distance_matrix = pd.DataFrame(data=f_distance_matrix,columns= instances[ selected_cols ].index )
        f_distance_matrix.set_index(instances[ selected_cols ].index, inplace=True)
        instances_index = instances[ selected_cols ].index
        sscluster = SSClustering(2, feature_distance_matrix=f_distance_matrix.to_numpy(), label_distance_matrix=df_labels_distances.loc[instances_index,instances_index].to_numpy() , just_base_cluster=True, use_gower=do_gower)
        
        
        #fit_time = time()
        sscluster.fit( instances[ selected_cols ] , labels , far_distance_on_attribute_space = True, overwrite_label_distance=df_labels_distances.to_numpy() )
        #print("SSClustering FIT time: " , (time() - fit_time) )
        
        #predict_time = time()
        labels_rbf = sscluster.predict(instances[ selected_cols ])
        #print("SSClustering Predict time: " , (time() - predict_time) )
        
        #labels_intersection = time()
        clustered_df = pd.DataFrame(data = labels_rbf)
        clustered_df.set_index(instances.index, inplace=True)
        
        left_cluster = clustered_df[clustered_df[0] == 0].index 
        right_cluster = clustered_df[clustered_df[0] == 1].index
        #print(f"-{tree_id} ",left_cluster)
        #+print("---")
        #print(f"+{tree_id} ",right_cluster)
        
        left_labels_index = list(set(left_cluster.array).intersection( set(label_index.array) ) )
        right_labels_index = list(set(right_cluster.array).intersection( set(label_index.array) )) 

        #print("Labels intersection: " , (time() - labels_intersection) )

        #print(f"-**{tree_id} ",left_labels_index)
        #print(f"-**{tree_id} ",right_labels_index)

        # print( df_labels_distances )
        if( len(left_labels_index) == 0 or len(right_labels_index) == 0 ):
            # print(f"ZERO! Label index err {left_labels_index} {right_labels_index}")
            # print(f"{node.id} ERROR; this is not a partition as there are no labeled samples on one side {len(left_labels_index)} { len(right_labels_index)} {instances.shape}")
            return None 
        
        #label_balance_score_time = time()

        label_balance_score_left = self.get_label_balance_score(labels.loc[left_labels_index])
        label_balance_score_right = self.get_label_balance_score(labels.loc[right_labels_index])
        
        #print("Label balance score : " , (time() - label_balance_score_time)  )

        label_balance_score  = (label_balance_score_left +  label_balance_score_right)*0.5 # bigger balance , bigger numbers, 
        c_score = label_balance_score
        

        #average_distance_time = time()

        left_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[left_labels_index , left_labels_index ]  )
        right_cluster_label_distance = self.get_average_distance( df_labels_distances.loc[right_labels_index , right_labels_index ]  )
        distance_avg = (left_cluster_label_distance+right_cluster_label_distance)/2.0

        #print("Average Distance on cluster score : " , (time() - average_distance_time)  )

        #s_score_time = time()
        if (self.node_hyper_params["pE"] <= 0):
            s_score = 0 # the same for everyone, but do not calculate
        else:
            # s_score = self.get_cluster_silhouette_score( instances[selected_cols] , clustered_df )
            s_score = self.get_cluster_silhouette_score( f_distance_matrix , clustered_df, True)
        #print("S Score: " , (time() - s_score_time)  )
        
        #feature_distance_time = time()
        attr_cluster_distance_A = self.get_features_distance(instances.loc[left_cluster,selected_cols] , dist= f_distance_matrix.loc[left_cluster,left_cluster])
        attr_cluster_distance_B = self.get_features_distance(instances.loc[right_cluster,selected_cols], dist = f_distance_matrix.loc[right_cluster,right_cluster])
        attr_cluster_distance =(attr_cluster_distance_A + attr_cluster_distance_B)*0.5 # lets try to find the clusters with the most space between them.
        
        #print("Feature Distance Score Time: " , (time() - feature_distance_time)  )
        #iq =  "   ---     ".join([ str(x) for x in  [len(left_labels_index), len(right_labels_index), len(left_cluster), len(right_cluster), min_supervised_per_leaf, self.get_leaf_instance_quantity() ] ])
        #print("instance quantities!!! " , iq  )
        if( len(left_labels_index)  <= min_supervised_per_leaf or 
            len(right_labels_index) <= min_supervised_per_leaf 
            or len(left_cluster) < self.get_leaf_instance_quantity()
            or len(right_cluster) < self.get_leaf_instance_quantity()
            ): # 2 and up
            # print(f"{tree_id} Label index err {iteration} {left_labels_index} {right_labels_index}")
            return None



        supervised_imbalance = min(len(left_labels_index), len(right_labels_index))/max(len(left_labels_index), len(right_labels_index))
        left_sup_unsup_ratio = len(left_labels_index)/len(left_cluster) 
        right_sup_unsup_ratio = len(right_labels_index)/len(right_cluster)
        sup_unsup_ratio_balance = min( left_sup_unsup_ratio , right_sup_unsup_ratio) / max( left_sup_unsup_ratio , right_sup_unsup_ratio)
        best_purity = 0
        
        
        # actual_coeff = (1-attr_cluster_distance)*self.node_hyper_params["pD"] +  (1-distance_avg)*self.node_hyper_params["pA"] + (1-s_score)*self.node_hyper_params["pB"] +  (c_score)*self.node_hyper_params["pC"] # c_score is already 
        actual_coeff = (1-distance_avg)*self.node_hyper_params["pA"] + supervised_imbalance*self.node_hyper_params["pB"] +  (sup_unsup_ratio_balance)*self.node_hyper_params["pC"] + (c_score)*self.node_hyper_params["pD"] +(s_score)*self.node_hyper_params["pE"]  + correlation_data*self.node_hyper_params["pF"] + (1-attr_cluster_distance)*self.node_hyper_params["pG"]
        
        #print(" checking values " , supervised_imbalance," ",left_sup_unsup_ratio," ",right_sup_unsup_ratio," ",sup_unsup_ratio_balance, " > ", actual_coeff )
        
        return left_labels_index,right_labels_index,actual_coeff,left_cluster,right_cluster,sscluster,actual_coeff,correlation_data

    # , tree , tree_id , instances, labels  ---> previously
    def generate_tree(self,ns, tid, shared_mem):
        #ns = tup[0]
        #tid = tup[1]
        
        # check if root node is None
        # the on should 
        # print(" Tree with proc " , mp.current_process().name )
        tree = DecisionTree()
        
        shared_mem.buf[tid:tid+1] = bytearray([1])
        
        self.cosine_distances = ns.cosine_distances
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

        shared_mem.buf[tid:tid+1] = bytearray([2])
        shared_mem.close()
        # tq_bar.update(1)
        print_numba_signatures()
        return tree #f"hello {i}"
    
    def generate_children(self, node,recursion, tree_id = 0 ,full_instances=None, full_labels=None ):
        # proxy
        return self.generate_childrenV1_1(node,recursion, tree_id, full_instances=full_instances, full_labels=full_labels  )
        
    def generate_childrenV1_1(self, node,recursion, tree_id = 0 , full_instances=None, full_labels=None):
        

        #generate_children_time = time()
        if(node.is_leaf):
            return 
        #generate_init_time = time()
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
        # supervised_examples = pairwise_distances( labels.loc[ label_index ].to_numpy() , metric='cosine' )
        #df_labels_distances = pd.DataFrame(data=supervised_examples, columns=label_index)
        #df_labels_distances.set_index(label_index, inplace=True)
        df_labels_distances = self.cosine_distances.loc[ labels.index , labels.index  ] # calculate once for all the system, and just use here

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
        #print("Generate Initialization time: " ,  (time() - generate_init_time) )

        if(node.level >= self.node_hyper_params["depth_limit"] ):
            node.set_leaf(True)
            return

        columns_selected_history = None

        while m < m_groups and iteration < max_iterations:
            optimization_task_time = time()
            m+=1
            iteration += 1
            #random_cols_time = time()
            cols_index = self.get_random_columns(n_features, selectable_columns, linear_selection=False)
            #print("Random cols time: " , (time() - random_cols_time) )

            # print("cols index " , columns_selected_history)
            ls = cols_index[0]
            x = instances.to_numpy()
            selected_cols = [instances.columns[i] for i in ls]
            
            res = self.get_option(instances, selected_cols, labels, label_index, df_labels_distances, min_supervised_per_leaf)
            #print(f"Optimization task time ({tree_id} - {node.id}): " , (time() - optimization_task_time) , " // "  , res )
                
            if( res is None):
                m -= 1
                #print("Optimization task time: " , (time() - optimization_task_time) )
                continue

            left_labels_index =res[0]
            right_labels_index =res[1]
            actual_coeff =res[2]
            left_cluster =res[3]
            right_cluster =res[4]
            sscluster =res[5]
            actual_coeff =res[6]
            correlation_data =res[7]

            new_best = False
            # print(f"---------------------- +++++++++++++ {tree_id} {actual_coeff}     {best_partition_coeff}" , flush=True)
                    
            if( actual_coeff > best_partition_coeff):
                    #print(f"{tree_id} {actual_coeff}     {best_partition_coeff}      {supervised_imbalance}_ {sup_unsup_ratio_balance}       {attr_cluster_distance} {distance_avg} {1-s_score} {c_score} ")
                    
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
            
            #columns_history_time = time()
            columns_selected_history.add( (tree_id, new_best,iteration,actual_coeff, correlation_data ,frozenset( cols_index[0]) ) )

            if(new_best and self.node_hyper_params["output_tree_sets"]):
                merged = pd.concat( [instances.loc[left_final_cluster_index,columns_to_select], labels.loc[left_final_cluster_index, :] ] , axis=1)
                merged["cluster"] = 0
                merged["center"] = 0
                if( cluster_centers is not None):
                    merged.loc[cluster_centers[0] , "center"] = 1
                merged.to_csv(f"nodes_story/{tree_id}_{node.level+1}_{m}_{iteration}_internal_left_{uuid.uuid4()}_{actual_coeff}.csv")

                merged = pd.concat( [instances.loc[right_final_cluster_index,columns_to_select], labels.loc[right_final_cluster_index, :] ] , axis=1)
                merged["cluster"] = 1
                merged["center"] = 0
                if(cluster_centers is not None):
                    merged.loc[cluster_centers[1] , "center"] = 1
                merged.to_csv(f"nodes_story/{tree_id}_{node.level+1}_{m}_{iteration}_internal_right_{uuid.uuid4()}_{actual_coeff}.csv")


            #print("Columns History time: " , (time() - columns_history_time) )


            #print("Optimization task time: " , (time() - optimization_task_time) )
            
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
            #print("Generate children time: " ,  (time() - generate_children_time) )
            return
    
        node.set_decision_columns(columns_to_select)
        
        #print("Generate children time: " ,  (time() - generate_children_time) )

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
  
    def error_handler(self, error):
        print("********°°°°°°°°°°°°°°°°°!!!!!!!!!!!!!!!!!!!°°°°" , error, flush=True)

    def fit(self, X, y):
        y = y.copy()
        #rint(self.unlabeledIndex)
        #print(self.compatibilityMatrix)
        #print(self.unlabeledIndex)
        self.dataset_assigned_label_count = None
        toSelect = None
        cantDo = (self.unlabeledIndex is None)
        
        C = CategoricalDictionary(X.columns, X.dtypes)
        self.node_hyper_params["categorical_dictionary"]  = C
        
        """
        Now, we must change the categorical columns to a code
        """
        cat_arr = C.get_categorical_features(X.columns,True)
        cat_names = []
        for i in cat_arr:
            cat_names.append( X.columns[i] )
        X[cat_names] = X[cat_names].astype("category")

        self.categoryMap = CategoricalDictionary.get_cat_codes(X, cat_names )
        # print("********************** Category Map " , self.categoryMap)

        
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

        tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage (prev to cosine distance) {current/1e6}MB; Peak: {peak/1e6}MB")

        # for the amount of trees needed, generate N decision trees and train them all. Remember to select a subset of the columns
        self.trees = []
        self.calculate_labels_distribution()
        
        

        mgr = Manager()
        ns = mgr.Namespace()
        ns.instances = X
        ns.labels = y

        cos_distances = pairwise_distances(y.to_numpy(),metric='cosine').astype(np.float16)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory WITH cosine distances {current/1e6}MB; Peak: {peak/1e6}MB")
        # df_labels_distances = pd.DataFrame(data=supervised_examples, columns=label_index)
        # df_labels_distances.set_index(label_index, inplace=True)

        cos_distances = pd.DataFrame( data=cos_distances, columns=y.index )
        cos_distances.set_index(y.index, inplace=True)

        ns.cosine_distances = cos_distances 
        print("Done with cosine distances")
        shared_mem = SharedMemory(create=True, size=self.trees_quantity)

        
        #from ctypes import c_wchar
        #update_list = [mp.RawArray(c_wchar, 2) for _ in range( self.trees_quantity)]
        #ns.pbar = pbar
        pool = mp.Pool(processes = self.njobs ) # this can grow with processors. 


        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage (prev to threads) {current/1e6}MB; Peak: {peak/1e6}MB")
        
        jobs = []
        tid = -1
        for i in range(0, self.trees_quantity):
            tid += 1
            jobs.append(
                pool.apply_async( self.generate_tree, args=(ns,tid,shared_mem ) , error_callback=self.error_handler) 
            )
        # result = pool.map_async( self.generate_tree, [(ns,i) for i in range(0,self.trees_quantity)] ) # it takes 22 secs to end them all
        pool.close()
        print("trees generated")
        
        
        # pool.join()
        pbar = tqdm(total = self.trees_quantity)

        ended = 0

        #for job in jobs:
        #    self.trees.append(job.get())
        #    pbar.update(1)
        
        jt = len(jobs)
        

        #for j in jobs:
        #    final_res.append("!")
        status = ['°','!','|']

        

        while ended < jt:
            results_str = ""
            for j in jobs:
                # print(f"Job { j.ready() }")
                # results_str +=  f"{j.ready()}".ljust(6,' ') + "|" 
                if(j.ready()):
                    self.trees.append(j.get())
                    jobs.remove(j)
                    pbar.update(1)
                    # final_res[ended] = "|"
                    ended += 1

                    #current, peak = tracemalloc.get_traced_memory()
                    #print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB Pool ")
                    #procs = mp.active_children()
                    #for p in procs:
                    #    print(" Proc "  , p.name , " Alive " , p.is_alive())

            #data = [int(shared_mem.buf[i]) for i in range(self.trees_quantity)]
            #results_str = "\r" + "".join( [ status[i] for i in data] )        
            #print(results_str , end='')


            

        #for tree in result.get():
        #    # print("returning " , tree)
        #    self.trees.append(tree)
        shared_mem.close()
        # release the shared memory
        shared_mem.unlink()

        pbar.close()
        
        print(f"Ended training")

        
        tracemalloc.stop()

        self.ready = True
        tree_index = 0
        return 

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

    def get_tree_structure_df(self, rules_list):

        print("Obtaining trees structure DATAFRAME")
        pb = tqdm(total = len(self.trees) )
        for tree in self.trees :
            # print(tree.root.instance_index)
            tree.root.get_structure_df(rules_list , 0 )
            pb.update(1)
        # the methods modify rules_db

        

    def get_tree_structure(self, true_y_df):

        print("Obtaining trees structure")
        pb = tqdm(total = len(self.trees) )
        structure = []
        for tree in self.trees :
            # print(tree.root.instance_index)
            structure.append( tree.root.get_structure(true_y_df ) ) 
            pb.update(1)
        return structure
    
    def predict(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return pred

    def predict_with_proba(self, X, print_prediction_log=False ,y_true = None, activations_list = None, explain_decisions = False):
        # check_is_fitted(self)
        #predictions = np.zeros(shape=[self.labels.shape[1]])
        #probabilities = np.zeros(shape=[self.labels.shape[1]])
        predictions = []
        probabilities = []
        y_counter = 0

        """
        CHANGE THE CATEGORIES!!!
        """
        
        for col in X.items():
            if col[0] in self.categoryMap:
                X[col[0]] = X[col[0]].map( self.categoryMap[col[0]] )
            
        #print("Check numba signatures...")
        #print_numba_signatures()

        pbar = tqdm(total = X.shape[0])
        rule_explanations = []
        if( y_true is not None):
            df_activations = pd.DataFrame()

        for index, row in X.iterrows():
            # print(f"I'm trying with this one {index}")
            tree_counter = 0
            instance_prediction = np.zeros(shape=[self.labels.shape[1]])
            instance_probability= np.zeros(shape=[self.labels.shape[1]])
            row = row.astype(np.float32)
            for tree in self.trees:
                # print(f"tree {tree_counter}")
                tree_counter += 1

                rule_explain_dict = None
                if(explain_decisions):
                    rule_explain_dict = row.to_dict()

                if y_true is not None:
                    prd, prb = tree.root.predict_with_proba(row, original_labels=y_true[y_counter] , activations_list = activations_list, explain_decisions = explain_decisions, rule_explain_dict=rule_explain_dict)
                else:
                    prd, prb = tree.root.predict_with_proba(row, original_labels=None, explain_decisions = explain_decisions, rule_explain_dict=rule_explain_dict)
                    
                #print(prd)
                if( explain_decisions ):
                    rule_explanations.append(rule_explain_dict)
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

            if(explain_decisions):
                rule_explain_dict = row.to_dict()
                rule_explain_dict["final_decision"]= np.where( np.array(instance_prediction)>=0.5,1,0 ) 
                rule_explain_dict["final_decision_probs"]=instance_probability
                rule_explanations.append(rule_explain_dict)

        
        pbar.close()
        #print("Final numba signatures")
        #print_numba_signatures()

        if(explain_decisions):
            rdf = pd.DataFrame(rule_explanations)
            rdf.to_csv("explanations.csv")
    
        return np.where( np.array(predictions)>=0.5,1,0 ) , np.array(probabilities)


    def predict_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return prob
            