import pandas as pd
from random import random, sample
import numpy as np
import traceback
from numpy.random import default_rng
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import traceback
import time
import sklearn
from sklearn import manifold
from .SSLearnerLeafNN import SSLearnerLeaf
from sklearn.metrics.pairwise import  euclidean_distances
from .utils import pairwise_distances


generator = default_rng()

def get_min(x):
    # print("tryinm to get min from " , x)
    try:
        return np.min(x)
    except:
        pass
    return 0.0
def get_max(x):
    try:
        return np.max(x)
    except:
        pass
    return 0.0


class DecisionTreeNodeV2:
    
    node_id = 0
    function_to_draw = None
    def __init__(self, node, index,labels_data,level = 0, is_leaf = False, dataset = None , hyper_params_dict = None,tree_id=0 ):
        self.id = DecisionTreeNodeV2.node_id 
        DecisionTreeNodeV2.node_id += 1
        self.parent = node
        self.instance_index = index
        self.right = None
        self.left = None
        self.level = level
        self.label_vector = None
        self.right_centroid = None 
        self.left_centroid = None
        self.tree_id = tree_id
        self.labels = labels_data
        self.dataset = dataset
        #self.is_leaf = is_leaf
        self.hyper_params_dict = hyper_params_dict
        self.is_leaf = False
        #self.do_label_proc = random() > hyper_params_dict["eta"]
        if(is_leaf):
            self.set_leaf(True) # calculate the model 
        self.decision_columns = None
        self.decision_value = -1
        self.get_distance_values = None
        self.decision_column_value_tuple = None
        self.global_labels_distribution = None

    def set_global_labels_distribution(self, global_labels_distribution):
        self.global_labels_distribution = global_labels_distribution
        
        if(self.left is not None):
            self.left.set_global_labels_distribution(global_labels_distribution)
        if(self.left is not None):
            self.right.set_global_labels_distribution(global_labels_distribution)

    def set_decision_value(self, value):
        self.decision_value = value
        
    def set_decision_instance_values(self, get_distance_values):
        self.get_distance_values = get_distance_values
    
    def set_decision_column_and_value(self, col, val):
        self.decision_column_value_tuple = (col,val)
        
    def set_decision_columns(self, cols):
        self.decision_columns = cols
    
    def set_left_centroid(self, centroid):
        self.left_centroid = centroid
    
    def set_right_centroid(self, centroid):
        self.right_centroid = centroid
        
    def get_instance_index(self):
        return self.instance_index

    def set_decision_column(self, label, value):
        self.decision_column_label = label
        self.decision_column_value = value
    def set_left(self, node):
        self.left = node
    def set_right(self, node):
        self.right = node
        
    def fill_semisupervised(self):
        pass

    def draw_data_set(self, dataset=None, labelset = None):
        
        # print("-*********************2", dataset)
        a  = int(random()*1000)
        b=  int(random()*1000)
        #joined_labels = self.labels.loc[self.instance_index ].copy()
        
        #final_joined_labels = joined_labels[self.labels.columns[0]].map(str)
        for label in range(0, len(labelset.columns.array)):
            
            l = labelset.columns[label]
            example = dataset.copy()
            
            # print( example.columns)
            example["category"] = labelset[l]
            example = example.reset_index()
            xvars = example.columns[1:50]
            yvars = ['index']
            # DecisionTreeNodeV2.function_to_draw(example,xvars, yvars,f'output_leaf_parent_{b}_node_{a}_label_{l}.png')
    
            #final_joined_labels = final_joined_labels + joined_labels[l].map(str)
        #print(final_joined_labels)
        # final_joined_labels.loc[row.name] = 'new_one'
        #instances_to_get =  self.instance_index 
        #print(final_joined_labels)
        #example = self.dataset.loc[self.instance_index].copy()
        # example.loc[row.name] = row
        #example["category"] = final_joined_labels
        #example["one"] = 0
        #print(xvars)
        #print(example)
        
        
        
        
    def set_leaf(self, is_leaf):
        
        self.is_leaf = is_leaf
        
        # print the dataset with their labels... 
        
        """
        if(is_leaf):
            example = SSLearnerLeaf.complete_train_dataset.loc[self.instance_index, list(self.dataset.columns.array)]
            example["category"] = SSLearnerLeaf.complete_train_dataset.loc[self.instance_index,self.labels.columns[0]]
            #example.loc[ right_cluster , "category" ] = 1
            example = example.reset_index()
            xvars = example.columns[1:50] #  just a subset
            yvars=['index']
            example = example.loc[:] # to see only the same subset as cols
            DecisionTreeNodeV2.function_to_draw(example,xvars, yvars,f'output_leaf_node_{self.id}_{0}_alt.png')
        """    
        if(not is_leaf):
            return
        
        if(self.hyper_params_dict["use_complex_model"]):
        
            self.model = SSLearnerLeaf(hyper_params_dict=self.hyper_params_dict)
            
            # instance_space, label_space = 
            """
            if(is_leaf):
                this_df = SSLearnerLeaf.complete_train_dataset.loc[self.instance_index]
                labeled_index = self.labels.loc[self.instance_index]
                labeled_index = labeled_index[ labeled_index[self.labels.columns[0]] != -1 ].index
                this_df["supervised"] = 0
                this_df.loc[labeled_index, "supervised"] = 1
                a  = int(random()*1000)
                this_df.to_csv(f"./datasets_tests/e_ds_{a}.csv")
            """
            #labels = self.labels.loc[self.instance_index]
            #supervised_instances = labels[ labels[labels.columns[0]] != -1 ].index 
            self.model.set_dataset_indices(self.instance_index )
            #self.model.set_dataset_indices(supervised_instances )
            
            self.model.fit(self.dataset,self.labels, self.tree_id, self.level ) # , self.parent.decision_columns for the test of KNN 
            # self.draw_data_set(dataset=instance_space, labelset =label_space )
        else:
            dataset = self.labels.loc[self.instance_index]
            #print(dataset)
            supervised_instances = dataset[ dataset[dataset.columns[0]] != -1 ].index 
            labels = dataset.loc[ supervised_instances ]    
            #print(labels)
            label_vector = labels.to_numpy().mean(axis=0)
            #print(f'Label vector is {label_vector}')
            self.label_vector = label_vector
        
    def get_draw_repr(self):
        # print(f' exporting {self.id} parent : {self.parent.id if self.parent is not None else -1 }')
        me = self.id
        left = f'' if self.left is None else self.left.get_draw_repr()
        right = f'' if self.right is None else self.right.get_draw_repr()
        return me + '\n' + left + '\n' + right
    

    def get_structure(self, true_y_df):
        structure = {"left":None, "right":None}


        structure["left"] = None if self.left is None else self.left.get_structure(true_y_df)
        structure["right"] = None if self.right is None else self.right.get_structure(true_y_df)

        structure["depth"] = 1 + max( 0 if self.left is None else structure["left"]["depth"], 0 if self.right is None else structure["left"]["depth"] )
        structure["index"] = self.instance_index.array
        structure["node_id"] = self.id
        structure["tree_id"] = self.tree_id
        structure["columns"] = self.decision_columns
        structure["is_leaf"] = self.is_leaf

        structure["supervised"] = self.labels.loc[self.instance_index.array]
        df = structure["supervised"]
        structure["supervised"] = df[df[self.labels.columns[0]] > -1 ].index.array
        structure["unsupervised"] = df[df[self.labels.columns[0]] == -1 ].index.array
        

        if(self.decision_columns is not None):
            colsA = set() if self.left is None else structure["left"]["joint_columns"] 
            colsB = set() if self.right is None  else structure["right"]["joint_columns"] 
            structure["joint_columns"] = set(self.decision_columns).union(colsA).union(colsB)
        else:
            structure["joint_columns"] = set()

        distance_matrix = pairwise_distances( true_y_df.loc[self.instance_index.array].to_numpy(), metric='cosine')
        this_avg = 0
        half_matrix = np.tril(distance_matrix)
        n = distance_matrix.shape[0] - 1 # n its the amount of elements, n-1 would take out the distances of one to itself
        elements = n*(n+1)/2 # just get the diagonal of the matrix, originally n*(n+1)/2 from https://en.wikipedia.org/wiki/Summation
        if(elements==0):
            this_avg= 0.0 # is just the distance of one to itself
        else:
            this_avg = half_matrix.ravel().sum()/elements
    
        structure["label_inner_distance"] = this_avg

        # all the features here
        cat_feats = self.node_hyper_params["categorical_dictionary"].get_categorical_features( self.dataset.columns, arr_name_based=True )
        # this one is not tested yet. 
        distance_matrix = pairwise_distances( self.dataset.loc[self.instance_index.array].to_numpy(), metric='gower', cat_features=cat_feats)
        this_avg = 0
        half_matrix = np.tril(distance_matrix)
        n = distance_matrix.shape[0] - 1 # n its the amount of elements, n-1 would take out the distances of one to itself
        elements = n*(n+1)/2 # just get the diagonal of the matrix, originally n*(n+1)/2 from https://en.wikipedia.org/wiki/Summation
        if(elements==0):
            this_avg= 0.0 # is just the distance of one to itself
        else:
            this_avg = half_matrix.ravel().sum()/elements
    
        structure["gower_inner_distance"] = this_avg

        return structure

    def predict_with_proba(self, row, original_labels=None):
        #print(row)
        
        val = None 
        
        if( self.label_vector is not None):
            return np.where(self.label_vector >= 0.5,1,0) , self.label_vector
            
        if( self.decision_columns is not None): # this does not happens on leaves
            # print(self.decision_columns)
            val = row[self.decision_columns]
        #print(val)
        
        if( self.is_leaf ):
            r = np.array([row.to_numpy()])
            pred, prob = self.model.predict_with_proba(r, self.global_labels_distribution)

            # pred = pred[0]
            return pred, prob
            
        # print(f'{self.left_centroid.to_numpy()} {self.right_centroid.to_numpy()} {val.to_numpy()}')
        
        # distances
        
        # we could calculate only a centroid over all the decision columns
        # and the total_min and total_max could be just another variable... <---- optimization. 
        # if we need the data, is already on the next nodes. On prediction we could make things 
        # a little bit quicker
        # could also test if its better to get all the instances, just the supervised ones, or a weight between them 
        left_side = self.dataset.loc[ self.left.instance_index , self.decision_columns ].to_numpy()
        right_side = self.dataset.loc[ self.right.instance_index , self.decision_columns ].to_numpy()

        lsmin  = np.apply_along_axis( get_min , 0, left_side).astype(np.float32)
        lsmax = np.apply_along_axis( get_max , 0, left_side).astype(np.float32)
        
        rsmin  = np.apply_along_axis( get_min , 0, right_side).astype(np.float32)
        rsmax = np.apply_along_axis( get_max , 0, right_side).astype(np.float32)
        
        
        try:
            total_min = np.min( np.array([lsmin,rsmin]) , axis = 0)
            total_max = np.max( np.array([lsmax,rsmax]) , axis = 0)
        except Exception as e:
            print("these are the cols --- " , self.decision_columns)
            print("rrrr- should be one vector--- " , lsmin)
            print("rrrrr- should be one vector--- " , rsmin)

            print(left_side)
            print(right_side)
            
            raise(e)

        # print("- should be one vector--- " , total_min)
        # print("+ should be one vector+++  " , total_max)
        
        # we should get the max and min distances here , or when it is leaf, and pass it to pairwise_distances
        # check what happens with numpy trying top get max of strings
        # we should change any np nans to 0. 
        # the total min and total max homologate the distances on the gower. Just used when the gower is on. 
        cat_feats = self.hyper_params_dict["categorical_dictionary"].get_categorical_features( self.decision_columns, arr_name_based=True )

        left_distances = pairwise_distances(  val.to_numpy().reshape(1,-1) , left_side ,metric=self.hyper_params_dict['distance_function'] , min_max_array = [total_min,total_max] , cat_features=cat_feats )
        right_distances = pairwise_distances(  val.to_numpy().reshape(1,-1) , right_side,metric=self.hyper_params_dict['distance_function'] , min_max_array =  [total_min,total_max]  , cat_features=cat_feats )
        #print( np.min(left_distances) , " ============ " , np.min(right_distances) )
        if( np.min(left_distances) < np.min(right_distances) ):
            
            p1,p2 = self.left.predict_with_proba(row, original_labels=original_labels)
            return p1, p2
        p1, p2 = self.right.predict_with_proba(row, original_labels=original_labels)
        return p1, p2 
            
            
class DecisionTree:
    
    def set_truth_rate(self, tr):
        self.truth_rate = tr
    def get_truth_rate(self):
        return self.truth_rate
    def add_root_node(self,node):
        self.root = node
    def print_tree(self):
        print( self.root.print_node(0) )
    def classify(self, row):
        return self.root.decide(row)
    def read_tree(self, file):
        print("when reading file rebuild tree to classify")
    