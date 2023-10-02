import pandas as pd
from random import random, sample
import numpy as np
import traceback
from voronoi import VoronoiClassifier
from numpy.random import default_rng
import warnings
# from pyhull.voronoi import VoronoiTess

generator = default_rng()

class DecisionTreeNodeV2:
    def __init__(self, node, index,labels_data,level = 0, is_leaf = False, dataset = None, do_balance_local_to_global = True ):
        self.regions = []
        self.parent = node
        self.instance_index = index
        self.is_leaf = is_leaf
        self.right = None
        self.left = None
        self.level = level
        self.ones_distribution_total = None
        self.decision_column_label = ""
        self.decision_column_value = 0
        self.balance_local_to_global =  do_balance_local_to_global
        self.centroid = None
        self.labels_classifiers = []
        self.acc_per_classifier = []
        # self.corr = None
        if( dataset is not None):
            # print(labels_data)
            self.centroid  = np.mean( dataset.loc[self.instance_index].to_numpy() , axis = 0)
            # print(self.centroid)
        # when index is assigned, check the different conditions.
        # Amount of nodes respect to total (hyperparam)
        # Total set labeled
        # and set itself as a leaf node. If is a leaf node, algorithm won't get next part.
    def set_decision_column(self, label, value):
        self.decision_column_label = label
        self.decision_column_value = value
    def set_left(self, node):
        self.left = node
    def set_right(self, node):
        self.right = node
    def calculate_labeled_ratio(self, labels):
        labels_on_instances = labels.loc[self.instance_index.array]
        # dataframe with the count of -1,0,1  for each of the instances
        return labels_on_instances.apply(pd.value_counts)
    def set_column_value(self, x, l, dist, dataset_ones_distribution):
        if x >= 0: # stay with the value already assigned, these are labeled instances
            return x
        #print(f'scv: {x} {l} {dist[l]}')
        # quesdtion , which is best... design experiment

        
        # r = random()*dist[l]
        
        if( not  self.balance_local_to_global):
            if x >=  0.5 :
                return 1
            return 0
        """
         modif, Andres formula
         lo modifuque de regreso por pruebas despues de modificar el get_compatibility
         """
        r = random()
        inv_dis = dist[l]
        ones_zeros_sum = dataset_ones_distribution[l]*inv_dis+ (1-dataset_ones_distribution[l])*(1-inv_dis)
        p1 = inv_dis*(1-dataset_ones_distribution[l])/(1-(ones_zeros_sum))
        if(r<p1): # if less than the extra probability assigned to 1, then is 1
            return 1
        return 0

    """
    This is getting the ones distribution
    """
    def fill_ones_distribution(self, labels, dataset_ones_distribution):
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        if not self.is_leaf:
            self.left.fill_ones_distribution(labels,dataset_ones_distribution)
            self.right.fill_ones_distribution(labels,dataset_ones_distribution)
            return
        #take the sum of all 1's of each column of the labeled instances
        # get only labels vectors of this nodes:
        instances_labels = labels.loc[self.instance_index]
        # another way to get the -1,0,1 : calculate_labeled_ratio().
        distribution = (instances_labels.replace(-1,0)).sum()
        labeled_data_count =  (instances_labels.replace({0:1,-1:0})).sum()
        ones_distribution_total = distribution/(labeled_data_count+0.00001)
        # gets one of these by each leaf node
        self.ones_distribution_total = ones_distribution_total


    """
    This is actually doing two things, getting the ones distribution and filling the -1s
    """
    def fill_semisupervised(self, labels, dataset_ones_distribution, report_final_data_distribution, dataset = None, out_of_bag_index = None):
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        if not self.is_leaf:
            a = self.left.fill_semisupervised(labels,dataset_ones_distribution,report_final_data_distribution, dataset = dataset, out_of_bag_index = out_of_bag_index)
            b = self.right.fill_semisupervised(labels,dataset_ones_distribution, report_final_data_distribution , dataset = dataset, out_of_bag_index = out_of_bag_index)

            

            # just the root will report this. As this is multithreaded, is not a simple return, but rather a callback
            if(report_final_data_distribution != None):
                report_final_data_distribution(  a + b )
            return a + b # the sum of both branches 
            
        #take the sum of all 1's of each column of the labeled instances
        # get only labels vectors of this nodes:
        instances_labels = labels.loc[self.instance_index]
        # another way to get the -1,0,1 : calculate_labeled_ratio().
        # print(instances_labels)
        supervised = instances_labels[ instances_labels[instances_labels.columns[0]] != -1].index
        unsupervised = instances_labels[ instances_labels[instances_labels.columns[0]] == -1].index

        # print(f" SUPERVISED INSTANCES, Going to voronoi {len(list(supervised))} {supervised} ")
        distribution = (instances_labels.replace(-1,0)).sum()
        labeled_data_count =  (instances_labels.replace({0:1,-1:0})).sum()
        ones_distribution_total = distribution/(labeled_data_count+0.00001)
        do_default = False
        # print( len(list(supervised)) )
        if( dataset is not None and len(list(supervised))>1 ):
        
            columns_for_voronoi = 3
            
            # should have a method for correlated, non correlated or totally random
            # we could change from supervised only to instances here, to check the entire space... even the unlabeled ones
            ds_01 = dataset.loc[supervised].to_numpy() 
            #print(f"DS SHAPE {ds_01.shape}")
            
            impossible_corr_columns = np.all( ds_01 == ds_01[0,:] , axis=0)
            impossible_idx = np.argwhere( impossible_corr_columns == True).flatten()
            # print(f'{impossible_idx} {np.all( ds_01 == ds_01[0,:] , axis=0)}  --> DS SHAPE {ds_01.shape}')
            # print(impossible_idx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = np.corrcoef(  ds_01 , rowvar = False )
            
            # print(corr)
            best_correlations = np.where( (corr) > 0.75, 1, 0).sum(axis=0)
            best_independent = np.where( (corr) < 0.25, 1, 0).sum(axis=0)
            #print(f'{corr.shape} and {subset_idx}')
            #print( f'{corr[subset_idx,:][:,subset_idx]}' )
            from_arr = best_correlations # or best_correlations
            all_sum = from_arr.sum()
            if( all_sum == 0 ): # there was not best best_independent or correlations
                from_arr = 1 - from_arr # equal all to 1
                all_sum = len(from_arr)
            order_index = from_arr.argsort()
            from_arr.sort()
            for idx in impossible_idx:
                all_sum -= from_arr[idx] 
                from_arr[idx] = 0
            prob_sort = from_arr/all_sum
            # print(f'{impossible_idx} {from_arr} = {all_sum} = {prob_sort}')
            random_selection = list(generator.choice(order_index , columns_for_voronoi+2, replace=False, p= prob_sort, shuffle=False))

            
            cols =[ i for i in range(0  , len(dataset.columns.tolist()) ) ]
            
            # 3 could be a hyper param. no more than 9. have 2 spare columns just in case one of them has only one value, 1 opportunity
            #subset_pairs = sample( list(enumerate(dataset.columns.tolist())),columns_for_voronoi+2)
            dscl = dataset.columns.tolist()
            subset = [dscl[n] for n in random_selection]
            subset_idx = random_selection
            
            #subset_pairs = [ (a,dscl[b])  for a in random_selection] 
            #subset = [n[1] for n in subset_pairs]
            #subset_idx = [n[0] for n in subset_pairs]
            
            # print(f'{ds_01.shape} \n bc: {best_correlations} \n bi: {best_independent} ==>\n {order_index} || {random_selection}\n{subset_idx}')

            values = dataset.loc[supervised][subset] # just a subset of the columns
            unique_values = values.nunique()
            #print(subset)
            #print(unique_values)
            if(1 in unique_values.tolist() ):
                # one opportunity to rescue this case. 
                to_remove = unique_values[unique_values == 1].index.tolist()
                if( len(subset) - len(to_remove) >= columns_for_voronoi ): #
                    subset = [sbi for sbi in subset if sbi not in to_remove]
                    one_to_remove = sample(list(enumerate(subset)),1)
                    #print(f"Trying to remove {one_to_remove} from {subset_idx} {subset}")
                    subset.remove( one_to_remove[0][1] )
                    subset_idx.remove( subset_idx[one_to_remove[0][0]] )
                    
                    subset = subset[0:columns_for_voronoi]
                    subset_idx = subset_idx[0:columns_for_voronoi]
                    
                    values = dataset.loc[supervised][subset] # just a subset of the columns
                    unique_values = [0] # do :D 
                else:
                    unique_values = [1] # could not rescue this case
                    pass
            else:
                random_selection = random_selection[0:columns_for_voronoi] # just the first three
                subset = [dscl[n] for n in random_selection]
                subset_idx = random_selection
                
                values = dataset.loc[supervised][subset] # just a subset of the columns
                unique_values = [0] # dont need to check
        else:
            do_default = True

        
        if(not do_default and dataset is not None and len(list(supervised)) > 4 and (1 not in unique_values)  ):
            try:

                values = values.to_numpy()
                v_labels = instances_labels[ instances_labels[instances_labels.columns[0]] != -1].to_numpy()
        
            
                
                self.labels_classifiers = []
                
                #print(f"Instances are  {values}")
                
                for i in range(0,v_labels.shape[1]):
                    #print(f"Subset index is {subset_idx}")
                    self.labels_classifiers.append( VoronoiClassifier(subset_idx) ) 
                    self.labels_classifiers[i].fit(values, v_labels[:,i].reshape(-1,1) )
                
                if not (out_of_bag_index is None):
                    # test everything, invert if necessary
                    test_values = dataset.loc[out_of_bag_index].to_numpy()
                    test_labels = labels.loc[out_of_bag_index].to_numpy()
                    
                    diff = []
                    for tv in range(0, test_values.shape[0] ):
                        r = test_values[tv]
                        prediction = []
                        for c in self.labels_classifiers:
                            p = c.predict(r)
                            prediction.append(p[0])
                        real = np.array(test_labels[tv])
                        diff.append(real  - prediction) 
                        #print(f'{prediction} vs {real} ')
                        
                    d = np.abs(np.array(diff))
                    errs = np.average(d,axis=0) # errs are actually accuracy
                    # print("Out of bag samples")
                    # print(errs)
                    self.acc_per_classifier = errs
                    #for e in range(0,len(errs)):
                        #if( errs[e] > .8 ): # hyperparam??? 
                        #self.labels_classifiers[e].invert_output()
                    
                instances_to_fill = dataset.loc[unsupervised] # [subset]
                
                if(not instances_to_fill.empty):
                    def predict_row(x):
                        prediction = []
                        for c in self.labels_classifiers:
                            p = c.predict(x)
                            prediction.append(p[0])
                        return pd.Series(prediction, index=labels.columns.tolist())
                    unsupervised_predictions = instances_to_fill.apply( predict_row , axis=1, result_type='expand' )
                    distribution2 = unsupervised_predictions.sum()
                    distribution = distribution.add(distribution2)
            except Exception as e:
                do_default = True  
                self.labels_classifiers = [] # leave no trail
                print(traceback.format_exc())
                # if voronoi cannot be done is because the model had too few instances. 
                # another cause is that instances are cocircular or cospherical
                # sometimes there is an error inside the method on the map.ridges check for pairs...
                # missing classifying with voronoi model, instead of the distribution. 
                # print(f"Responsible: {dataset.loc[supervised][subset]}")
        else:
            do_default = True        
                # print(f"Exception on voronoi :: {e}" )
            # should avoid the calculation via Andres' Formula. Instead train the minimodel
        if( do_default ):
            # on random forest, update cannot happen, at least not on the original data set. Ones distribution from each of them and locally could be updated,but
            # on self.ones_distribution_total
            # should we want to update, they instances have to be classified normally and update the ones ones_distribution_total
            
            # for each column, apply the function, should be faster than for each row.
            # also, this set_column_value could be a funmction passed by reference, to be able to change it

            # we could filter instances_labels to only the ones that do have a -1.... 
            # this has to change to cover the case on partial labeling. 
            instances_to_fill = instances_labels[ instances_labels[ list(instances_labels.columns)[0] ] == -1  ]
    
            for label, column_series in instances_labels.items() :
                instances_labels.loc[  instances_labels[ label ] == -1  , label] = column_series.apply(lambda x: self.set_column_value(x,label,ones_distribution_total,dataset_ones_distribution) )    
                # instances_labels[label]=column_series.apply(lambda x: self.set_column_value(x,label,ones_distribution_total,dataset_ones_distribution) )

            # this do not updates original label dataset 
            distribution = instances_labels.sum()


        ones_distribution_total = distribution/instances_labels.shape[0]
        self.ones_distribution_total = ones_distribution_total

        # if the root is the only node....
        if(report_final_data_distribution != None):
            report_final_data_distribution(  distribution )
            
        return distribution # the sum of labels of the instances of this node 
            

    def get_column_prediction(self, x):
        r = random()
        """ originally
        if x >=  0.5 :
            return 1
        return 0

        """
        # another idea: calculate the relation in this nodes between the features and the labels. The average of the features or something
        if r >=  1-x : # the bigger the 1, the most probable to be 1
            return 1
        return 0

    def get_proba_column_prediction(self, x):
        return x

    def get_prediction_with_proba_column_prediction(self, x):

        r = random()

        if r >=  1-x : # the bigger the 1, the most probable to be 1
            return 1, x
        return 0, x

    def predict_with_proba(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = []
            probability = []
            d = np.inf
            
            if( self.centroid is not None):
                d = np.mean(np.sum((instance - self.centroid)**2)**(0.5))
                # print(f'Centroid : {self.centroid} \t\t\t {d} \t\t\t prob {probability}')
            # prediction, probability = self.ones_distribution_total.apply(self.get_prediction_with_proba_column_prediction)


            if( len(self.labels_classifiers) > 0 ):
                prediction = []
                probability = []
                prediction2= []
                probability2= []
                for c,acc in zip(self.labels_classifiers,self.acc_per_classifier):
                    p = c.predict( instance )
                    prediction.append(1-p[0])
                    probability.append( ((1-p[0])-.5)*acc+.5 ) # if both probabilities are below 0.5 should be 0.5 and viceversa
                    
                for pr in self.ones_distribution_total:
                    pred, prob = self.get_prediction_with_proba_column_prediction(pr)
                    prediction2.append(pred)
                    probability2.append(prob)
                
                probability_total =np.asarray(probability)*.40 + np.asarray(probability2)*.60
                # no probability available... I do not know if the distance to voronoi planes can be represented as probabilities 
                # print( f'{np.asarray(prediction)} {np.asarray(probability2)} with {np.asarray(probability)} => {np.asarray(probability_total)} {original_labels} \n {self.corr}' )
                return np.asarray(prediction) , np.asarray(probability_total) , d    
            else:
                for pr in self.ones_distribution_total:
                    pred, prob = self.get_prediction_with_proba_column_prediction(pr)
                    
                    prediction.append(pred)
                    probability.append(prob)
                
                
                if( log_file is not None):
                    scores = ["%.6f" % x for x in self.ones_distribution_total]
                    vector_score = " ".join(scores)
                    p = " ".join(["%.1f" % x for x in prediction])
                    # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                    log_file.write(f'{vector_score}\n')
                    #log_file.write("\n----------------------------------------------------------------------------------\n")

                #if(original_labels != None)
                # this is just to be compliant with other calls.
                return np.asarray(prediction) , np.asarray(probability) , d
                # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] <= self.decision_column_value

        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict_with_proba(instance,  original_labels = original_labels , log_file = log_file, level = level+1)
        return self.right.predict_with_proba(instance , original_labels = original_labels , log_file = log_file, level = level+1)


    def predict_proba(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = self.ones_distribution_total.apply(self.get_proba_column_prediction)

            if( log_file is not None):
                scores = ["%.6f" % x for x in self.ones_distribution_total]
                vector_score = " ".join(scores)
                p = " ".join(["%.1f" % x for x in prediction])
                # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                log_file.write(f'{vector_score}\n')
                #log_file.write("\n----------------------------------------------------------------------------------\n")

            #if(original_labels != None)

            return prediction.values
            # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] <= self.decision_column_value

        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict_proba(instance, log_file = log_file, level = level+1)
        return self.right.predict_proba(instance ,log_file = log_file, level = level+1)

    """
        force prediction will have a purpose when a validation for the amount of labeled data is done.
        When the amount of labeled data is no enough, it will have to force the return prediction of the father (not implemented)
        This is not needed as the method on semisupervised step updates everything
    """
    def predict(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = self.ones_distribution_total.apply(self.get_column_prediction)

            if( log_file is not None):
                scores = ["%.6f" % x for x in self.ones_distribution_total]
                vector_score = " ".join(scores)
                p = " ".join(["%.1f" % x for x in prediction])
                # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                log_file.write(f'{vector_score}\n')
                #log_file.write("\n----------------------------------------------------------------------------------\n")

            #if(original_labels != None)

            return prediction.values
            # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] <= self.decision_column_value
        
        
        
        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict(instance, log_file = log_file, level = level+1)
        return self.right.predict(instance ,log_file = log_file, level = level+1)

    def __str__(self):
        tabs = "".join(["\t"]*self.level)
        return f'{tabs}({self.level}) node: ({self.is_leaf}) [({self.instance_index.size})] {self.instance_index}'

    def printRules(self, file, level = 0):
        st = "\n"
        st = st + "\t"*level

        if self.is_leaf :
            prediction = self.ones_distribution_total.apply(self.get_column_prediction)
            p = ""
            for v in prediction:
                p = p + f' {v}'
            scores = ["%.6f" % x for x in self.ones_distribution_total]
            ones_d = " ".join(scores)
            st = st + f"pred on {len(self.instance_index)} nodes is pred_labels: [{p}] dist:[{ones_d}]"
        else:
            st = st + f"Decision on col:{self.decision_column_label} with value {self.decision_column_value} on {len(self.instance_index)} nodes"
        file.write(st)
        if not ( self.left is None ):
            self.left.printRules(file, level+1)
        if not ( self.left is None ):
            str = self.right.printRules(file,level+1)

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
