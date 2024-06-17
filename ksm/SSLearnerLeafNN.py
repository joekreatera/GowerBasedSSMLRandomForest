import numpy
from numpy.random import default_rng
import uuid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
import pandas
import math
import random
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel,check_pairwise_arrays, euclidean_distances
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier

from .utils import pairwise_distances

import warnings    
rng = default_rng()

class SSLearnerLeaf(BaseEstimator, ClassifierMixin):
    """
    X is a complete dataset but its indices to be used have to be set through set_dataset_indices 
    If the function hand't been called upon fit, raise a warning 

    """
    complete_train_dataset = None
    def __init__(self, hyper_params_dict = None):
        super().__init__()

        self.output_features_ = 0
        self.indices  = None
        self.base_dataset = None
        if( hyper_params_dict is not None):
            self.hyper_params_dict = hyper_params_dict
        else:
            self.hyper_params_dict = {
                'gamma' : 10000,
            }
    
    def decide_label(self, row, labels , label):
        # print(f"row:{row}, labels:{labels}, label:{label}")
        idx = row["closest"]

        #print("*-----------------" , labels.index)
        #print( "Supervised closest label " , idx )
        return labels.loc[ idx ]
    
    def set_dataset_indices(self, indices):
        """
        Although inherited from estimator, the implementation will be on each leaf, we dont need to have 
        the entire dataset on each leaf memory. As the dataset indices are stored, the X won't need to be copied.
        Indices is an array or a pandas Index that can be transformed to an array 
        """
        self.indices = indices.to_numpy(dtype=numpy.dtype(int)) if type(indices) != list else numpy.array(indices, dtype=numpy.dtype(int) )

        
    def fit(self, X, y, tree_id=0,level=0, feature_subset = None):
        """
        X includes ONLY data , data frame 
        y includes only labels, dataframe with index as X
        """
        if( self.indices is None):
            print("WARNING: the indices are not set, then the indices to be set are going to be all the ones from X")
            if( type(X) is pandas.DataFrame):
                self.set_dataset_indices(X.index)
            else:
                indices = [i for i in range(0, X.shape[0]) ]
                X = pandas.DataFrame(data = X)
                self.set_dataset_indices(indices)
        self.output_features_ = y.shape[1]
        train_set = X.loc[self.indices, :]
        train_labels = y.loc[self.indices,:]
        # print(train_labels)
           
        
        self.base_dataset = X # just a view, reference
        self.columns = y.columns
        sup_amount = train_labels.shape[0] - train_labels[ train_labels[train_labels.columns[0]] == -1 ].shape[0]
        is_semisupervised = sup_amount >= 1 and sup_amount < train_labels.shape[0] # because of how the tree is formed, the first condition is true always, but the second evaluates that at least we have 1 unsupervised sample
        complete_labels = []
        
        if is_semisupervised:
            
            #print(train_set)
            #print(train_labels[ train_labels[train_labels.columns[0]] != -1 ].index)
            supervised = train_set.loc[train_labels[ train_labels[train_labels.columns[0]] != -1 ].index]
            supervised_labels = train_labels[ train_labels[train_labels.columns[0]] != -1 ]
            unsupervised = train_set.loc[train_labels[ train_labels[train_labels.columns[0]] == -1 ].index]
            unsupervised_labels = train_labels[ train_labels[train_labels.columns[0]] == -1 ]
            
            #if( feature_subset is not None):
            #    data = train_set[feature_subset].to_numpy()
            #else:
            
            data = train_set.to_numpy()
            
            #print("is semisupervised")
            for label in self.columns:
                

                #print(f"{tree_id} label " )
                values = numpy.unique(train_labels[train_labels[label] != -1][label].to_numpy())
                if(len(values) == 1):
                    # cant do no more.
                    # print(f"cant do more all labeels are the same tree {tree_id}")
                    complete_labels.append( numpy.ones(shape=train_labels[label].to_numpy().shape)*values[0] )
                    #print(f"{tree_id} full out {values[0]} :: " , train_labels[train_labels[label] != -1][label].to_numpy() )
                    #print(complete_labels)
                    continue
                #kernel =  rbf_kernel_safe # rbf_kernel_safe,linear_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
                #semisup_model = LabelSpreading(gamma=self.hyper_params_dict["gamma"], max_iter=60, alpha=self.hyper_params_dict["alpha"], kernel=kernel)
                # #--- semisup_model = LabelPropagation(gamma=200, max_iter=1000,kernel=rbf_kernel_safe, n_neighbors=12)
                #semisup_model.fit(data, train_labels[label].to_numpy() )
        
                # final_trainset = pandas.DataFrame(index=train_set.index)
                label_value = None
                np_labels = train_labels[label].to_numpy()
                supervised_np_labels = (np_labels>=0).nonzero()
                unsupervised_np_labels = (np_labels<0).nonzero()
                N_reps = self.hyper_params_dict['leaves_SS_N_reps']
                M_attributes = self.hyper_params_dict['leaves_SS_M_attributes']
                for i in range(0,N_reps):
                    attribute_subset = rng.choice(data.shape[1], min(M_attributes,data.shape[1]-1), replace=False) #should we check if they are correlated??
                    # instead of checking if thy are correlated. We could try to get the correlation average
                    # and multiply the 0 or 1 value of this label by 1-correlation average. 
                    # The final average is not over the total set of N_reps, but on the sum of the sum(1-correlation) 0<i<N_reps
                    # can a study over the colinearity of attributes on multivariate trees exist


                    # try nearest neighbors on attribute subsets, finally, label the example and train the final method. 
                    #print("attribute subset " , attribute_subset)
                    #print("supervised np labels " , list(supervised_np_labels[0]))

                    cat_feats = self.hyper_params_dict["categorical_dictionary"].get_categorical_features( attribute_subset)
                    
                    gower_dist = pairwise_distances(data[:,attribute_subset] ,metric= self.hyper_params_dict['distance_function'], cat_features=cat_feats )
                    gower_dist[:, unsupervised_np_labels ] = 100
                    
                    
                    #gower_dist = gower_dist[:, list(supervised_np_labels[0]) ] # just the supervised ones
                    #print("gower dist" , gower_dist)
                    
                    ordered_index= numpy.argsort( gower_dist ) 
                    
                    # return from numpy indices to pandas indices
                    
                    # i = 0
                    #print(gower_dist)
                    # this will tell the closes supervised sample
                    pandas_index = []
                    for row in ordered_index:
                        # some of them will have the min equals to themselves, they are the supervised ones, we dont care
                        pandas_index.append( train_set.index[row[0]] )
                        # print("train set index " , row , " --- " , train_set.index[row[0]] )
                        
                    
                    
                    #print("pandas index, " ,  pandas_index)
                    df= pandas.DataFrame( pandas_index ,index=train_set.index, columns=['closest'])
                    df["do_label"] = 0
                    df.loc[unsupervised.index, "do_label"] = 1
                    df[label] = train_labels[label]
                    current_supervised_label = supervised_labels[label]
                    
                    try:
                    #print(df)
                    #print(df.loc[supervised.index])
                        df.loc[unsupervised.index, label] = df.loc[unsupervised.index].apply(self.decide_label,axis=1,args=(current_supervised_label,label))
                    except Exception as e:
                        print("******************* This is the exception " , e)
                    
                        #for di in data[:,attribute_subset]:
                        #    print(di)
                        pandas.DataFrame( data[:,attribute_subset] ).to_csv("problematic.csv")
                        pandas.DataFrame( gower_dist ).to_csv("problematic_result.csv")
                        pandas.DataFrame( supervised.index ).to_csv("problematic_supervised_index.csv")
                        #print("-----------------")
                        #for gi in gower_dist:
                        #    print(gi)
                        raise Exception("",e)
                        #raise Exception("stpped," , e)
                        print("label" , label , "   i: " , i, "df   " , df)
                        print("       -----------    " , unsupervised.index, "  " , df.loc[unsupervised.index] )

                    if(label_value is None):
                        label_value = df[label]
                    else:
                        label_value = label_value + df[label]
                        
                
                label_value = label_value/N_reps
                #print(label_value)

                # at this point we could be more open to do a different threshold than 0.5 
                # maybe corresponding with the imbalance?
                complete_labels.append( label_value.to_numpy().round().astype(int)   )
                #print(complete_labels)
                # semisup_model.fit(data, train_labels[label].to_numpy() )
                
                # tyhe model transduction outputs all the assigned labels
                #complete_labels.append(semisup_model.transduction_) 
                # it can happen that after the process, the threshold did not make it enough, thats why the self training changed from threshold to k_best.... they all should
                # be classified!! The parameter, max inter and minimum leaft quantity isntances should be connected
                #print("calculated ",  complete_labels[-1] )
        else:
            for label in self.columns:
                complete_labels.append( train_labels[label].to_numpy() )

        #print(train_set)
        #print(train_labels)
        #print(pandas.DataFrame(complete_labels).T)
        
        if ( self.hyper_params_dict['output_tree_sets'] ):
            ttl = pandas.concat([train_set, train_labels, pandas.DataFrame(complete_labels).T.set_index(train_set.index)], axis=1 )
            ttl.to_csv(f"emotions_nodes_csvs/{tree_id}_{level}_{uuid.uuid4()}.csv")
            
        
        self.models = []
        self.thresholds = []
        i = 0
        # print(f"Node with final : {train_set.shape}")
        for label in self.columns:
            # check labels unique_values_count
            values = numpy.unique(complete_labels[i])
            if(len(values) == 1):
                self.models.append(values[0])
                self.thresholds.append((0.5,0.5))
            else:
                #self.models.append( LogisticRegression(C=self.hyper_params_dict["C"], solver='lbfgs', max_iter=3000) ) 
                self.models.append( RandomForestClassifier(
                    n_estimators = self.hyper_params_dict['leaves_RF_estimators']  , 
                    n_jobs=1 , 
                    max_depth= self.hyper_params_dict['leaves_max_depth'] , 
                    min_samples_split= self.hyper_params_dict['leaves_min_samples_split'] , 
                    min_samples_leaf= self.hyper_params_dict['leaves_min_samples_leaf'] , 
                    max_features= self.hyper_params_dict['leaves_max_features']
                    ) ) # , class_weight='balanced', took out, and got better results
                # self.models.append(  KNeighborsClassifier(n_neighbors=sup_amount ))
                #self.models[-1].fit( kernelized_train_set , complete_labels[i] )
                self.models[-1].fit( train_set.to_numpy() , complete_labels[i] )
                
                if( len(self.models[-1].classes_) > 2 ):
                    print("---------------------------------------------------------------------------- Error! in classes found")
                    # this c an happen if not every instance was labeled by one of the final models
                    print(self.models[-1].classes_)
                else:
                    # represent the imbalance towards no label, and label assignment
                    value_counts = numpy.bincount(complete_labels[i]) # they will be exactly two, bin count gets the count for 0's and 1's in the array at 0 and 1, yeah, amazing
                    total_counts = value_counts[0] + value_counts[1]
                    zero_compensation = value_counts[0]/total_counts # this is the imbalance of 0's
                    one_compensation = value_counts[1]/total_counts # this is the imbalance on 1's
                    
                         
                    # imbalance_compensation = (0.25 + 0.5*zero_compensation , 0.25 + 0.5*one_compensation )
                    imbalance_compensation = (0.95*zero_compensation , 0.95*one_compensation ) 
                    self.thresholds.append(imbalance_compensation)
            i+=1
        return complete_labels
        # print( "Thresholds" + str([m[1] for m in self.thresholds ]) )
        # return pandas.DataFrame(data=kernelized_train_set, columns=train_set.index, index=train_set.index ), to_print_df
        # no need for complete_labels structure anymore
    def predict(self, X):
        pred, prob =  self.predict_with_proba(X)
        return pred
    
    def predict_proba(self, X):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X)
        return prob
    
    def get_models_feature_importance(self):
        frs = []
        if ( self.models is not None and len(self.models) > 0 ):
            for i in self.models:
                
                if( type(i) is RandomForestClassifier):
                    forest_fi = i.feature_importances_ # there is a technique for permutation feature importance, less biased, not implemented
                    frs.append(  forest_fi )
                else:
                    frs.append(0)
        else:
            return [0 for i in self.columns] # not tested
        return frs

    def predict_with_proba(self, X, optional_global_labels_distribution = None, y_true = None):

        
        pred = numpy.zeros(shape=(X.shape[0],self.output_features_) )
        train_set = self.base_dataset.loc[self.indices, :]
        if( type(X) is pandas.DataFrame ):
            x = X.to_numpy()
        else:
            x  = X
        
        
        # the optimization here is to save some of these
        #kernelized_set = linear_kernel(x, train_set.to_numpy() )  
        # we were combining a rbf with a linear kernel

        #print("X " , x )
        # U = rbf_kernel(x,train_set.to_numpy() , gamma =self.hyper_params_dict["gamma"] )
        
        """ Not needed anymore
        A = euclidean_distances(x,train_set.to_numpy(), squared=True)
        A = A*-self.hyper_params_dict["gamma"]
        numpy.exp(A,A)
        kernelized_set = A
        kernelized_set = numpy.array([kernelized_set[0]]) # as it is the distance of x to the rest
        """
        
        #print(kernelized_set)
        i=0
        predictions = []
        probabilities = []
        for label in self.columns:
            if( type(self.models[i]) is KNeighborsClassifier ):
                prob = self.models[i].predict_proba(x)
                probabilities.append(prob[0][1])# supposing the first class is 0, just get prob of being 1
                p = 1 if prob[0][1] > self.thresholds[i][1] else 0 # first seat is reserved for 0 # it was 0.7 on best results
                # p = self.models[i].predict(kernelized_set)
                predictions.append(  int(p)  ) # prediction is the whole column!

            elif( type(self.models[i]) is RandomForestClassifier ):
                #prob = self.models[i].predict_proba(kernelized_set)
                try:
                    prob = self.models[i].predict_proba(x)
                except:
                    # this happens when the model is not able to return a probability due to 
                    # unknown values in categorical columns. Default to full probability on 0. No label assigned. 
                    prob = numpy.array([1,0]).reshape(1,-1)
                # print(self.models[i].classes_ , "Class assign  :" , prob , " compensation " , self.thresholds[i] )
                """
                if(len(prob[0]) > 1 ):
                    print(self.models[i].n_features_in_ )
                    print( self.models[i].classes_ )
                else:
                    print( self.models[i].classes_ )
                """
                # recall that prob is an  array of two class predictions
                if( optional_global_labels_distribution is not None and False): # it didnt worked ...
                    probabilities.append( (prob[0][1]+optional_global_labels_distribution[i])*0.5 )
                else:
                    #probabilities.append( (prob[0][1]+self.thresholds[i][1])*0.5  )# supposing the first class is 0, just get prob of being 1
                    probabilities.append( prob[0][1]  )# supposing the first class is 0, just get prob of being 1
                
                #p = 0 if prob[0][0] > self.thresholds[i][0] else 1 # first seat is reserved for 0 # it was 0.7 on best results
                # it is clearer this way. This one counts for the confusion matrix.
                # also, this is a way to balance skewed ones and zeros distributions 
                # remember that the thresholds are the amount of ones divided by the nodes instances
                # If threshold is low, it should be "easier" to give a 1.
                
                thres = self.thresholds[i][1]
                #thres = 1

                if( optional_global_labels_distribution is not None and False):
                    thres =  (thres + optional_global_labels_distribution[i])*0.5 # a simple average between local and global

                p = 1 if prob[0][1] > thres else 0 # first seat is reserved for 0 # it was 0.7 on best results
                
                # p = 1 if (prob[0][1]+optional_global_labels_distribution[i])*0.5 > thres else 0 
                # p = 1 if prob[0][1] > 0.5 else 0 # first seat is reserved for 0 # it was 0.7 on best results
                   
                # p = self.models[i].predict(kernelized_set)
                predictions.append(  int(p)  ) # prediction is the whole column!
            else:
                predictions.append(self.models[i]) # as it is a constant
                probabilities.append(self.models[i])
            i+=1
        #print("predictions, will have to change them for numpy array, and eventually predict_proba")
        # print(predictions)
        pred = numpy.array(predictions).T
        probs = numpy.array(probabilities).T
        #print(pred)
        return pred,probs