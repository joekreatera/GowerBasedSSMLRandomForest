import numpy
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
    
import warnings    


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
                'alpha' : 0.1,
                'C' : 0.01,
                'gamma' : 10000,
            }
    
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
        is_semisupervised = sup_amount >= 1
        complete_labels = []
        
        if is_semisupervised:
            
            supervised = X.loc[train_labels[ train_labels[train_labels.columns[0]] != -1 ].index]
            supervised_labels = train_labels[ train_labels[train_labels.columns[0]] != -1 ]
            unsupervised = X.loc[train_labels[ train_labels[train_labels.columns[0]] == -1 ].index]
            unsupervised_labels = train_labels[ train_labels[train_labels.columns[0]] == -1 ]
            
            #if( feature_subset is not None):
            #    data = train_set[feature_subset].to_numpy()
            #else:
            
            data = train_set.to_numpy()
            
            #print("is semisupervised")
            for label in self.columns:
                
                values = numpy.unique(train_labels[train_labels[label] != -1][label].to_numpy())
                if(len(values) == 1):
                    # cant do no more
                    complete_labels.append( numpy.ones(shape=train_labels[label].to_numpy().shape)*values[0] )
                    #print("full out " , values[0])
                    #print(complete_labels[-1])
                    continue
                #kernel =  rbf_kernel_safe # rbf_kernel_safe,linear_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
                #semisup_model = LabelSpreading(gamma=self.hyper_params_dict["gamma"], max_iter=60, alpha=self.hyper_params_dict["alpha"], kernel=kernel)
                # #--- semisup_model = LabelPropagation(gamma=200, max_iter=1000,kernel=rbf_kernel_safe, n_neighbors=12)
                #semisup_model.fit(data, train_labels[label].to_numpy() )
        
                final_trainset = pandas.DataFrame(index=train_set.index)
        

                sup_model = RandomForestClassifier(n_estimators = 7 , n_jobs=1) # class_weight='balanced'
                #sup_model = DecisionTreeClassifier(criterion='gini', min_samples_split =3)
                
                #sup_model = SGDClassifier(loss='modified_huber', alpha=0.001)
                
                # sup_model= KNeighborsClassifier(n_neighbors=sup_amount )
                #sup_model = SVC(kernel='rbf', probability=True, gamma=1, C=1)
                # semisup_model = SelfTrainingClassifier(sup_model, max_iter=200, criterion='k_best', k_best=10)
                #semisup_model = SelfTrainingClassifier(sup_model, max_iter=60)
                
                
                
                this_unsupervised_labels = unsupervised_labels[label].index

                sup_model.fit(supervised, supervised_labels[label].to_numpy().ravel() )
                predictions = sup_model.predict(unsupervised)
                
                final_trainset.loc[this_unsupervised_labels , label] = predictions.ravel()
                final_trainset.loc[supervised.index, label] = supervised_labels[label]
                
                
                complete_labels.append(final_trainset[label].to_numpy().ravel().astype(int) )
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
        
        ttl = pandas.concat([train_set, train_labels, pandas.DataFrame(complete_labels).T.set_index(train_set.index)], axis=1 )
        ttl.to_csv(f"emotions_nodes_csvs/{tree_id}_{level}_{uuid.uuid4()}.csv")
        
        
        """
        to_print_orig_label= train_labels[self.columns[0]].to_numpy().reshape(1,-1)
        #print(to_print_orig_label)
        #print(numpy.array(complete_labels))
        to_print_labels = numpy.concatenate( [numpy.array(complete_labels), to_print_orig_label]  )# labels 
        to_print_index = train_labels[self.columns[0]].index
        #print(to_print_index)
        to_print_original_set = SSLearnerLeaf.complete_train_dataset.loc[to_print_index]
        #print(to_print_original_set)
        #print(to_print_original_set[self.columns].to_numpy().T)
        
        to_print_np = numpy.concatenate( [to_print_original_set[self.columns].to_numpy().T, to_print_labels] ).T
        #print(to_print_np)
        to_print_df_columns = list(self.columns.array) + list(map(lambda a : a + '_pred', list(self.columns.array))) + ['unsup']
        
        to_print_df = pandas.DataFrame(data = to_print_np, columns=to_print_df_columns, index = to_print_index  )
        to_print_random_name = int(random.random()*1000)
        """
        #to_print_df.to_csv(f"learner_{to_print_random_name}.csv")
        
         
        #  linear_kernel , laplacian_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
        # kernelized_train_set = rbf_kernel( X = train_set.to_numpy() , gamma = 1000) became too too positive.
        
        # this transformation was obtained from manifold regularization paper, Semi-supervised multi-label classification using an extendedgraph-based manifold regularization
        # print(f"Doing with gamma: " , self.hyper_params_dict["gamma"])
        # U = rbf_kernel(X=train_set.to_numpy() , gamma = self.hyper_params_dict["gamma"])
        

        """ Not needed
        A = euclidean_distances(X=train_set.to_numpy(), squared=True)
        A = A*(-self.hyper_params_dict["gamma"])
        #print(A)
        numpy.exp(A,A)
        #print(A)
        U = A
        kernelized_train_set = U
        """
        
        #print("------------------------------2")
        #print(complete_labels)
        # print("Kernelized:::: -> \n" , kernelized_train_set)
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
                self.models.append( RandomForestClassifier(n_estimators = 5, n_jobs=1, class_weight='balanced') )
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
        
        # return pandas.DataFrame(data=kernelized_train_set, columns=train_set.index, index=train_set.index ), to_print_df
        # no need for complete_labels structure anymore
    def predict(self, X):
        pred, prob =  self.predict_with_proba(X)
        return pred
    
    def predict_proba(self, X):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X)
        return prob
        
    def predict_with_proba(self, X):
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
            if( type(self.models[i]) is RandomForestClassifier ):
                #prob = self.models[i].predict_proba(kernelized_set)
                prob = self.models[i].predict_proba(x)
                
                # print(self.models[i].classes_ , "Class assign  :" , prob , " compensation " , self.thresholds[i] )
                """
                if(len(prob[0]) > 1 ):
                    print(self.models[i].n_features_in_ )
                    print( self.models[i].classes_ )
                else:
                    print( self.models[i].classes_ )
                """
                # recall that prob is an  array of two class predictions
                probabilities.append(prob[0][1])# supposing the first class is 0, just get prob of being 1
                p = 0 if prob[0][0] > self.thresholds[i][0] else 1 # first seat is reserved for 0 # it was 0.7 on best results
                    
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