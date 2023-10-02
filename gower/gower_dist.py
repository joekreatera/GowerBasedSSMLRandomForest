from scipy.sparse import issparse
import numpy as np
import pandas as pd
import timeit
# min_max_array is added by joe 
# to be able to predict using this function. The max and mins are obtained on runtime
# the min_max_array has the mins in the first row and max in the last row.... 
# Should be a numpy array that overrides tge values. Categorical should have 0 on both 



def gower_matrix(data_x, data_y=None, weight=None, cat_features=None, casting='same_kind', min_max_array = None):  
    
    # function checks
    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 
    if not isinstance(X, np.ndarray): 
        if not np.array_equal(X.columns, Y.columns): raise TypeError("X and Y must have same columns!")   
    else: 
         if not X.shape[1] == Y.shape[1]: raise TypeError("X and Y must have same y-dim!")  
                
    if issparse(X) or issparse(Y): raise TypeError("Sparse matrices are not supported!")        
            
    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    if cat_features is None:
        if not isinstance(X, np.ndarray): 
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)    
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col]=True
    else:          
        cat_features = np.array(cat_features)
    
    # print(cat_features)
    
    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)
    
    Z = np.concatenate((X,Y))
    
    x_index = range(0,x_n_rows)
    y_index = range(x_n_rows,x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    
    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)
    
    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32) 
        max = np.nanmax(col_array)
        min = np.nanmin(col_array)
     
        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        
        num_max[col] = max
        num_ranges[col] = np.abs(1 - min / max) if (max != 0) else 0.0
        
        if(min_max_array is not None):
            num_max[col] = min_max_array[1,col]
            num_ranges[col] = np.abs(1 - min_max_array[0,col] / min_max_array[1,col] ) if (min_max_array[1,col] != 0) else 0.0
             
    # print(num_ranges)
    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(Z_num ,num_max,out=np.zeros_like(Z_num), where=num_max!=0, casting=casting)
    Z_cat = Z[:,cat_features]
    
    if weight is None:
        weight_cat = None
        weight_num= None   
        weight_sum = None
    else:
        weight = np.ones(Z.shape[1])
        weight_cat=weight[cat_features]
        weight_num=weight[np.logical_not(cat_features)]   
        weight_sum = weight.sum()
        
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)
        
    
    X_cat = Z_cat[x_index,]
    X_num = Z_num[x_index,]
    Y_cat = Z_cat[y_index,]
    Y_num = Z_num[y_index,]


    
    # print(X_cat,X_num,Y_cat,Y_num)
    for i in range(x_n_rows):       
        #print(i)   
        j_start= i        
        if x_n_rows != y_n_rows:
            j_start = 0
        # call the main function
        res = gower_get(X_cat[i,:], 
                          X_num[i,:],
                          Y_cat[j_start:y_n_rows,:],
                          Y_num[j_start:y_n_rows,:],
                          cat_features,
                          num_ranges,
                          num_max,
                          weight_cat,
                          weight_num,
                          weight_sum,
                          casting=casting) 
        #print(res)
        out[i,j_start:]=res
        if x_n_rows == y_n_rows: out[i:,j_start]=res
        #print(out)
    return out

# @jit(nopython=True, cache=True)
def gower_get(xi_cat,xi_num,xj_cat,xj_num,
              categorical_features,
              ranges_of_numeric,max_of_numeric, 
              feature_weight_cat = None,feature_weight_num= None,feature_weight_sum= None,
              casting='same_kind' ):
    

    #return np.zeros( xj_num.shape[0] )
    #print(xi_num)
    #print(xj_num)
    #print("----------------------------")
    # categorical columns

    sij_cat = np.where(xi_cat == xj_cat,np.zeros_like(xi_cat),np.ones_like(xi_cat))
    
    if( feature_weight_sum is not None):
        sum_cat = np.multiply(feature_weight_cat,sij_cat).sum(axis=1) 
    else:
        sum_cat = sij_cat.sum(axis=1)
    # print(xi_cat , " :: "  , sij_cat , " --- " , sum_cat)    
    # numerical columns

    # t = timeit.default_timer()
   
    to_abs = np.zeros_like(xj_num)
    #t1 = timeit.default_timer() - t
    #xi_numB = xi_num-xj_num
    np.subtract(xi_num,xj_num, out=to_abs)
    #xi_numB = xi_num - xj_num
    #print((xi_num-xi_numB))
    abs_delta=np.absolute(to_abs)
    #t2 = timeit.default_timer() - (t1 +t)
    
    sij_num = np.zeros_like(abs_delta)
    np.divide(abs_delta, ranges_of_numeric, out=sij_num, where=ranges_of_numeric!=0, casting=casting)

    #t3 = timeit.default_timer() - (t2 + t1 + t)

    
    if( feature_weight_sum is not None):
        sum_num = np.multiply(feature_weight_num,sij_num).sum(axis=1)
    else:
        sum_num = sij_num.sum(axis=1)
    
    sums= np.add(sum_cat,sum_num)
    #print(">>> " , sum_cat , "\n " , sums)
    if( feature_weight_sum is not None):
        sum_sij = np.divide(sums , feature_weight_sum  )
    else:
        sum_sij = np.divide(sums, sij_cat.shape[1] + sij_num.shape[1]) # all with equal weight

    #t4 = timeit.default_timer() - (t3 + t2 + t1 + t)


    #total = timeit.default_timer() - t
    #print(f"{t} \t {t1/total} \t {t2/total} \t {t3/total} \t {t4/total}  {total}" )
    return sum_sij

def smallest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    #n += 1
    flat = np.nan_to_num(ary.flatten(), nan=999)
    indices = np.argpartition(-flat, -n)[-n:]
    indices = indices[np.argsort(flat[indices])]
    #indices = np.delete(indices,0,0)
    values = flat[indices]
    return {'index': indices, 'values': values}

def gower_topn(data_x, data_y=None, weight=None, cat_features=None, n = 5):
    
    if data_x.shape[0] >= 2: TypeError("Only support `data_x` of 1 row. ")  
    dm = gower_matrix(data_x, data_y, weight, cat_features)
          
    return smallest_indices(np.nan_to_num(dm[0], nan=1),n)
