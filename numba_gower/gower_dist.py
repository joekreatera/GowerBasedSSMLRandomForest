import numpy
# from fastdist import fastdist
import pandas
from numba import jit,njit, vectorize
from numba.typed import List

"""
Reimplementation with numba. 
requires numba and works better with specialized frameworks such as intel OneAPI 

THis isnot prepared for sparse matrices, yet. 
Requires an array static the categorical features. 0 for numerical ones, 1 for categorical. 

Needs categorical columns as np.uint8 (assuming this is enough for different categorical values)



Based on equations from: https://arxiv.org/ftp/arxiv/papers/2101/2101.02481.pdf

Otyher packages normalize data, but there is no need to do that as it has a normalization
procedure due to the radius (max-min) that is performed when obtaining the distance on numerical values. 
"""




# we should cover two cases, the distance between the full matrix, and the distance from vector to matrix
@njit(fastmath=True, error_model='numpy') # error model comes from avoid implicit branching on divisions!
def gower_distance_matrix(X,cat_cols , mn=None, mx = None):
    """
    Calculates de distance from every x_i in X to every other x_j in X. It generates the operations for (x_rows)(x_rows-1)/2 and copies the values to the rest of the matrix 

    cat_cols should be a numba typed List. For this reasoin when there are no cat_cols, cat cols should be = List([-1]) as there is no index -1 on numpy arrays

    """
    # print("CAT COLS*************** " , cat_cols)
    
    m,n = X.shape
    div  = numpy.zeros(n)
    div_calc = numpy.zeros(n, dtype=numpy.int8) # needed to check which columns have already been computed. 
    do_min_max = mn is None

    dm = numpy.zeros( shape=(m,m) )
    k = 0
    for i in range(0,m-1) :   
        x = X[i,:]
        M = X[i+1:,:]
        res = numpy.zeros( M.shape[0] )
        cat_init = False
        
        for col in range(X.shape[1]):
            if(div_calc[col]  == 0 ):
                if( col not in cat_cols ):
                    if do_min_max:
                        nmn = numpy.nanmin(X[:,col])
                        nmx = numpy.nanmax(X[:,col])
                    else:
                        nmn = mn[col]
                        nmx = mx[col] 
                    
                    # div[col] = numpy.abs(1 - nmn/nmx ) if( nmx != 0 ) else 0
                    div[col] = (nmx-nmn)+0.000000000001
                    div_calc[col] = 1
                    # X[:,col] = X[:,col]/nmx if( nmx != 0 ) else X[:,col]
                    
                    # print(f"Col {col}  " , div[col])
                    # x[col] = x[col]/mx if( mx != 0 ) else x[col]
                    
            if( not  cat_init ):
                cat_init = True
                res += len(cat_cols)/n # initialize the categorical base value
                div_calc[col] = 1
                
            if( cat_cols[0] > -1 and col  in cat_cols ):
                res -=   ( ( M[:,col].astype(numpy.int8) == int(x[col])  ).astype(numpy.int8) )/n
            else: 
                # print(f"Col {col}" , div[col])
                # a =  numpy.absolute(x[col]-M[:,col])/div[col]/n
                res += (numpy.abs(x[col]-M[:,col])/div[col] )/n
                #print( x[col]-M[:,col] )
               
        dm[i,i+1:] = res
        dm[i+1:,i] = res
    #    k = k+1
    return dm


@njit(fastmath=True, debug=True, error_model='numpy') # error model comes from avoid implicit branching on divisions!
def gower_distance_vector_to_matrix(x , X,cat_cols , mn=None, mx = None):
    """
    Returns the distance from x to each X not including the distance from x to itself, which is 0. 

    cat_cols should be a numba typed List. For this reasoin when there are no cat_cols, cat cols should be = List([-1]) as there is no index -1 on numpy arrays
    """

    m,n = X.shape
    div  = numpy.zeros(n, dtype=numpy.float32)
    div_calc = numpy.zeros(n, dtype=numpy.int8) # needed to check which columns have already been computed. 
    do_min_max = mn is None
    
    M = X
    res = numpy.zeros( M.shape[0], dtype=numpy.float32 )
    
    cat_init = False
    #print("1 this is insider woeert distance vector to matrix " , x)
    #print("2 this is insider woeert distance vector to matrix " , M)
    
    for col in range(X.shape[1]):
        #print(f" ********* Cycling cols " , col)
        #print(f" ********* Div calc " , div_calc[col] )
        #print(f" ********* check calc " , (col not in cat_cols) )
       
        
        
        if(div_calc[col]  == 0 ):
            if( col not in cat_cols ):
                if do_min_max:
                    nmn = numpy.nanmin(X[:,col])
                    nmx = numpy.nanmax(X[:,col])
                else:
                    nmn = mn[col]
                    nmx = mx[col] 
                
                # div[col] = numpy.abs(1 - nmn/nmx ) if( nmx != 0 ) else 0
                div[col] = (nmx-nmn)+ numpy.float32(0.000000000001) 
                div_calc[col] = 1

        #print("-******* Cat init " , cat_init )
        #print(f" ********* Div Afters calc " , div_calc[col] )
        
        if( not  cat_init ):
            cat_init = True
            res += len(cat_cols)/n # initialize the categorical base value
            # div_calc[col] = 1 # doesnt matter for cat cols
        
        #print(f" ********* Cat cols [0]  Afters calc " ,  (cat_cols[0] > -1) )

        if( cat_cols[0] > -1 and col  in cat_cols ): # we should convert the different columns to int values.
            # print(" -*-*-*-*-*-*-**-*-*--*-*-*-* " ,  cat_cols[0]  )
            # the problem is that x[col] 
            a1 = M[:,col] # .astype(numpy.int8)
            # print(x , " -> " ,  x.shape)
            a2 = x[0,col] #.astype(numpy.int8)
            a3 = ( a1  == a2  ).astype(numpy.int8)
            res -=   ( a3 )/n
        else:
            
            #res +=  numpy.absolute(x[col]-M[:,col])/div[col]/n
            res += (numpy.abs(x[0,col]-M[:,col])/div[col] )/n
            

    #print("Result of the function of distance " , res)
    return res.reshape(1,-1)