import cupy as cp
import timeit
from cupyx.profiler import benchmark
import pandas as pd
import numpy as np

def f(a,b):
    # exp, sin
    return cp.sum(cp.exp(cp.sin(cp.sqrt(a+b))))

t = timeit.default_timer()

"""

x = cp.random.rand(20_000_000).reshape(-1,2)

b = benchmark(f, (x[:,0] , x[:,1]), n_repeat=3)
print("Time  " , (timeit.default_timer() - t))
print(b , " > ")
"""

def gower_test(X,cat_cols , mn=None, mx = None):
    m,n = X.shape
    div  = cp.zeros(n)
    div_calc = cp.zeros(n, dtype=cp.int8) # needed to check which columns have already been computed. 
    do_min_max = mn is None

    dm = cp.zeros( shape=(m,m) )
    k = 0
    for i in range(0,m-1) :   
        x = X[i,:]
        M = X[i+1:,:]
        res = cp.zeros( M.shape[0] )
        cat_init = False
        
        for col in range(X.shape[1]):
            if(div_calc[col]  == 0 ):
                if( col not in cat_cols ):
                    if do_min_max:
                        nmn = cp.min(X[:,col])
                        nmx = cp.max(X[:,col])
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
                res -=   ( ( M[:,col].astype(cp.int8) == int(x[col])  ).astype(cp.int8) )/n
            else: 
                # print(f"Col {col}" , div[col])
                # a =  numpy.absolute(x[col]-M[:,col])/div[col]/n
                res += (cp.abs(x[col]-M[:,col])/div[col] )/n
                #print( x[col]-M[:,col] )
               
        dm[i,i+1:] = res
        dm[i+1:,i] = res
    #    k = k+1
    return dm

"""

It is too slow, there is no guarantee that a server will have an advantage over the multiprocessed memmory shared version. 



"""
df  = pd.read_csv("mediamill.csv")
label_cols = [f"label_{i}" for i in range (0,101)]
df = df[df.columns.difference(label_cols)]
print(df)
print(cp.cuda.is_available())

total_rows = 10_000

df_cp  = cp.asarray(df.loc[0:total_rows,:].to_numpy())
print(df_cp.shape)
A = gower_test(df_cp , cp.array([-1]))
print(A)
#A = np.random.random( (total_rows,20) ).astype(np.float32)
#D_cuda = cuda.device_array((total_rows,total_rows), dtype=A.dtype)
