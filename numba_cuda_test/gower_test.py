import pandas as pd
import numpy as np
import timeit
import tracemalloc
from numba import cuda
from numba_gower import gower_dist as gd
from numba.typed import List
from math import sqrt, floor


df  = pd.read_csv("mediamill.csv")
label_cols = [f"label_{i}" for i in range (0,101)]
df = df[df.columns.difference(label_cols)]
print(df)
print(cuda.devices)
print(cuda.is_available())
print(cuda.detect())
total_rows = 4096
threads_per_block = int(sqrt(512)) # cause it is an area... top is 1024
A = np.random.random( (total_rows,20) ).astype(np.float32)
D_cuda = cuda.device_array((total_rows,total_rows), dtype=A.dtype)

#print(A)



A_cuda = cuda.to_device( A )
block_square_size = int(total_rows/threads_per_block)
blocks = int((block_square_size+1)*(block_square_size)/2)

print("Host to device completed")


n = block_square_size

cols = A.shape[0]
@cuda.jit
def dist(instances, distances):
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y
    k = cuda.blockIdx.x
    
    row = n -1 - floor((-1+sqrt((2*n+1)**2 - 8*(k+1)))/2)
    col = k-row*(2*n-row-1)//2
    # from https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    # https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
    # https://dongkwan-kim.github.io/blogs/indices-for-the-upper-triangle-matrix/

    to_decrement = cuda.blockIdx.x//block_square_size 
    row1= cuda.blockIdx.x//(block_square_size-to_decrement)
    col1= row1 + cuda.blockIdx.x%block_square_size # obtaining x and y from just x..We dont need N*N calculations

    c = int(col)*threads_per_block+idx
    r = int(row)*threads_per_block+idy
    A = instances[c]
    B = instances[r]
    
    instance_sum = 0.0
    for i in range(0, cols):
        instance_sum += abs(A[i]-B[i])
    # next step: get the instances and calculate the manhattan distance
    distances[r,c] = instance_sum
    cuda.syncthreads()

# max amount of threads is 1024, 
# bidimensionally is 32 x 32
t = timeit.default_timer()
print("starting calc")
dist[ blocks  , (threads_per_block,threads_per_block) ](A_cuda, D_cuda)
cuda.synchronize()
print("GD cuda " , (timeit.default_timer() - t ))

t = timeit.default_timer()
dist[ blocks  , (threads_per_block,threads_per_block) ](A_cuda, D_cuda)
cuda.synchronize()
print("GD cuda " , (timeit.default_timer() - t ))

t = timeit.default_timer()
D = D_cuda[0:100,0:100].copy_to_host()
print("GD cuda device to host " , (timeit.default_timer() - t ))

cuda.close()
# print(D)
#pd.DataFrame(D).to_csv("res.csv", index=None, header=None)




cats = List([-1])

t = timeit.default_timer()
df_np = df.loc[0:100,:].to_numpy() # just for compilation 
gm = gd.gower_distance_matrix(df_np[:,0:20] , cat_cols=cats )
print("GD numba " , (timeit.default_timer() - t ))
#print(gm)

t = timeit.default_timer()
df_np = df.loc[0:total_rows,:].to_numpy()
gm = gd.gower_distance_matrix(df_np[:,0:20] , cat_cols=cats )
print("GD numba " , (timeit.default_timer() - t ))
print(gm)
