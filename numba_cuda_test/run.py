from numba import cuda,njit
from math import sqrt, sin, exp
import numpy as np
import timeit
import tracemalloc
from numba.types import float32


N = 200_000_000
A = np.random.random(N ).astype(np.float32)
B = np.random.random(N ).astype(np.float32)
nelem = 1024 # len(A)
@njit(fastmath=True)
def f2(a,b):
    return np.exp( np.sin(np.sqrt(a + b)) )

@njit(fastmath=True)
def sum_all(a):
    s = 0
    for i in a:
        s += i
    
    return s


"""
The index of a thread and its thread ID relate to each other in a straightforward way: 
For a one-dimensional block, they are the same; for a two-dimensional block of size (Dx, Dy), 
the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of 
size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).
"""

threads_per_block = 1024  # Why not!
blocks_per_grid = 32 * 100  # Use 32 * multiple of streaming multiprocessors

# Example 2.1: Naive reduction
@cuda.jit
def reduce_better(array, partial_reduction):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    s_thread = 0.0
    for i_arr in range(i_start, array.size, threads_per_grid):
        s_thread += array[i_arr]

    # We need to create a special *shared* array which will be able to be read
    # from and written to by every thread in the block. Each block will have its
    # own shared array. See the warning below!
    s_block = cuda.shared.array((threads_per_block,), float32)
    
    # We now store the local temporary sum of the thread into the shared array.
    # Since the shared array is sized threads_per_block == blockDim.x,
    # we should index it with `threadIdx.x`.
    tid = cuda.threadIdx.x
    s_block[tid] = s_thread
    
    # The next line synchronizes the threads in a block. It ensures that after
    # that line, all values have been written to `s_block`.
    cuda.syncthreads()

    i = cuda.blockDim.x // 2
    while (i > 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2

    if tid == 0:
        partial_reduction[cuda.blockIdx.x] = s_block[0]


@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = exp(sin(sqrt(a[tid] + b[tid])))


tracemalloc.start()

t = timeit.default_timer()

a = cuda.to_device(A)
b = cuda.to_device(B)
#c = cuda.device_array_like(a)

f.forall(len(a))(a, b, a) # the first one is to compile

t = timeit.default_timer()
f.forall(len(a))(a, b, a)
# func[griddim, blockdim, stream, sharedmem](x, y, z)
dev_partial_reduction = cuda.device_array((blocks_per_grid,), dtype=a.dtype)

reduce_better[blocks_per_grid, threads_per_block](a, dev_partial_reduction)
dpr = dev_partial_reduction.copy_to_host()
print(dpr.shape)
print(dpr.sum())


f.forall(len(a))(a, b, a)
#array_sum[1, nelem](c)
f.forall(len(a))(a, b, a)
#array_sum[1, nelem](c)
#print(c[0])
print( "Time gpu" , (timeit.default_timer() -t ) )
a,b = tracemalloc.get_traced_memory()
print( "GPU c:" , (a/1e6) )


t = timeit.default_timer()

C = np.exp( np.sin(np.sqrt(A + B)) )
g = C.sum()
A = np.exp( np.sin(np.sqrt(C + B)) )
g = A.sum()
C = np.exp( np.sin(np.sqrt(A + B)) )
g = C.sum()
print(g)
print( "Time Numpy" , (timeit.default_timer() -t ) )
a,b = tracemalloc.get_traced_memory()
print( "Numpy c:" , (a/1e6) )



t = timeit.default_timer()
A = np.random.random(N)
B = np.random.random(N)
C = f2(A,B) # this rrun is just to compile
d = sum_all(C)
t = timeit.default_timer()
C = f2(A,B)
A = f2(B,C)
C = f2(A,B)
d = sum_all(C)
print( "Time Numba" , (timeit.default_timer() -t ) )
a,b = tracemalloc.get_traced_memory()
print( "Numba c:" , (a/1e6) )
tracemalloc.stop()
