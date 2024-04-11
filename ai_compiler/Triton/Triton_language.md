# triton language

## 🌰：add

```python
import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

1. `@triton.jit` 装饰器decorator，表示下面这段代码是一个triton kernel
2. `x_ptr, y_ptr` 指针，为其代表的tensor的第一个元素的地址。用来将数据load到memory
3. 输入中一般也有stride，对于n维的tensor a，a.stride()会输出一个n维数组。stride用来找每个元素的指针

```python
a = torch.rand([3,6])
a.stride() # (6, 1)
# 这里的第一个维度的 stride 是 6, 因为从 a[m, k] 的地址 到 a[m+1, k] 的地址,
```

1. 超参数 `tl.constexptr` ，对于不同的硬件使用时，最佳性能的参数可能是不同的，后续由 Triton compiler 来进行搜索不同的值
2. 虚拟循环 `pid = tl.program_id(axis=0)` ，每个kernel可能被执行多次
    1. program_id是这个虚拟的 for "循环" 里面的 index (第几次循环，实际中这些循环是并行)
    2. `axis` , 是说明 "循环"有几层，此处 axis = 0表示展开为1维来访问（维度概念类比memref的维度，第一维相当于memref的最内维u）
    
    ```python
    pid = tl.program_id(axis=0)
    # 当访问数据总长256, BLOCK_SIZE=64
    # tl.arange(0, BLOCK_SIZE) -> [0, 63]
    # 0， 64， 128， 192
    block_start = pid * BLOCK_SIZE
    # 所以数据访问时是按照 [0:64, 64:128, 128:192, 192:256]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    ```
    
    > axis是启动3d Grid的索引，必须是0 / 1 / 2

    c. 调用kernel时，需要说明该kernel执行循环有几层，每层有几次，这就是 `grid` 的概念
    
3. 显示地load和store，批量数据处理，一次处理一个BLOCK_SIZE的数据，SIMD行为

```python
    # load 和 store 时都是使用基地址加偏移 获得一片数据，mask表示只获得这片数据中的一部分
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 写回时也需要mask
    tl.store(output_ptr + offsets, output, mask=mask)
```


## num_warp

一般体现在module Attr上

```python
"triton_gpu.num-warps" = 4 : i32
```

tritongpu ir相比ttir仅多了一个Blocked Layout，本质上描述的是Block对Memory的Access Pattern

```python
 #blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

就是一个Block里有4个Warp，一个Warp有32个Thread，一个Thread处理1个元素。

Blocked Layout只是一种Pattern，但按照这个Pattern会多次访问，总访问量达到BLOCK_SIZE

## tl.max_contiguous

```python
  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
```

由于编译器无法感知数据的连续性，所以加载数据时会**离散地**处理数据。
如果编写kernel时提前已知数据连续，可以使用 `tl.max_contiguous & tl.multiple_of` 去标识加载数据的连续性，这样编译器就可连续地处理该段数据。

```python
  offs_am = tl.max_contiguous(tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
  offs_am = tl.max_contiguous(tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
```

## elementwise op