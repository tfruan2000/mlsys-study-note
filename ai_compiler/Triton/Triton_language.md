# [triton language](https://triton-lang.org/main/python-api/triton.language.html)

理解triton语法的repo：[triton-puzzles](https://github.com/srush/Triton-Puzzles)

[🌰: vector-add](./Triton_base.md##elements)

```python
import torch
import triton
import triton.language as tl
```

## detector

- triton.autotune：自动调优detector，用于自动找到最佳配置

使用上需要提供一个configs（包含在kernel中定义的 `tl.constexpr`）列表，autotune会多次运行kernel函数来评估configs中的所有配置。（配置是人为给出的，所以空间不大，依赖人为经验）

- triton.heuristics：启发式detector，根据输入参数动态调整 kernel 的行为

例如，如果B（偏置）不为None，则HAS_BIAS为真

```python
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
```

## tl.constexpr

超参数，对于不同的硬件使用时，最佳性能的参数可能是不同的，其值由 Triton Compiler 进行搜索，会人为给一个由 **`@auto-tuning` 标记的 `configs`**（依赖人为经验）。

### [tl.load](https://triton-lang.org/main/python-api/generated/triton.language.load.html#triton.language.load) & [tl.store](https://triton-lang.org/main/python-api/generated/triton.language.store.html#triton.language.store)



## Compiler Hint Ops

### tl.max_contiguous & tl.multiple_of

```python
  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
```

由于编译器无法感知数据的连续性，所以加载数据时会**离散地**处理数据。
如果编写kernel时提前已知数据连续，可以使用 `tl.max_contiguous & tl.multiple_of` 去标识加载数据的连续性，这样编译器就可连续地处理该段数据。

- max_contiguous(input, values)：标识input中前values个元素为连续

- multiple_of(input, values)：标识input中的元素是values的倍数

```python
  offs_am = tl.max_contiguous(tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
  offs_am = tl.max_contiguous(tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
```

### tl.max_constany

max_constany(input, values)：标识input中前values个元素为常量
