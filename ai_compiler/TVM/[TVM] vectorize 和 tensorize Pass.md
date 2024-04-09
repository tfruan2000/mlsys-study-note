# [TVM] vectorize 和 tensorize Pass

## Pass含义

### 1、tvm中的Pass

> [https://tvm.apache.org/docs/arch/index.html](https://tvm.apache.org/docs/arch/index.html)
> 
> 
> [https://tvm.apache.org/docs/arch/pass_infra.html](https://tvm.apache.org/docs/arch/pass_infra.html)
> 
> [https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_infra.html](https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_infra.html)
> 
> [https://tvm.apache.org/docs/reference/api/python/ir.html?#tvm.instrument.PassInstrument](https://tvm.apache.org/docs/reference/api/python/ir.html?#tvm.instrument.PassInstrument)
> 
> [https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_instrument.html](https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_instrument.html)
> 

tvm的结构

<div style="text-align: center;"><img src="./img_vectorize 和 tensorize Pass/image-20230329133901335.png" alt="image-20230329133901335.png" style="width: 90%;"></div>

- `IRModule`：它是functions的集合，其中包含两种最关键的Function集合，即`relay::Function`和`tir::PrimFuc` 。
- 上层`relay::Function`继承自`BaseFunction`，`relay::Function`对应一个end2end的模型，可以理解为一个支持控制流，递归，以及复杂数据结构的计算图。
- 下层`tir::PrimFunc`也继承自`BaseFunction`，`tir::PrimFunc`包含了一些底层threading，vector/tensor的指令。通常为模型中的一个OP执行单元。
- `Target Translation` 编译器将IRModule变换为目标硬件上可执行的格式（即代码生成），生成的代码被封装为运行时。
- `Passes`：pass是对计算图的一些优化和转换，比如常量折叠，算符融合，死代码消除等等。
- 在编译阶段，一个`relay::Function`可能会被`lower`成多个`tir::PrimFunc`。

tvm中的Pass是一种用于**优化和转换计算图的模块化组件**。在计算图上执行一系列的优化和转换过程，从而使得计算图能够更好地映射到硬件平台上。

（1）TVM中的Pass有两种：

- Relay层的Pass

Relay的Transform是硬件无关的Pass，例如常规的constant folding、dead-code elimination以及张量计算相关的一些特殊Pass如transformation，scaling factor folding。

> 在跑完硬件无关的Pass之后，TVM会将relay::Function lower成多个tir::PrimFunc，然后针对每个tir::PrimFunc进行编译和优化。
> 
- TIR层的Pass

Tir Transform主要包含Tir级别的各种Pass，是偏向编译器方面的优化。这部分的主要功能是lower，不过也有一些optimization，比如多维数据扁平化为一维指针访问、针对特定的后端进行intrinsics扩展。同时也保留了一些底层优化交给下游LLVM做。

实现上，Pass分为：

- Module-Level Pass
    - 利用全局信息进行优化
    - 可以删减Function，如DSE Pass
    - 核心Pass函数是PackedFunc类型
- Function-Level Pass
    - 对Module中的每个Function进行优化，只有局部信息，例如公共子表达式消除
    - 不允许删减Function

（2）Pass操作

Pass的转化逻辑可以简化为：IRModule -> Pass -> … -> IRModule

所有Pass需要继承自ExprFunctor接口。

- 首先**遍历AST，确定哪些Node需要修改**。深度优先遍历，节点可以分为Expr和Stmt。Expr表示表达式节点，如加减乘除、变量等；Stmt表示语句节点，如for循环、if语句等。AST的每个节点都包含了一些属性，例如节点的类型、数据类型、输入输出信息等
- 然后进行**节点修改**。Expression Mutators，用于修改和替换满足条件的Node。对于Expr节点，常见的修改操作包括常量折叠、算子融合、算子替换等；对于Stmt节点，常见的修改操作包括循环展开、循环融合、循环变量重用等

> [【从零开始学深度学习编译器】七，万字长文入门TVM Pass](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247494923&idx=1&sn=0cdde2ecdd1cee546b0847d03cc40b2c&scene=21#wechat_redirect)
> 

### 2、mlir中的Pass

> [https://mlir.llvm.org/docs/PassManagement/](https://mlir.llvm.org/docs/PassManagement/)
> 
> 
> [https://mlir.llvm.org/docs/Passes/](https://mlir.llvm.org/docs/Passes/)
> 

mlir中的Pass是一组通用的IR优化和转换组件，它会**遍历(mlir中的)IR表达式，并对其进行修改或者重构**。Pass可以用来实现不同的优化和转换，会将代码转换为(对部署硬件而言)更高效或者执行静态分析以检测错误等特定优化。

Pass一般使用mlir提供的工具和框架来实现(Pass Manager和Transformation Framework)。

> https://zhuanlan.zhihu.com/p/582635481
> 
> 
> <div style="text-align: center;"><img src="./img_vectorize 和 tensorize Pass/v2-ee6ca5e08aee17b8f9998dd3a3da75c1_r.jpg" alt="v2-ee6ca5e08aee17b8f9998dd3a3da75c1_r.jpg" style="width: 90%;"></div>
> 
- Pass Manager可以自动化地运行一组Passes，以便为特定的目标生成高效的代码
- Transformation Framework允许编写自定义的Pass，这些Pass可以在编译器的不同阶段使用

### 3、tvm和mlir中pass的区别

tvm和mlir中pass的在功能上都是实现代码转化和优化，以获得更优的性能，但在实现和使用在存在一些区别。

- 实现上
    - tvm中的Pass是一组基于HalideIR的模块化优化组件，主要用于优化计算图的结构和算子实现，以提高计算性能和效率
    - mlir中的Pass是一组通用的IR优化和转换组件，可以实现多种功能，例如代码生成、类型检查、调试等
- 使用上
    - tvm中的Pass主要面向深度学习任务，基于静态图
    - mlir中的Pass基于可扩展的高级IR，可以用于处理任何高级IR，并且可以处理动态计算图。提供了更多的优化技术，如内联函数、循环分块、控制流重组等

## tvm中Pass实现

> tvm中vectorize和tensorize两个Pass实现
> 
> 
> [https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Stage](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Stage)
> 
> 一个计算图可以被拆分为多个阶段，每个阶段包含一组计算操作和它们之间的依赖关系。te.Stage就是用来描述一个阶段的，其有助于对计算图进行划分和优化。
> 

### 1、te.Stage的vectorize

（1）函数说明

vectorize(*var*)[¶](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Stage.vectorize)

- Vectorize the iteration.
- Parameters
    - **var** (*[IterVar](https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.IterVar)*) – 需要被vectorize的轴（axis）

（2）用法示例

`vectorize`是用来 将一个`te.Tensor`的某一个维度进行向量化操作 的函数。当使用`vectorize`时，tvm需要指定作用tensor和作用维度，进而对该维度进行向量化并进行SIMD并行加速。

```python
import tvm
from tvm import te

n = te.var('n')
m = te.var('m')
# 占位符张量
A = te.placeholder((n, m), name='A')
# 使用te.compute函数定义了B的计算方式
B = te.compute((n, m), lambda i, j: A[i, j] * 2, name='B')
# 创建一个调度器s，对B进行向量化
s = te.create_schedule(B.op)
# 对张量B的第一个维度进行划分，划分为8个子块
xo, xi = s[B].split(B.op.axis[0], factor=8)
# 对张量B的第二个维度进行张量化操作
s[B].vectorize(xi)
# 生成优化后的代码，以实现向量化优化。
func = tvm.build(s, [A, B], target='llvm')
```

上面的代码中，使用 `vectorize` 对`B`的 `xi`维度进行了向量化操作，tvm会将 `B` 中 `xi` 所在的维度视为向量化维度，并使用 SIMD 指令进行加速。最后调用 `tvm.build` 方法，使用创建的计算图编译出一个可执行的函数。

### 2、te.Stage的tensorize

（1）函数说明

tensorize(*var*, *tensor_intrin*)[¶](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Stage.tensorize)

- Tensorize the computation enclosed by var with tensor_intrin
- Parameters
    - **var** (*[IterVar](https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.IterVar)*) – 需要被tensorize的轴（axis）
    - **tensor_intrin** (*[TensorIntrin](https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.TensorIntrin)*) – 自定义的张量级别实现函数，这个轴上的每个元素都会运用这个函数。

（2）用法示例

`tensorize` 是用来对一个Stage进行自定义的张量级别实现的函数。当使用`tensorize` 时，tvm需要指定作用对象并接受一个函数(tensorized function)作为输入参数，这个函数包含用户自定义的张量级别实现。

`tensorize` 是用来对一个Stage进行自定义的张量级别实现的函数。当使用`tensorize` 时，tvm需要指定作用对象并接受一个函数(tensorized function)作为输入参数，这个函数包含用户自定义的张量级别实现。

[https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html](https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html)

[https://github.com/apache/tvm/blob/main/src/te/operation/tensorize.cc](https://github.com/apache/tvm/blob/main/src/te/operation/tensorize.cc)

- 自定义函数

这个函数其实代表的是某种后端，例如下面的GEMV实现。它包括两部分：

第一部分是GEMV的计算定义，TVM使用它来匹配原始Matmul调度中的计算模式；

第二个是指定如何在设备上执行GEMV，这在intrin_func下面完成。

最后的te.decl_tensor_intrin声明如何执行计算c.op

```python
def intrin_gemv(m, l):
    a = te.placeholder((l,), name="a")
    b = te.placeholder((m, l), name="b")
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])
 
    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "gemv_update",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
                m,
                l,
                bb.strides[0],
            )
        )
        return ib.get()
 
    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})
```

- tensorize用法如下

`tensorize` 是用来对一个算子进行自定义的张量级别实现的函数。当使用`tensorize` 时，tvm需要指定作用对象并接受一个函数(tensorized function)作为输入参数，这个函数包含用户自定义的张量级别实现。

```python
# 代码定义了长度为32的一维向量求和计算
import tvm
from tvm import te

n = 16
m = 32
x = te.placeholder((m,), name="x")
y = te.placeholder((m,), name="y")
z = te.compute(x.shape, lambda i: x[i] + y[i], name="z")
s = te.create_schedule(z.op)
# 将循环次数为32的一重for循环拆分为二重for循环
# 其中外层循环的迭代次数为2，内层循环的迭代次数为16
xo, xi = s[z].split(z.op.axis[0], factor=n)

# tvm.tir.call_packed()调用自定义函数vadd()完成向量求和，并指定了vadd()函数的输入输出数据及数据形状
def vadd(n, x_ptr, y_ptr, out_ptr):
  for i in range(n):
    out_ptr[i] = x_ptr[i] + y_ptr[i]
def intrin_func(ins, outs):
  return tvm.tir.call_packed("vadd", ins[0].data, outs[0].data, ins[0].shape[0])

# 定义计算模式及其对应的intrinsic函数后，可通过调用te.decl_tensor_intrin()函数声明如何执行计算
# tvm.tir.decl_buffer()：声明了intrinsic函数要求的输入输出缓冲区的形状、数据类型和数据布局等信息
Xb = tvm.tir.decl_buffer(x.shape, x.dtype, name="X", offset_factor=1, strides=[1])
Yb = tvm.tir.decl_buffer(y.shape, y.dtype, name="Y", offset_factor=1, strides=[1])
Zb = tvm.tir.decl_buffer(z.shape, z.dtype, name="Z", offset_factor=1, strides=[1])
# te.decl_tensor_intrin()函数
# - 第二个参数intrin_func：指定了执行计算的IR程序
# - 第三个参数binds：Tensor结构到Buffer结构的映射
intrin = te.decl_tensor_intrin(z.op, intrin_func, binds={x: Xb, y: Yb, z: Zb})

# 将intrin函数应用到中间节点的内层循环上
s[z].tensorize(xi, intrin)
```

> 上面的代码可以说明tensorize的使用方法，但应该还存在一些问题，可正确运行的代码见
> 
> 
> [https://zhuanlan.zhihu.com/p/339356901](https://zhuanlan.zhihu.com/p/339356901)