# [IREE] 性能测试

## IREE end2end

> 位于 iree/tests/transform_dialect/cpu/
> 

```bash
iree-compile matmul.mlir --iree-hal-target-backends=llvm-cpu --compile-to=hal \
  --iree-codegen-llvmcpu-use-transform-dialect=./matmul_codegen_default_spec.mlir >a.log

iree-opt matmul.mlir --iree-hal-target-backends=llvm-cpu\
  --iree-common-input-transformation-pipeline \
  --iree-abi-transformation-pipeline \
  --iree-flow-transformation-pipeline  \
  --iree-stream-transformation-pipeline \
  --iree-hal-configuration-pipeline | \
iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
  --iree-codegen-llvmcpu-use-transform-dialect=./matmul_codegen_default_spec.mlir >b.log

iree-opt matmul.mlir --iree-hal-target-backends=llvm-cpu\
  --iree-common-input-transformation-pipeline \
  --iree-abi-transformation-pipeline \
  --iree-flow-transformation-pipeline  \
  --iree-stream-transformation-pipeline \
  --iree-hal-transformation-pipeline \
  --iree-codegen-llvmcpu-use-transform-dialect=./matmul_codegen_default_spec.mlir >c.log

# git diff --no-index a.log c.log
# a.log == c.log
```

```cpp
--dump-pass-pipeline              - Print the pipeline that will be run
--mlir-print-ir-after-all          - Print IR after each pass
```

下面这两个命令行是等同的

```bash
iree-compile --iree-hal-target-backends=llvm-cpu --compile-to=hal

<-->

iree-opt --iree-hal-target-backends=llvm-cpu\
  --iree-common-input-transformation-pipeline \
  --iree-abi-transformation-pipeline \
  --iree-flow-transformation-pipeline  \
  --iree-stream-transformation-pipeline \
  --iree-hal-transformation-pipeline
```

## 基于IREE的Auto-scheduler

[http://62.234.201.16/mlir-related/iree-auto-scheduler](http://62.234.201.16/mlir-related/iree-auto-scheduler)

IREE 有个[调度探查器](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/README.md) （dispatch_profiler）, 该调度探查器会随机生成输入文件，然后针对每一输入文件，通过在Linalg层注入operation 的Attribution的方式，进行不同的调度（比如不同的tile_sizes, vector_size等）。 然后对不同的调度可以进行性能测试。

我们对此进行了改进，并使其可用于自动调度程序。主要如下：

1. 对于输入文件，通过MLIR  静态分析的方法，会遍历模块（module）中的所有操作(operation)，得到待调度的目标算子（比如matmul）, 并获取该算子操作数的信息（operand），比如matmul中的输入的shape大小。
2.  根据上述步骤1中分析得到信息生成搜索空间, 用户也可以通过配置文件进行配置该搜索空间。
3. 根据搜索空间生成对应的变换后的MLIR文件，然后编译，运行或者通过算法等进行性能比较，从而获得最佳的调度。

举个例子，对于输入：

```bash
!matrixA_type = tensor<1024x256xf32>
!matrixB_type = tensor<256x512xf32>
!matrixC_type = tensor<1024x512xf32>
 
module @arithmetic {
   func.func @matmul(
        %arg0: !matrixA_type , %arg1: !matrixB_type)
            -> !matrixC_type {
        %cst = arith.constant 0.000000e+00 : f32
        %init = tensor.empty() : !matrixC_type
        %1 = linalg.fill ins(%cst : f32) outs(%init : !matrixC_type) -> !matrixC_type
        %3 = linalg.matmul ins(%arg0, %arg1: !matrixA_type, !matrixB_type)
                            outs(%1: !matrixC_type)
            -> !matrixC_type
        func.return %3 : !matrixC_type
    }
}
```

会生成不同的下面的文件：

```bash
#config = #iree_codegen.lowering_config<tile_sizes = [[8, 2, 4]]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulSimt>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation, workgroup_size = [32, 1, 1]>
module @arithmetic {
  func.func @matmul(%arg0: tensor<1024x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<1024x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %2 = linalg.matmul {compilation_info = #compilation} ins(%arg0, %arg1 : tensor<1024x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    return %2 : tensor<1024x512xf32>
  }
}
```

**使用方法：**

对于一个用户输入来说，比如一个matmul.mlir, 用户需要进行配置

```bash
[Env]
iree_bin = ~/work/iree-build-auto/tools 
input_file = matmul.mlir
output_file = matmul.csv

[SearchSpace]
tile_x = [2,4,8,16]
tile_y = [2,4,8,16]

[SearchStrategy]
tuner = GridTuner
n_trial = 1000
```

然后运行：

```bash
python tunner.py
```

如下：

```bash

Step 1: Generate search space and transform the original IR
===========================================================
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_2x4_simt_ffma.mlir
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_4x4_simt_ffma.mlir
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_4x2_2x4_simt_ffma.mlir
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_4x2_4x4_simt_ffma.mlir
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_8x2_2x4_simt_ffma.mlir
[Generating]: generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_8x2_4x4_simt_ffma.mlir
==========================================================

Step 2: Compile the transform'd IR
[Compiling (profile)] ../iree-build/tools/iree-compile generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_2x4_simt_ffma.mlir -o generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_2x4_simt_ffma_profile.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --iree-hal-benchmark-dispatch-repeat-count=100
[Compiling (verify)] ../iree-build/tools/iree-compile generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_2x4_simt_ffma.mlir -o generated/linalg/matmul/matmul_1024x512x256_f32t_f32t_f32t/tile_config_2x2_2x4_simt_ffma_verify.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --iree-hal-benchmark-dispatch-repeat-count=1
...
==========================================================

Step 3: Search on the space to get best performace' config
==========================================================
---------------------------------------------------------------- 
Dispatch      : matmul_1024x512x256_f32t_f32t_f32t_tile_config_4x2_2x4_simt_ffma
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_1024x512x256_f32t_f32t_f32t
Configuration : tile_config_4x2_2x4_simt_ffma
Arguments     : --batch_count=1 --m=1024 --n=512 --k=256 --lhs=f32t --rhs=f32t --result=f32t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : Not verified
Runtime(ms)   : 68.5
GFLOPs        : 3.92
---------------------------------------------------------------- 
Dispatch      : matmul_1024x512x256_f32t_f32t_f32t_tile_config_8x2_2x4_simt_ffma
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_1024x512x256_f32t_f32t_f32t
Configuration : tile_config_8x2_2x4_simt_ffma
Arguments     : --batch_count=1 --m=1024 --n=512 --k=256 --lhs=f32t --rhs=f32t --result=f32t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : Not verified
Runtime(ms)   : 67.6
GFLOPs        : 3.97
---------------------------------------------------------------- 
...

The best tile sizes config is:
['2', '2', '4']
```
```

## 对比

测试的对象是矩阵乘（matmul），shape大小如下, 数据类型为Float32,

```jsx
GMM (Matrix Multiply).
– (128, 128, 128)
– (512, 512, 512)
– (1024, 1024, 1024)
- (4096, 4096, 4096)
```

使用的命令

```bash
iree-compile matmul.mlir \
--iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_70 \
--iree-codegen-llvmgpu-use-mma-sync \
--iree-hal-benchmark-dispatch-repeat-count=64 \
-o 1.vmfb
```

sm_70是V100，sm_80是A100

```bash
iree-benchmark-module \
--device=cuda --benchmark_repetitions=50 --batch_size=64 \
--module=1.vmfb \
--function=matmul \
--input=1024x1024xf32 \
--input=1024x1024xf32

$IREE_OPT/iree-compile matmul.mlir -o 1.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_70 --iree-codegen-llvmgpu-use-mma-sync --iree-hal-benchmark-dispatch-repeat-count=64
$IREE_OPT/iree-benchmark-module --module=1.vmfb --device=cuda --benchmark_repetitions=50 --batch_size=64 --function=matmul --input=1024x1024xf32 --input=1024x1024xf32
```

- 测试了IREE auto-scheduler与IREE本身的结果对比：

| 矩阵大小 | 性能(TFLOPS)（IREE-auto-scheduler/IREE） |
| --- | --- |
| 128, 128, 128 | 0.419/0.418 |
| 512, 512, 512 | 3.355/3.327 |
| 1024, 1024, 1024 | 9.296/9.138 |
| 4096,4096,4096 | 26.430/26.124 |

对比了一下，比如4096，4096，4096， IREE中的tile_sizes = [[32, 32, 16]， 而我们搜索出来的较优的tile_sizes=[[128, 64, 32]]

- tvm（auto_scheduler）自动调优测试程序

```python
import numpy as np
import tvm
import argparse
from tvm import te, auto_scheduler
import time

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j], name="out")

    return [A, B, out]

def get_run_time(run_time):
    hour = int(run_time/3600)
    minute = int((run_time-3600*hour)/60)
    second = run_time - 3600*hour -60*minute
    return "{} hours {} minutes {} seconds".format(hour, minute, second)

def test_matmul(target, M, N, K):
  starttime = time.time()
  task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, "float32"), target=target)

  log_file = "matmul_{}_{}_{}.json".format(M, N, K)
  tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=0,
  )

  task.tune(tune_option)
  endtime = time.time()

  all_run_time = int(round(endtime -starttime))
  t = get_run_time(all_run_time)
  print("Search Time for matix:", M, N, K)
  print(t)

  filepath = "run.log"
  with open(filepath, "a") as fp:
    fp.write("Search time for matmul with {}_{}_{}:\n".format(M, N, K))
    fp.write(t)

  sch, args = task.apply_best(log_file)

  func = tvm.build(sch, args, target)
  a_np = np.random.uniform(size=(M, K)).astype(np.float32)
  b_np = np.random.uniform(size=(K, N)).astype(np.float32)
  out_np = a_np.dot(b_np)

  dev = tvm.cuda()
  a_tvm = tvm.nd.array(a_np, device=dev)
  b_tvm = tvm.nd.array(b_np, device=dev)
  out_tvm = tvm.nd.empty(out_np.shape, device=dev)
  func(a_tvm, b_tvm, out_tvm)

  # Check results
  np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

  # Evaluate execution time.
  evaluator = func.time_evaluator(func.entry_name, dev, number=100)
  execute_time = np.median(evaluator(a_tvm, b_tvm,
      out_tvm).results) * 1000
  num_flops = M * N * K * 2
  gflops = num_flops / execute_time / 1e6
  print(
    "Execution time of this operator: %.3f ms"
    % execute_time)
  print("GFLOPS: %f" % gflops)
  with open(filepath, "a") as fp:
      fp.write("\nExecution time:{}".format(execute_time))
      fp.write("\ngflops:{}".format(gflops))

if __name__ == '__main__':
  test_matmul("cuda", 128, 128, 128)
  test_matmul("cuda", 512, 512, 512)
  test_matmul("cuda", 1024, 1024, 1024)
  test_matmul("cuda", 4096, 4096, 4096)
```