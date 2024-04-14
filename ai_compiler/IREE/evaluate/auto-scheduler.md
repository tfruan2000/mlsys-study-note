# IREE中的scheduler和tuning

## 相关项目和工具

### [mmperf](https://github.com/mmperf/mmperf)

将测试性能的繁琐工作自动化以及可视化，常见的比较对象有TVM IREE HALIDE cublas 等，可惜已经没再更新了

### [SHARK](https://github.com/nod-ai/SHARK/tree/main)

感觉像为IREE开发的tuning工具，将lowerconfig中的参数提取出来进行tuning

### [dispatch_profiler](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/README.md)

IREE 有个调度探查器（dispatch_profiler），该调度探查器会使用 [generator.py](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/generator.py) 生成MLIR Dispatch，然后将这些MLIR Dispatch编译为二进制(vmfb)以验证功能和性能

- 使用 [generator.py](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/generator.py) 生成dispatch

```bash
$ python3 dispatch_profiler/generator.py --generated-dir </path/to/create/`generated`/dir>
[Generating]: ./generated/linalg/matmul/matmul_128x128x256_f16t_f16t_f16t/matmul_128x128x256_f16t_f16t_f16t.mlir
    Emitting tuning configuration : tile_config_128x128_64x4_tensorcore_mmasync
    Emitting tuning configuration : tile_config_128x128_32x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_128x64_32x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_64x64_64x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_64x64_32x10_tensorcore_mmasync
    ...
```

每个Dispatch包含两部分：测试MLIR文件（例如包含 a Matmul operation includes the datatype, layout, and matrix multiplication problem shape） 和 tuning策略（比如不同的tile_sizes, vector_size等）。

- 使用[compile.py](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/compile.py)将生成的dispatch编译为二进制（vmfb）

```bash
$ python3 ../iree/experimental/dispatch_profiler/compile.py --build-dir </path/to/iree/build/dir> --generated-dir </path/to/create/`generated`/dir>

$ ls ./generated/linalg/matmul/matmul_64x64x4096_f16t_f16t_f16t/
iree_compile_cmd_stdout.mlir  matmul_64x64x4096_f16t_f16t_f16t.mlir  matmul_64x64x4096_f16t_f16t_f16t_profile.vmfb  matmul_64x64x4096_f16t_f16t_f16t_verify.vmfb
```

- 使用[profiling.py](https://github.com/openxla/iree/blob/main/experimental/dispatch_profiler/profiler.py)进行功能验证和性能调优

```bash
$ python3 profiler.py --build-dir </path/to/iree/build/dir> --generated-dir </path/to/create/`generated`/dir> --dispatches=matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync --verification-enabled=true --profiling-enabled=true
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_128x128_32x5_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : SUCCESS
Runtime(ms)   : 0.062
GFLOPs        : 233798.62
```



## ours

IREE的dispatch_profiler的不足

- 生成测试文件是随机的，我们期望其能接受特定的输入来做auto-scheduler（其实只是生成一些简单的tile config等）
- 输入的mlir中可能有多个op需要schedule，简单地文本替换是不合适的

### 介绍

对dispatch_profiler进行小小地修改，并使其可用于自动调度程序。主要如下：

1. 对于输入文件，通过MLIR  静态分析的方法，会遍历模块（module）中的所有操作(operation)，得到待调度的目标算子（比如matmul）, 并获取该算子操作数的信息（operand），比如matmul中的输入的shape大小。
2.  根据上述步骤1中分析得到信息生成搜索空间, 用户也可以通过配置文件进行配置该搜索空间。
3. 根据搜索空间生成对应的变换后的MLIR文件，然后编译，运行或者通过算法等进行性能比较，从而获得最佳的调度。

举个例子，对于输入：

```llvm
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

```llvm
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

### 使用

对于一个用户输入来说，比如一个matmul.mlir，用户需要进行配置`auto-scheduler.cfg`，例如

```bash
[Env]
iree_bin = xxx/iree_build/tools 
input_file = matmul.mlir
output_file = matmul.csv

input_file = matmul.mlir
output_file = matmul.csv

[SearchSpace]
tile_m = [64,128]
tile_n = [32,64]
tile_k = [32,64,128,256]
block_dim_x = [64]
block_dim_y = [2]
block_dim_z = [1]                                                                         
pipeline_depth = [2]

[SearchStrategy]
tuner = XGBoostTuner
n_trial = 16 
```

参数说明：

- tile_m, tile_n, tile_k 表示 in m, n, k dimensions, 值的范围是 [2,4,8,16,32,64...]

- block_dim_x, block_dim_y, block_dim_z （必须为1）表示一个thread block的x, y, z维度所包含的线程数 值的范围是 [2,4,8,16,32,64...]，并且三者乘积必须不大于1024

- tuning的代码是来源于tvm的[tuner](https://github.com/apache/tvm/tree/main/python/tvm/autotvm/tuner)

然后运行：

```bash
python run.py
```


如下：

```bash
Step 1: Generate search space and transform the original IR
===========================================================
Transforming ......
1
0 hours 0 minutes 1 seconds
==========================================================

Step 2: Search the space to get the best performance config
==========================================================
[Compiling (profile)] ../iree-build/tools/iree-compile generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_64x64x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync.mlir -o generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_64x64x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync_profile.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --iree-codegen-llvmgpu-use-mma-sync --iree-hal-benchmark-dispatch-repeat-count=64
---------------------------------------------------------------- 
Dispatch      : matmul_512x512x512_f32t_f32t_f32t_tilesize_64x64x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_512x512x512_f32t_f32t_f32t
Configuration : tilesize_64x64x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync
Arguments     : --batch_count=1 --m=512 --n=512 --k=512 --lhs=f32t --rhs=f32t --result=f32t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : 
Runtime(ms)   : 0.097
GFLOPs        : 2767.38
[Compiling (profile)] ../iree-build/tools/iree-compile generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_128x64x64_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync.mlir -o generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_128x64x64_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync_profile.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --iree-codegen-llvmgpu-use-mma-sync --iree-hal-benchmark-dispatch-repeat-count=64
---------------------------------------------------------------- 
Dispatch      : matmul_512x512x512_f32t_f32t_f32t_tilesize_64x32x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_512x512x512_f32t_f32t_f32t
Configuration : tilesize_64x32x128_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync
Arguments     : --batch_count=1 --m=512 --n=512 --k=512 --lhs=f32t --rhs=f32t --result=f32t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : 
Runtime(ms)   : 0.095
GFLOPs        : 2825.64
[Compiling (profile)] ../iree-build/tools/iree-compile generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_64x32x64_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync.mlir -o generated/linalg/matmul/matmul_512x512x512_f32t_f32t_f32t/tilesize_64x32x64_interchange_0x1x2_vectorsize_32x32_2_64x2x1_tensorcore_mmasync_profile.vmfb --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --iree-codegen-llvmgpu-use-mma-sync --iree-hal-benchmark-dispatch-repeat-count=64
---------------------------------------------------------------- 
...

********************************************************
[Performance]: The best is tilesizes:(64x32x128)_interchange:(012)_vectorsizes:(32x32)_stages:2_blockdims:(64x2x1)    tensorcore mmasync 
with gflops 2825.64

0 hours 0 minutes 30 seconds
********************************************************
==========================================================
```
```bash
## 对比
测试的对象是矩阵乘（matmul），shape大小如下, 数据类型为Float32,

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

对比可以发现，在[4096，4096，4096]规模大matmul中， IREE中的tile_sizes = [32, 32, 16]， 而我们搜索出来的较优的tile_sizes=[128, 64, 32]



### 代码分析

`run.py` 流程

(1) 解析 `auto-schedule.cfg` 文件，获得调优配置

- Env: 工具链位置、输入文件名、输出文件名
- SearchSpace: 定义调优参数和范围，类似triton中的auto-tuning配置的 `configs`
- SearchStrategy: 调优器类型、迭代次数

然后将调优配置写入为ctx对象

`ctx = build_context_from_arguments(parser)`

```bash
# option.py
def build_context_from_arguments(parser):
  """Adds and parse all the arguments for the *profiler.py* script."""
  add_typical_arguments(parser)
  add_matmul_arguments(parser)
  add_compilation_arguments(parser)
  add_iree_compile_arguments(parser)
  add_verification_arguments(parser)
  add_profiling_arguments(parser)
  add_performance_report_arguments(parser)

  # Additional arguments for the profiler.
  parser.add_argument("--save-cmds", action='store_true', \
                      help='Saves commands and their output that are executed '\
                      'by the profiler in a file.')

  args = parser.parse_args()
  args.verification_enabled = False if args.verification_enabled in [
      'False', 'false', '0'
  ] else True
  args.profiling_enabled = False if args.profiling_enabled in [
      'False', 'false', '0'
  ] else True
  args.dims=[0 for i in range(7)]
  args.is_valid_index=[]
  args.valid_indexes=[]
  return args
```

(2) 根据ctx对输入的mlir进行下降变换 `transform(ctx)`

```python
# transform.py
def transform(args):
  analysisObject = Analysis(args.input_mlir)
  analysisObject.anlysis()
  args.problem_m = analysisObject.problem_m
  args.problem_n = analysisObject.problem_n
  args.problem_k = analysisObject.problem_k
  # Manifest dispatches for a group of accompanying operations and configurations.
  manifest = Manifest(args)
  print("Transforming ......")
  # Load all the pre-defined dispatches in a manifest.
  manifest.initialize()

  # Emit the dispatches in MLIR source files.
  manifest.emit()
```

其中Manifest对象是存储dispatch的数据结构

(3) 空间探索，获得最优策略

根据搜索空间，调优参数，以 `gridTuner` 为例

```python
def getNextIndex(index, step, begin, end):
  if index < begin or index > end:
    return None
  if step <= 0:
    return None
  i = index + step
  if i > end:
    return end
  else:
    return i

def gridTuner(args):
  # Create manifest object and load dispatches.
  manifest = Manifest(args)
  manifest.load()

  # Performance report
  perf_report = PerformanceReport(args)
  n_trial = int(args.n_trial)
  print("try run ")
  print(n_trial)
  visited = []

  current_index = 0
  i = 0
  error_run = 0
  space_length = 0
  # For all the operations in the manifest compile (if needed), verify, and profile.
  for _, dispatch_collection_list in manifest.dispatch_collection_map.items():
    for dispatch_collection in dispatch_collection_list:
      operation = dispatch_collection.operation
      # Select and create an instance of operation_launcher for the operation.
      valid_space_length = len(args.valid_indexes)
      #space_length = len(dispatch_collection.configuration_list)

      filepath = "run.log"
      with open(filepath, "a") as fp:
        fp.write("search space size is : {}\n".format(space_length))

      max_iteration = min(n_trial, valid_space_length)
      step = valid_space_length / n_trial
      if step < 1:
        step = 1
      while (i < max_iteration):
        i = i + 1
        next_index = getNextIndex(current_index, int(step), 0, valid_space_length)
        if next_index == None:
          break
        current_index = next_index
        r = args.valid_indexes[current_index]
        configuration = dispatch_collection.configuration_list[r]
        operation_launcher = IreeToolsLauncher(args, operation, configuration.name())
        # Create a dispatch object.
        dispatch = Dispatch(operation, configuration)

        # Skip the dispatch if filter returns false.
        if not manifest.is_enabled(dispatch):
          continue

        # If dry run is enabled, skip the dispatch.
        if args.dry_run:
          print(f'[Dry run] : {dispatch.name()}')
          continue

        # Initialize verification and profiling results.
        verification_result = 'Not verified' if not args.verification_enabled else 'Failed'
        runtime = -1.0

        # Launch the operation dispatches for verification and profiling.
        if args.verification_enabled:
          verification_result = operation_launcher.verify(configuration)
        if args.profiling_enabled:
          runtime = operation_launcher.profile(configuration)

        if runtime == "":
          error_run = error_run + 1
          continue
        # Create performance result.
        result = PerformanceResult(operation, configuration,
                                   verification_result, runtime)

        # Print the performance result.
        if args.debug:
          result.print()

        # Append the performance result to the performance report.
        perf_report.append_perf_result(result)

  valid_run = space_length - error_run
  filepath = "run.log"
  with open(filepath, "a") as fp:
    fp.write("valid run: {}\n".format(valid_run))
```

XGBoostTuner的参数搜索效率会更好，核心思想类似tvm的[tuner](https://github.com/apache/tvm/tree/main/python/tvm/autotvm/tuner)

> XGBoost使用梯度提升算法构建一个由多个弱学习器组成的集成模型。它通过迭代地训练弱学习器，每次迭代都尝试减小模型在训练数据上的损失函数。每个弱学习器都是在前一个弱学习器的残差基础上进行训练，以逐步减小预测误差。

`run.py`代码：

```python
import configparser
import re

import argparse
from analysis import *
from library import *
from matmul import *
from manifest import *
from options import build_context_from_arguments
from transform import *
import time
from searchspace import *
from compile import *
from profiler import *
from performance_analysis import *
from xgboost_tuner import *
from sa_model_optimizer import *
from datetime import datetime

def init_context():
    # 解析cfg内容
    config = configparser.ConfigParser()
    config.read('auto-scheduler.cfg')
    input_file = config['Env']['input_file']
    output_file = config['Env']['output_file']
    ss_key = 'SearchSpace'
    if config.has_option(ss_key, 'tile_m'):
      tile_m = config[ss_key]['tile_m']
    else:
      tile_m = ''
    if config.has_option(ss_key, 'tile_n'):
      tile_n = config[ss_key]['tile_n']
    else:
      tile_n = ''
    if config.has_option(ss_key, 'tile_k'):
      tile_k = config[ss_key]['tile_k']
    else:
      tile_k = ''

    if config.has_option(ss_key, 'block_dim_x'):
      block_dim_x = config[ss_key]['block_dim_x']
    else:
      block_dim_x = ''
    if config.has_option(ss_key, 'block_dim_y'):
      block_dim_y = config[ss_key]['block_dim_y']
    else:
      block_dim_y = ''
    if config.has_option(ss_key, 'block_dim_z'):
      block_dim_z = config[ss_key]['block_dim_z']
    else:
      block_dim_z = ''

    if config.has_option(ss_key, 'pipeline_depth'):
      pipeline_depth = config[ss_key]['pipeline_depth']
    else:
      pipeline_depth = ''

    tuner = config['SearchStrategy']['tuner']
    n_trial = config['SearchStrategy']['n_trial']

    parser = argparse.ArgumentParser(description="")
    ctx = build_context_from_arguments(parser)
    ctx.input_mlir = input_file
    # input:'[1, 2, 4]'
	# output: [1,2,4]
    ctx.tile_m = getTileSpace(tile_m)
    ctx.tile_n = getTileSpace(tile_n)
    ctx.tile_k = getTileSpace(tile_k)
    if block_dim_x != '':
      ctx.block_dim_x = getTileSpace(block_dim_x)
    else:
      ctx.block_dim_x = ''
    if block_dim_y != '':
      ctx.block_dim_y = getTileSpace(block_dim_y)
    else:
      ctx.block_dim_y = ''
    if block_dim_z != '':
      ctx.block_dim_z = getTileSpace(block_dim_z)
      if len(ctx.block_dim_z) != 1 or ctx.block_dim_z[0] != 1:
        print("[Error]: Expected workgroup size in z-dim = 1")
        print("Please set `block_dim_z = [1]` in the config file.")
        exit(1)
    else:
      ctx.block_dim_z = ''

    if pipeline_depth != '':
      ctx.pipeline_depth = getTileSpace(pipeline_depth)
    else:
      ctx.pipeline_depth = ''

    ctx.verification_enabled = False
    ctx.output = output_file
    ctx.tuner = tuner
    ctx.n_trial = n_trial
    return ctx

def compute_searchspace_size(ctx):
    manifest = Manifest(ctx)
    manifest.load()
    #searchspace_size = 0
    ss = 0
    #ss = ctx.dims[0] * ctx.dims[1] * ctx.dims[2] * ctx.dims[3] * ctx.dims[4] * ctx.dims[5] * ctx.dims[6]
    for _, dispatch_collection_list in manifest.dispatch_collection_map.items():
          for dispatch_collection in dispatch_collection_list:
            operation = dispatch_collection.operation
                # Select and create an instance of operation_launcher for the operation.
            configuration_list = dispatch_collection.configuration_list
            searchspace_size = len(configuration_list)
            ctx.configuration_list = configuration_list
            ss = len(configuration_list)
    return ss

def getRunTime(run_time):
    hour = int(run_time / 3600)
    minute = int((run_time - 3600 * hour) / 60)
    second = run_time - 3600 * hour -60 * minute
    return "{} hours {} minutes {} seconds".format(hour, minute, second)

if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    filepath = "run.log"
    with open(filepath, "a") as fp:
      fp.write("[Date]: {}\n\n".format(dt_string))

    ctx = init_context()
    print("Step 1: Generate search space and transform the original IR")
    print("===========================================================")
    starttime = time.time()
    transform(ctx)
    endtime = time.time()
    transform_time = int(round(endtime -starttime))
    print(transform_time)
    t = getRunTime(transform_time)
    print(t)
    with open(filepath, "a") as fp:
      fp.write("[Transform Time]: {}\n\n".format(t))

    print("==========================================================")
    print("\nStep 2: Search the space to get the best performace' config")
    print("==========================================================")
    s = compute_searchspace_size(ctx)
    ctx.searchspace_size = s
    with open(filepath, "a") as fp:
      fp.write("[Search Space Size]: {}\n\n".format(s))

    starttime = time.time()
    profile(ctx)
    print("\n********************************************************")
    getBestConfig()
    endtime = time.time()
    search_time = int(round(endtime -starttime))
    t = getRunTime(search_time)
    print(t)
    with open(filepath, "a") as fp:
      fp.write("[Search Time]: {}\n\n".format(t))
      fp.write("=======================================\n\n")
    print("********************************************************")
    print("==========================================================")

```

