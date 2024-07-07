# [IREE] run-module & benchmark-module

如何用起来 `iree-run-module` 和 `iree-benchmark-module`

[[IREE] 编译流程](https://www.notion.so/IREE-0a7077148d804ed19f45f5a5cbc0f730?pvs=21)

## tools 介绍

### `iree-compile`

[tools\iree-compile-main.cc](https://github.com/openxla/iree/blob/11388b8f54620c1f960aa9eb7a8d573e3b6d1335/tools/iree-compile-main.cc)

```bash
# 下面是一个简单的.mlir生成.vmfb的例子，只需要在host端编译，就避免了使用hal dialect
iree-compile --iree-hal-target-backends=llvm-cpu \
     samples/custom_module/static/test/example.mlir \
     -o=/tmp/example.vmfb
```

```bash
--compile-to=<value>                                                       - Compilation phase to run up until before emitting output.
    =input                                                                   -   Performs input processing and lowering into core IREE input dialects (linalg/etc).
    =abi                                                                     -   Adjusts program ABI for the specified execution environment.
    =preprocessing                                                           -   Compiles up to the `preprocessing` specified
    =flow                                                                    -   Compiles up to the `flow` dialect.
    =stream                                                                  -   Compiles up to the `stream` dialect.
    =executable-sources                                                      -   Compiles up to just before `hal.executable`s are translated, excluding codegen.
    =executable-targets                                                      -   Compiles up to translated `hal.executable`s, including codegen.
    =hal                                                                     -   Compiles up to the `hal` dialect, including codegen.
    =vm                                                                      -   Compiles up to the `vm` dialect.
    =end                                                                     -   Complete the full compilation pipeline.

--output-format=<value>                                                    - Format of compiled output
    =vm-bytecode                                                             -   IREE VM Bytecode (default)
    =vm-c                                                                    -   C source module
    =vm-asm                                                                  -   IREE VM MLIR Assembly

--iree-vm-bytecode-module-optimize                                         - Optimizes the VM module with CSE/inlining/etc prior to serialization
```

### `iree-run-module`

[tools\iree-run-module-main.c](https://github.com/openxla/iree/blob/11388b8f54620c1f960aa9eb7a8d573e3b6d1335/tools/iree-run-module-main.c)

对 单个入口函数的调用 进行测试。对 MLIR 中的函数性能测试一般需要使用 C++ module wrapper layer 生成调用接口并编写相应的 C++ 函数，而 `iree-run-module`  工具可以 **自动加载测试目标函数** 。

```bash
# 测试的这个函数不需要输入
iree-run-module \
    --device=local-task \
    --module=/tmp/example.vmfb \
    --function=main
```

### `iree-benchmark-module`

[iree\tools\iree-benchmark-module-main.cc](https://github.com/openxla/iree/blob/11388b8f54620c1f960aa9eb7a8d573e3b6d1335/tools/iree-benchmark-module-main.cc)

对 单个入口函数的调用 进行基准测试，接受和 `iree-run-module` 相同的参数，测量 VM 调用该函数所花费时间，包括分配和释放 output buffers

先使用 `iree-compile` 为目标后端后端生成 `IREE module` ，然后对 module 中暴露的 function 进行 benchmark 测试（使用google benchmark）

```bash
iree-run-module \
    --device=local-task \
    --module=/tmp/example.vmfb \
    --function=main
```

如果没有指定 `entry_function` （即不止一个函数）， `iree-benchmark-module` 会为每一个没有输入的函数都注册一个 benchmark（同时测多个函数的benchmark）

> IREE 基准测试为我们提供了特定粒度级别的程序性能的 准确且可重现的视图，但是为了更深地分析程序性能，我们需要了解 [IREE Profiling](https://github.com/openxla/iree/blob/main/docs/developers/developing_iree/profiling.md)
>

如果是在需要在device端运行（例如cuda），则需要使用完整的编译流程

```bash
$IREE_OPT/iree-run-module --list_drivers=vulkan
# ============================================================================
# Available HAL drivers
# ============================================================================
# Use --list_devices={driver name} to enumerate available devices.

            cuda: CUDA (dynamic)
      local-sync: Local execution using a lightweight inline synchronous queue
      local-task: Local execution using the IREE multithreading task system
          vulkan: Vulkan 1.x (dynamic)
```

使用 [hal](https://www.notion.so/2023-5-1-5-7-33fe6955c4b64690bb38913c9d41b72f?pvs=21) 相关语句 来指定后端、指定后端架构等

```bash
iree-compile matmul.mlir  \
    --iree-hal-target-backends=cuda \
    --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
    --iree-codegen-llvmgpu-use-transform-dialect=transform_wmma.mlir \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-hal-benchmark-dispatch-repeat-count=10 \
    -o transformed.vmfb

iree-run-module \
    --device=cuda \
    --module=transformed.vmfb \
    --function=matmul_static \
    --input="3456x2048xf32=1" --input="2048x1024xf32=1"

iree-benchmark-module \
    --device=cuda \
    --module=transformed.vmfb \
    --function=matmul_static \
    --input="3456x2048xf32=1" --input="2048x1024xf32=1"
```

## 运行示例

### [simple_abs.mlir](https://github.com/openxla/iree/blob/93038251a167c67a045fab2896ff34d3ed72aa9e/samples/models/simple_abs.mlir)

```bash
$IREE_OPT/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    simple_abs.mlir -o ./tmp/module.vmfb

$IREE_OPT/iree-run-module \
    --device=local-task \
    --module=./tmp/module.vmfb \
    --function=abs \
    --input=f32=-2

EXEC @abs
result[0]: hal.buffer_view
f32=2

$IREE_OPT/iree-benchmark-module \
    --device=local-task \
    --module=./tmp/module.vmfb \
    --function=abs \
    --input=f32=-2

Run on (12 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 1024K (x6)
  L3 Unified 8448K (x1)
Load Average: 2.21, 1.93, 3.34
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may
 be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_RunModule/process_time/real_time       0.22 ms         0.23 ms         3356
```

```mlir
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
```

### add.mlir (cuda为后端)

```bash
# First compile into a VM bytecode module.
# --iree-hal-target-backends=llvm-cpu
$IREE_OPT/iree-compile \
    --iree-hal-target-backends=cuda \
    add.mlir -o ./tmp/add.vmfb

# Run the module through CUDA HAL backend.
$IREE_OPT/iree-run-module \
    --device=cuda \
    --module=./tmp/add.vmfb \
    --function=add \
    --input="4xf32=[1 2 3 4]" \
    --input="4xf32=[2 2 2 2]"

$IREE_OPT/iree-benchmark-module \
    --device=cuda \
    --module=./tmp/add.vmfb \
    --function=add \
    --input="4xf32=[1 2 3 4]" \
    --input="4xf32=[2 2 2 2]"

EXEC @add
result[0]: hal.buffer_view
4xf32=3 4 5 6
```

```mlir
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
      outs(%0 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
```

### [matmul.mlir](https://github.com/openxla/iree/blob/11388b8f54620c1f960aa9eb7a8d573e3b6d1335/tests/transform_dialect/cpu/matmul.mlir)

```bash
iree-compile matmul.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-codegen-llvmcpu-use-transform-dialect=./matmul_codegen_default_spec.mlir \
  -o tmp/matmul.vmfb

iree-run-module \
    --device=local-task \
    --module=tmp/matmul.vmfb \
    --function=matmul_static \
    --input="3x5xf32=1" \
    --input="5x3xf32=2" \
    --input="3x3xf32=42"

iree-benchmark-module \
    --device=local-task \
    --module=tmp/matmul.vmfb \
    --function=matmul_static \
    --input="3x5xf32=1" \
    --input="5x3xf32=2" \
    --input="3x3xf32=42"
```

```mlir
!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
```

```mlir
// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall with tile_sizes [2].
  // ===================================================
  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %matmul tile_sizes [2]
      ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
    : (!transform.any_op) -> ()

  // Step 2. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
}
```

### our_matmul.mlir

```bash
❯ iree-compile matmul.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-codegen-llvmcpu-use-transform-dialect=./matmul_codegen_default_spec.mlir \
  -o tmp/matmul.vmfb

❯ iree-benchmark-module \
          --device=local-task \
          --module=tmp/matmul.vmfb \
          --function=matmul_tensors \
          --input="128x128xf32=1" \
          --input="128x128xf32=2" \
          --input="128x128xf32=0"

Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
2023-06-16T11:21:49+08:00
Running iree-benchmark-module
Run on (8 X 24.0874 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.04, 1.87, 1.89
---------------------------------------------------------------------------------------------------
Benchmark                                         Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------------------
BM_matmul_tensors/process_time/real_time      0.683 ms         2.65 ms          989 items_per_second=1.46409k/s
```

```mlir
func.func @matmul_tensors(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>,
    %D: tensor<128x128xf32>)
        -> tensor<128x128xf32> {
    %ret = linalg.matmul ins(%A, %B: tensor<128x128xf32>, tensor<128x128xf32>)
                        outs(%D: tensor<128x128xf32>)
        -> tensor<128x128xf32>
    func.return %ret : tensor<128x128xf32>
}

/// RUN: mlir-opt -llvm-request-c-wrappers --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule matmul.mlir \
/// RUN: | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1" -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
/// RUN: | mlir-translate -mlir-to-llvmir \
/// RUN: | llc -filetype=obj -o llvm.o

// module {
//     func.func @matmul_tensors(
//         %A: tensor<128x128xf32>, %B: tensor<128x128xf32>,
//         %D: tensor<128x128xf32>)
//             -> tensor<128x128xf32> {
//         %ret = linalg.matmul ins(%A, %B: tensor<128x128xf32>, tensor<128x128xf32>)
//                             outs(%D: tensor<128x128xf32>)
//             -> tensor<128x128xf32>
//         func.return %ret : tensor<128x128xf32>
//     }

//     transform.sequence failures(propagate) {
//     ^bb1(%module: !pdl.operation):
//         %matmul = transform.structured.match ops{["linalg.matmul"]}
//             in %module : (!pdl.operation) -> !pdl.operation
//         %func = transform.structured.match ops{["func.func"]}
//             in %module : (!pdl.operation) -> !pdl.operation
//         transform.structured.tile %matmul [16, 2, 1]
//          : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
//         transform.structured.vectorize %func : (!pdl.operation) -> !pdl.operation
//     }
// }
```

参考：[codegen_spec.mlir](https://github.com/openxla/iree/blob/93038251a167c67a045fab2896ff34d3ed72aa9e/tests/transform_dialect/cpu/attention_codegen_spec.mlir) 修改

```mlir
transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall with tile_sizes [2].
  // ===================================================
  %forall, %tiled_generic =
    transform.structured.tile_to_forall_op %matmul tile_sizes [2]
      ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
    : (!transform.any_op) -> ()

    // Step 2. Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %variant_op : !transform.any_op
    transform.iree.apply_cse %variant_op : !transform.any_op

    // Step 3. Bufferize and drop HAL decriptor from memref ops
    // ==========================================
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

    // Step 4. Post-bufferization mapping workgroup and vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.iree.apply_cse %func_8 : !transform.any_op
    transform.iree.apply_buffer_optimizations %func_8 : (!transform.any_op) -> ()
}
```

## summary

<div style="text-align: center;"><img src="./img_benchmark-module/compilation_flow.png" alt="compilation_flow" style="width: 90%;"></div>

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

下面这两个命令行是等同的，符合[IREE Pipeline](../pipeline/pipeline.md) 中的描述

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