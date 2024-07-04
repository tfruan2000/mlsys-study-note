# Triton-Linalg

doing
- [ ] 介绍（背景、优缺点、和triton-shared的区别）
- [x] 环境配置 : clone & 编译
- [ ] 测试使用（测试一些例子，简单介绍一下使用）
- [ ] dialect
  - [ ] Arith
  - [ ] Auxiliar
  - [ ] LinalgExt
  - [ ] MathExt
  - [ ] Triton
- [ ] analysis
- [ ] conversion
- [ ] pipeline


## 介绍

### what's this

[triton-linalg](https://github.com/Cambricon/triton-linalg)

### what can we do with this

- 扩展性： linalg - to - HW special dialect
- 中间层级优化：trion目前GPU的下降路线过于生硬，可以说是直接一把 `conversion`，一把下降会导致难以优化中间 IR（例如离散性优化），这对 `SIMT` 虽然影响不大，但是离散地访存行为对 `SIMD` 的影响无疑是巨大的。

### triton-shared

[triton-shared](https://github.com/microsoft/triton-shared) 是 microsoft（巨硬）家实现 triton-to-linalg 的工作（以及实现以CPU作为后端），也扩展了特定的 Dialect。

### diff with triton-shared

## 配置环境

- clone

```bash
export TRITON_PLUGIN_DIRS=$(pwd)/triton-linalg
git clone --recurse-submodules https://github.com/Cambricon/triton-linalg.git
cd triton-linalg/triton
```

- python 环境

```bash
conda create --name triton_env python=3.10 # 版本要大于等于3.8
conda activate triton_env
conda install numpy matplotlib pybind11 lit pytest isort pandas tabulate scipy flake8 autopep8
conda install pytorch torchvision # 我也不确定需不需要，反正cpu的不大
```

- 相关工具
按照 https://github.com/Cambricon/triton-linalg README 内容进行环境配置和编译

由于我是在 macOS 上编译，所以直接通过 `brew` 安装了相关工具
```bash
brew install cmake ninja-build ccache clang lld
conda install pytest-xdist cython # 记得要装cython
```

正常在 linux 下使用 `apt-get` 安装相关工具链即可
```bash
python3 -m pip install --upgrade pip
python3 -m pip install cmake ninja pytest-xdist cython # 这样装的cmake版本目前是3.26
sudo apt-get update -y
sudo apt-get install -y ccache clang lld
```

- 编译
```bash
# macos中lld是不能work的，所以不要添加相关的编译选项，在linux下就没问题
#TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true pip install -e python --no-build-isolation -vvv
pip3 install -e python --no-build-isolation
```

> note: 我用 macOs 编译的时候遇见编译报错  “找不到 `bits/std_abs.h`”。
> 翻了一下 macOS (`/Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/usr/include/`)的clang确实没有。
> 只需要把 `lib/Dialect/Triton/Transforms/InferAxisInfoInterfaceImpl.cpp` 中的 `bits/std_abs.h` 换成 `stdlib.h` 即正常编译

编译完成：
![编译成功](./img_Triton_linalg/success.png)

编译好的 `triton-linalg-opt` 在 `triton-linalg/triton/python/build/{current_cmake_version}/third_party/triton_linalg/bin/triton-linalg-opt` ，如果没有找到，说明没有设置环境变量 `export TRITON_PLUGIN_DIRS=$(pwd)/triton-linalg` 没有配置对，重新设置下再运行一次编译命令即可。

![opt](./img_Triton_linalg/opt.png)

## 测试使用

### 差生文具多

为了方便索引，编译完成后在 `triton-linalg/triton/python/build/` 目录下有一个 `compile_commands.json`，将其 `cp` 到 `triton-linalg`目录下， 再在 `vscode` 中简单地配置下
ctrl + p 输入 clangd，先点击 下载language server；然后 加 settings.json , ctrl + p → '> 打开工作区设置json’

```bash
{
    "clangd.arguments": [
        "--header-insertion=never",
        "--compile-commands-dir=${workspaceFolder}/",
        "--query-driver=**",
    ]
}
```

加个环境变量，方便使用
```bash
export PATH=xxx/triton-linalg/triton/python/build/{current_cmake_version}/third_party/triton_linalg/bin:$PATH
```

### 打印ir的方法

- 在kernel后增加

```python
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel = matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    print(kernel.asm['ttir'])
    print(kernel.asm['ttgir'])
    print(kernel.asm['llir'])
    print(kernel.asm['ptx'])
```

- 运行的时候加上 `MLIR_ENABLE_DUMP=1`

dumps the IR before every MLIR pass Triton runs

### 一窥ttir

`triton-llinalg-opt` 真正能吃下的输入并不是 python，而是 `ttir` (triton ir)，可以理解成一般性流程是 python -> ttit -> linalg / gpu dialect -> llvm


以 `tutorials/03-matrix-multiplication.py` 为例，输入：

```python
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

输出 ttir 时
- 地址相关计算会下降到 arith.ops

```llvm
module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    // blockarg对应：（都是根据ir推出来的）
    // %arg0: a_ptr, %arg1: b_ptr, %arg2: c_ptr
    // %arg3: M, %arg4: N, %arg5: K
    // %arg6: stride_am, %arg7: stride_bk, %arg8: stride_cm
    // 推出： stride_ak = 1, stride_bn = 1, stride_cn = 1

    // 常量部分，有些是这次 tuning 选择的 config 中的 超参数（tl.constexpr）具体值
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<64> : tensor<128x64xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32

    // pid = tl.program_id(axis=0)
    %0 = tt.get_program_id x : i32

    // %arg3: M, %arg4: N
    // num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    // (x + (2^n - 1)) / 2^n -> 实现向上取整
    // 这次 tuning config的 BLOCK_SIZE_M 是 128
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32

    // num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    // 这次 tuning config的 BLOCK_SIZE_N 是 32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32

    // num_pid_in_group = GROUP_SIZE_M * num_pid_n
    // 这次 tuning config的 GROUP_SIZE_M 是 8
    %5 = arith.muli %4, %c8_i32 : i32

    // group_id = pid // num_pid_in_group
    %6 = arith.divsi %0, %5 : i32

    // frist_pid_m = group_id * GROUP_SIZE_M
    %7 = arith.muli %6, %c8_i32 : i32

    // group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32

    // pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    // %11 = pid % group_size_m + first_pid_m 这里对不上 pid_m 的计算
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32

    // pid_n = (pid % num_pid_in_group) // group_size_m
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32

    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    %14 = arith.muli %11, %c128_i32 : i32
    // %15 = tl.arange(0, BLOCK_SIZE_M)
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %16 = tt.splat %14 : i32 -> tensor<128xi32>
    // %17 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    %17 = arith.addi %16, %15 : tensor<128xi32>
    %18 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %19 = arith.remsi %17, %18 : tensor<128xi32>

    // offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    %20 = arith.muli %13, %c64_i32 : i32
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %22 = tt.splat %20 : i32 -> tensor<64xi32>
    // %23 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    %23 = arith.addi %22, %21 : tensor<64xi32>
    %24 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %25 = arith.remsi %23, %24 : tensor<64xi32>

    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    %26 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    // %arg6: stride_am, splat 成同shape tensor，然后乘
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    // %29 = offs_k = tl.arange(0, BLOCK_SIZE_K)
    %29 = tt.expand_dims %21 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    // 这里是 stride_ak = 1
    %30 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32>
    %31 = tt.broadcast %29 : tensor<1x64xi32> -> tensor<128x64xi32>
    %32 = arith.addi %30, %31 : tensor<128x64xi32>
    // 把 a_ptr splat 成 对应 shape
    %33 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %34 = tt.addptr %33, %32 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>

    // b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    %35 = tt.expand_dims %21 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    // %arg7: stride_bk
    %36 = tt.splat %arg7 : i32 -> tensor<64x1xi32>
    %37 = arith.muli %35, %36 : tensor<64x1xi32>
    %38 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %39 = tt.broadcast %37 : tensor<64x1xi32> -> tensor<64x64xi32>
    // 这里是 stride_bn = 1
    %40 = tt.broadcast %38 : tensor<1x64xi32> -> tensor<64x64xi32>
    %41 = arith.addi %39, %40 : tensor<64x64xi32>
    %42 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
    %43 = tt.addptr %42, %41 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>

    // scf.for 循环上界 tl.cdiv(K, BLOCK_SIZE_K)
    // %arg5: K, 这次 tuning config的 BLOCK_SIZE_K 是 64
    %44 = arith.addi %arg5, %c63_i32 : i32
    %45 = arith.divsi %44, %c64_i32 : i32

    // %47 = BLOCK_SIZE_K * stride_bk
    %46 = arith.muli %arg7, %c64_i32 : i32
    %47 = tt.splat %46 : i32 -> tensor<64x64xi32>

    // for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)), %arg9->k
    // 每一轮都在改变：%arg10: accumulator, %arg11: %a_ptrs, %12: %b_ptrs
    %48:3 = scf.for %arg9 = %c0_i32 to %45 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %34, %arg12 = %43) -> (tensor<128x64xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x64x!tt.ptr<f16>>)  : i32 {
      // %67 = K - k * BLOCK_SIZE_K
      %66 = arith.muli %arg9, %c64_i32 : i32
      %67 = arith.subi %arg5, %66 : i32

      // a_mask计算 splat 后和 offsets_k 比较
      %68 = tt.splat %67 : i32 -> tensor<1x64xi32>
      %69 = arith.cmpi slt, %29, %68 : tensor<1x64xi32>
      %70 = tt.broadcast %69 : tensor<1x64xi1> -> tensor<128x64xi1>
      // tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
      %71 = tt.load %arg11, %70, %cst : tensor<128x64x!tt.ptr<f16>>

      // b_mask计算 splat 后和 offsets_k 比较
      %72 = tt.splat %67 : i32 -> tensor<64x1xi32>
      %73 = arith.cmpi slt, %35, %72 : tensor<64x1xi32>
      %74 = tt.broadcast %73 : tensor<64x1xi1> -> tensor<64x64xi1>
      // b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
      %75 = tt.load %arg12, %74, %cst_0 : tensor<64x64x!tt.ptr<f16>>

      // accumulator = tl.dot(a, b, accumulator)
      %76 = tt.dot %71, %75, %arg10, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>

      // a_ptrs += BLOCK_SIZE_K * stride_ak
      // 前面的代码推论出 stride_ak = 1，%cst_1 = arith.constant dense<64> : tensor<128x64xi32>，相当于 1x64(BLOCK_SIZE_K)
      %77 = tt.addptr %arg11, %cst_1 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>

      // b_ptrs += BLOCK_SIZE_K * stride_bk
      %78 = tt.addptr %arg12, %47 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>

      // 返回给次循环
      scf.yield %76, %77, %78 : tensor<128x64xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x64x!tt.ptr<f16>>
    }

    // c = accumulator.to(tl.float16)
    %49 = arith.truncf %48#0 : tensor<128x64xf32> to tensor<128x64xf16>

    // c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    // offs_cm = %17 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    %50 = tt.expand_dims %17 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    // %arg8: stride_cm, %52 = stride_cm * offs_cm[:, None]
    %51 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %52 = arith.muli %51, %50 : tensor<128x1xi32>
    // %arg2: c_ptr
    %53 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %54 = tt.addptr %53, %52 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    // offs_cn = %23 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    // 推断出 stride_cn = 1
    %55 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %56 = tt.broadcast %54 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
    %57 = tt.broadcast %55 : tensor<1x64xi32> -> tensor<128x64xi32>
    %58 = tt.addptr %56, %57 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>

    // c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    %59 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %60 = arith.cmpi slt, %50, %59 : tensor<128x1xi32>
    %61 = tt.splat %arg4 : i32 -> tensor<1x64xi32>
    %62 = arith.cmpi slt, %55, %61 : tensor<1x64xi32>
    %63 = tt.broadcast %60 : tensor<128x1xi1> -> tensor<128x64xi1>
    %64 = tt.broadcast %62 : tensor<1x64xi1> -> tensor<128x64xi1>
    %65 = arith.andi %63, %64 : tensor<128x64xi1>

    // tl.store(c_ptrs, c, mask=c_mask)
    tt.store %58, %49, %65 : tensor<128x64x!tt.ptr<f16>>
    tt.return
  }
}
```
