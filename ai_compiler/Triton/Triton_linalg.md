# Triton-Linalg

这将是一篇长文（老太太牌裹脚布），大该有以下部分，但是都还在施工～

知识深度有限，欢迎大家指正

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

- linalg

了解 `mlir` 的同学一定不陌生 `linalg`，简单来说这是一个胶水层，能表示很多信息，起承上启下的作用，这就意味着硬件厂商支持 `mlir`的主要工作在 `linalg` 往下。

- triton

很多大佬都介绍过了，都写得很好，例如：
[bbuf大佬的笔记](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ)，
[董鑫​大佬关于如何入门的回答](https://www.zhihu.com/question/622685131/answer/3217107882)

一搜一个不吱声，直接埋头开卷！

简单来说，`triton` 可以让大家用更少的时间获得较为不错的性能，来验证自己的想法，深受现在学界的喜爱。当然工业界一些很好的 triton 工作了，例如 [lightllm](https://github.com/ModelTC/lightllm)中有很多用triton实现的kernel。


- triton-linalg

[triton-linalg](https://github.com/Cambricon/triton-linalg) 顾名思义，是**为triton(dialect)下降到linalg(dialect)提供了一条可行的路线**。如果大家看过 `triton` 的源码就会发现目前它的下降行为十分直接，一个猛子完成 `triton dialect->triton gpu dialect->llvm`(见[triton conversion](https://github.com/triton-lang/triton/tree/main/lib/Conversion))，在这些转换中分布着一些gpu硬件特有的trick来codegen出的ir性能不错。

“但是，古尔丹，代价是什么呢” -> 于我而言，代价是需要很多硬件背景知识才能读懂为什么要那么做，以及只能用在 GPU 上，为NV帝国添砖瓦，什么时候才能把价钱打下来！-> 期待早日见到国产卡重回潮头

![bqb1](./img_Triton_linalg/bqb1.jpg)

开始“龙场悟道“（自闭）：

那么有没有一种和硬件无关的层级表示 ir 能方便大家读懂且接入自己的硬件呢？
->
直接从 ttir(triton dialect ir) 接自己的 dialect(类似 TritonGPUDialect)?
->
那万一以后 `triton` 又不行了，出来一个其他的呢，又适配一遍么？
->
开摆！（x）看看业界领先经验（√）-> 跟紧 [mojo](https://github.com/modularml/mojo)拥抱 `mlir` 社区，而 `linalg` 作为 `mlir` 社区中很重要的一个中间层，可以考虑下。

### what can we do with this

- triton 重要性： triton 从 pytorch2.0 后已正式作为 `inductor` 的 gpu 后端，也就是说用户写到的 python 代码会经过 `inductor` 得到 `triton language`，然后经过编译后再执行，实现性能提升。接入 triton = 接入 pytorch = 走上人生巅峰 = 给别人埋bug...

> 感兴趣的同学可以了解下 [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

- 扩展性： linalg - to - HW special dialect
（借用一下大佬的图，来源见水印）
![triton_ext_pipeline](./img_Triton_linalg/mlir_pipeline.jpg)

- 中间层级优化：trion目前GPU的下降路线过于生硬，可以说是直接一把 `conversion`，一把下降会导致难以优化中间 IR（例如离散性优化），这对 `SIMT` 虽然影响不大，但是离散地访存行为对 `SIMD` 的影响无疑是巨大的。
以说是直接一把 `conversion`，一把下降会导致难以优化中间 IR（例如离散性优化），这对 `SIMT` 虽然影响不大，但是离散地访存行为对 `SIMD` 的影响无疑巨大

### triton-shared

[triton-shared](https://github.com/microsoft/triton-shared) 是 microsoft（巨硬）家实现 triton-to-linalg 的工作（以及实现以CPU作为后端），也扩展了特定的 Dialect。

### diff with triton-shared

- 下降行为不同
例如，对于指针的处理 `triton-shared` 会将指针转成memef<*xf32>，而 `triton-linalg` 会转为 `llvm.inttoptr`。 `triton-linalg` 对指针的处理保持了和 `triton` (官方一致)[https://github.com/triton-lang/triton/blob/main/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L792]。当然不同的行为都是为了自己的流程中更方便。

- balabalabal...

## 环境配置

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
主要下降为 tt.ops + arith.ops，具体参考 `triton/python/triton/language/semantic.py`

```llvm
module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
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

## 瞅瞅linalg

继续下到linalg上，一眼 `tensor` + `linalg` + `bufferization`，以及两个该仓库自定义的dialect `aux` + `linalg_ext`（后节会讲讲）

```llvm
// triton-linalg-opt -triton-to-linalg matmul.ttir
#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
module {
  func.func @matmul_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<64x64xf16>
    // arith.constant : tensor<128x64xf16> -> linalg.fill
    %1 = linalg.fill ins(%cst_0 : f16) outs(%0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %2 = tensor.empty() : tensor<128x64xi32>
    %3 = linalg.fill ins(%c64_i32 : i32) outs(%2 : tensor<128x64xi32>) -> tensor<128x64xi32>
    %4 = tensor.empty() : tensor<128x64xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
    // tt.get_program_id 在目前还没该变，可以考虑转为循环的上界
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %11 = arith.muli %10, %c8_i32 : i32
    %12 = arith.divsi %6, %11 : i32
    %13 = arith.muli %12, %c8_i32 : i32
    %14 = arith.subi %8, %13 : i32
    %15 = arith.minsi %14, %c8_i32 : i32
    %16 = arith.remsi %6, %15 : i32
    %17 = arith.addi %13, %16 : i32
    %18 = arith.remsi %6, %11 : i32
    %19 = arith.divsi %18, %15 : i32
    %20 = arith.muli %17, %c128_i32 : i32
    %21 = tensor.empty() : tensor<128xi32>
    // tt.make_range -> linalg_ext.make_range
    %22 = linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%c0_i32, %c128_i32 : i32, i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %23 = linalg.fill ins(%20 : i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %mapped = linalg.map { arith.addi } ins(%23, %22 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi32>)
    %24 = linalg.fill ins(%arg3 : i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %mapped_1 = linalg.map { arith.remsi } ins(%mapped, %24 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi32>)
    %25 = arith.muli %19, %c64_i32 : i32
    %26 = tensor.empty() : tensor<64xi32>
    %27 = linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%c0_i32, %c64_i32 : i32, i32) outs(%26 : tensor<64xi32>) -> tensor<64xi32>
    %28 = linalg.fill ins(%25 : i32) outs(%26 : tensor<64xi32>) -> tensor<64xi32>
    %mapped_2 = linalg.map { arith.addi } ins(%28, %27 : tensor<64xi32>, tensor<64xi32>) outs(%26 : tensor<64xi32>)
    %29 = linalg.fill ins(%arg4 : i32) outs(%26 : tensor<64xi32>) -> tensor<64xi32>
    %mapped_3 = linalg.map { arith.remsi } ins(%mapped_2, %29 : tensor<64xi32>, tensor<64xi32>) outs(%26 : tensor<64xi32>)
    %expanded = tensor.expand_shape %mapped_1 [[0, 1]] : tensor<128xi32> into tensor<128x1xi32>
    %30 = tensor.empty() : tensor<128x1xi32>
    %31 = linalg.fill ins(%arg6 : i32) outs(%30 : tensor<128x1xi32>) -> tensor<128x1xi32>
    %mapped_4 = linalg.map { arith.muli } ins(%expanded, %31 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%30 : tensor<128x1xi32>)
    %collapsed = tensor.collapse_shape %mapped_4 [[0, 1]] : tensor<128x1xi32> into tensor<128xi32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<128xi32>) outs(%2 : tensor<128x64xi32>) dimensions = [1]
    %broadcasted_5 = linalg.broadcast ins(%27 : tensor<64xi32>) outs(%2 : tensor<128x64xi32>) dimensions = [0]
    %mapped_6 = linalg.map { arith.addi } ins(%broadcasted, %broadcasted_5 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%2 : tensor<128x64xi32>)
    %expanded_7 = tensor.expand_shape %27 [[0, 1]] : tensor<64xi32> into tensor<64x1xi32>
    %32 = tensor.empty() : tensor<64x1xi32>
    %33 = linalg.fill ins(%arg7 : i32) outs(%32 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %mapped_8 = linalg.map { arith.muli } ins(%expanded_7, %33 : tensor<64x1xi32>, tensor<64x1xi32>) outs(%32 : tensor<64x1xi32>)
    %collapsed_9 = tensor.collapse_shape %mapped_8 [[0, 1]] : tensor<64x1xi32> into tensor<64xi32>
    %34 = tensor.empty() : tensor<64x64xi32>
    %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<64xi32>) outs(%34 : tensor<64x64xi32>) dimensions = [1]
    %broadcasted_11 = linalg.broadcast ins(%mapped_3 : tensor<64xi32>) outs(%34 : tensor<64x64xi32>) dimensions = [0]
    %mapped_12 = linalg.map { arith.addi } ins(%broadcasted_10, %broadcasted_11 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%34 : tensor<64x64xi32>)
    %35 = arith.addi %arg5, %c63_i32 : i32
    %36 = arith.divsi %35, %c64_i32 : i32
    %37 = arith.muli %arg7, %c64_i32 : i32
    %38 = linalg.fill ins(%37 : i32) outs(%34 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %39 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %40 = tensor.empty() : tensor<128x64xf16>
    %41 = tensor.empty() : tensor<64x1xi1>
    %42 = tensor.empty() : tensor<64x64xi1>
    // tt.addptr -> llvm.inttoptr
    %43 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %44:3 = scf.for %arg9 = %c0_i32 to %36 step %c1_i32 iter_args(%arg10 = %5, %arg11 = %mapped_6, %arg12 = %mapped_12) -> (tensor<128x64xf32>, tensor<128x64xi32>, tensor<64x64xi32>)  : i32 {
      %71 = arith.muli %arg9, %c64_i32 : i32
      %72 = arith.subi %arg5, %71 : i32
      %73 = arith.index_cast %72 : i32 to index
      %74 = arith.maxsi %73, %c0 : index
      %75 = arith.minsi %74, %c64 : index
      %view_memref_14 = aux.view %39 to offset: [0], sizes: [9223372036854775807], strides: [1] : !llvm.ptr to memref<9223372036854775807xf16>
      %extracted_slice_15 = tensor.extract_slice %arg11[0, 0] [128, 1] [1, 1] : tensor<128x64xi32> to tensor<128x1xi32>
      %76 = linalg.fill ins(%c0_i32 : i32) outs(%30 : tensor<128x1xi32>) -> tensor<128x1xi32>
      %mapped_16 = linalg.map { arith.addi } ins(%extracted_slice_15, %76 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%30 : tensor<128x1xi32>)
      %77 = bufferization.to_tensor %view_memref_14 restrict writable : memref<9223372036854775807xf16>
      %78 = tensor.empty(%75) : tensor<128x?xf16>
      // 带mask的访存行为
      %79 = linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(false) ins(%77, %mapped_16 : tensor<9223372036854775807xf16>, tensor<128x1xi32>) outs(%78 : tensor<128x?xf16>) {
      ^bb0(%arg13: f16, %arg14: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<128x?xf16>
      %80 = arith.subi %c64, %75 : index
      %81 = linalg_ext.pad ins(%79 : tensor<128x?xf16>) outs(%40 : tensor<128x64xf16>) pvalue(%cst_0 : f16) low = [0, 0] high = [0, %80] {
      ^bb0(%arg13: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<128x64xf16>
      %82 = tensor.empty(%75) : tensor<?x1xi1>
      %83 = linalg.fill ins(%true : i1) outs(%82 : tensor<?x1xi1>) -> tensor<?x1xi1>
      // pad为static方便处理
      %84 = linalg_ext.pad ins(%83 : tensor<?x1xi1>) outs(%41 : tensor<64x1xi1>) pvalue(%false : i1) low = [0, 0] high = [%80, 0] {
      ^bb0(%arg13: i1):
        linalg_ext.yield %arg13 : i1
      } -> tensor<64x1xi1>
      %collapsed_17 = tensor.collapse_shape %84 [[0, 1]] : tensor<64x1xi1> into tensor<64xi1>
      %broadcasted_18 = linalg.broadcast ins(%collapsed_17 : tensor<64xi1>) outs(%42 : tensor<64x64xi1>) dimensions = [1]
      %85 = linalg.fill ins(%c0_i32 : i32) outs(%34 : tensor<64x64xi32>) -> tensor<64x64xi32>
      %mapped_19 = linalg.map { arith.addi } ins(%arg12, %85 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%34 : tensor<64x64xi32>)
      %view_memref_20 = aux.view %43 to offset: [0], sizes: [9223372036854775807], strides: [1] : !llvm.ptr to memref<9223372036854775807xf16>
      %extracted_slice_21 = tensor.extract_slice %mapped_19[0, 0] [%75, 64] [1, 1] : tensor<64x64xi32> to tensor<?x64xi32>
      %expanded_22 = tensor.expand_shape %extracted_slice_21 [[0], [1, 2]] : tensor<?x64xi32> into tensor<?x64x1xi32>
      %extracted_slice_23 = tensor.extract_slice %broadcasted_18[0, 0] [%75, 64] [1, 1] : tensor<64x64xi1> to tensor<?x64xi1>
      %86 = bufferization.to_tensor %view_memref_20 restrict writable : memref<9223372036854775807xf16>
      %87 = tensor.empty(%75) : tensor<?x64xf16>
      %expanded_24 = tensor.expand_shape %87 [[0], [1, 2]] : tensor<?x64xf16> into tensor<?x64x1xf16>
      %88 = linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(false) ins(%86, %expanded_22, %extracted_slice_23 : tensor<9223372036854775807xf16>, tensor<?x64x1xi32>, tensor<?x64xi1>) outs(%expanded_24 : tensor<?x64x1xf16>) {
      ^bb0(%arg13: f16, %arg14: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<?x64x1xf16>
      %collapsed_25 = tensor.collapse_shape %88 [[0], [1, 2]] : tensor<?x64x1xf16> into tensor<?x64xf16>
      %89 = linalg_ext.pad ins(%collapsed_25 : tensor<?x64xf16>) outs(%0 : tensor<64x64xf16>) pvalue(%cst_0 : f16) low = [0, 0] high = [%80, 0] {
      ^bb0(%arg13: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<64x64xf16>
      %mapped_26 = linalg.map { arith.select } ins(%broadcasted_18, %89, %1 : tensor<64x64xi1>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%0 : tensor<64x64xf16>)
      // tt.dot -> linalg.matmul
      %90 = linalg.matmul {__allow_tf32__} ins(%81, %mapped_26 : tensor<128x64xf16>, tensor<64x64xf16>) outs(%arg10 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mapped_27 = linalg.map { arith.addi } ins(%arg11, %3 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%2 : tensor<128x64xi32>)
      %mapped_28 = linalg.map { arith.addi } ins(%38, %arg12 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%34 : tensor<64x64xi32>)
      scf.yield %90, %mapped_27, %mapped_28 : tensor<128x64xf32>, tensor<128x64xi32>, tensor<64x64xi32>
    }
  ...
  }
}
```

## 占坑

## dialect

## analysis

## conversion

## pipeline

