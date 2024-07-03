# Triton-Linalg

doing
- [ ] 介绍（背景、优缺点、和triton-shared的区别）
- [x] 环境配置 : clone & 编译
- [ ] 测试使用（测试一些例子，看pass）
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

加个环境变量
```bash
export PATH=xxx/triton-linalg/triton/python/build/{current_cmake_version}/third_party/triton_linalg/bin:$PATH
```

输入 IR

```llvm
// 还没想好
```