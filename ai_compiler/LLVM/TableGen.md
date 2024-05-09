# TableGen

目前在 `MLIR` 中使用 `TableGen` 的场景主要有注册 `Dialect`、`Operation`、`Pass`，并生成对应的 `.inc` 文件。 见[MLIR_Note中的tablegen使用](../MLIR/MLIR_Note.md#tablegen)

现在有个工作需要为 `RISCV` 扩展新的 `Intrinsic`，所以深入学习下 `TableGen` 的使用。

> 相关文件
> `llvm/include/llvm/IR/IntrinsicsRISCV.td`
> `llvm/lib/Target/RISCV/RISCVInstrInfo.td`
> ...

## 环境准备

先按照网上各种教程中编译的方法编译一下，主要注意以下 `cmake` 参数

- `-DLLVM_TARGETS_TO_BUILD` : 这里记得加上 `RISCV`
- `DLLVM_ENABLE_PROJECTS` : 我多加了点 "clang;mlir;compiler-rt"
- `DLLVM_CCACHE_BUILD` : ON
- `DCMAKE_EXPORT_COMPILE_COMMANDS` : ON

编译完成后可以在 `build` 目录下找到 `.td` 文件生成的东西，例如
`llvm/include/llvm/IR/IntrinsicsRISCV.td` 生成了 `build/include/llvm/IR/IntrinsicsRISCV.h` （在对应的 `llvm/include/llvm/IR/CMakeLists.txt` 中有描述关系）。

设置 `DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 后会在 `build` 目录下生成的`compile_commands.json` ，复制到 `llvm-project` 目录下(`mv build/compile_commands.json ./`)，然后配置vscode的clangd插件，方便索引文件：

ctrl + p 输入 clangd，先点击 下载language server；然后 加 settings.json , ctrl + p → '> 打开工作区设置json’

```json
{
    "clangd.arguments": [
        "--header-insertion=never",
        "--compile-commands-dir=${workspaceFolder}/",
        "--query-driver=**",
    ]
}
```

## TableGen基本语法
