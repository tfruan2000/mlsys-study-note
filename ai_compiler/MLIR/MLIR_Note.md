# [MLIR] Code Note

## 工欲善其事，必先利其器

### 推荐项目

- 学习 mlir 的项目

[mlir-tutorial](https://github.com/j2kun/mlir-tutorial) 使用 `bazel` 构建项目，相比 `cmake` 构建个人感觉更适合新手。

- 可以抄代码的项目

[IREE](https://github.com/iree-org/iree) 架构、风格上很有 Google 的风范。

[ByteIR](https://github.com/bytedance/byteir) ，字节开源项目。

### 跳转工具clangd

`vscode` 专属。

1.首先我们需要生成 `compile_commands.json`，以编译 `llvm` 为例：

- 如果是cmake编译

```bash
# mac上编译mlir
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_BUILD_EXAMPLES=ON

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

生成的 `compile_commands.json` 在 `build` 目录下，复制到llvm-project目录即可

- 如果是bazel编译

在BUILD文件配置一下下面的内容，再bazel run 一下就可以编译出compile_commands.json
详情自学：[https://github.com/hedronvision/bazel-compile-commands-extractor/tree/main](https://github.com/hedronvision/bazel-compile-commands-extractor/tree/main)

(1) 修改WORKSPACE，添加

```bash
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "hedron_compile_commands",

    # 记得把下面两处 commit hash 换成 github 上最新的版本
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

```

(2) 在根目录下的 `BUILD.bazel` 中添加下面语句

```bash
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # 指定目标 target 及其编译选项/参数，例如 `mlir-opt` 、`config=clangd`
    targets = {
      "//:my_output_1": "--important_flag1 --important_flag2=true"
    },
)
```

(3) 运行 `bazel run :refresh_compile_commands`

2.然后，配置vscode的clangd插件

`ctrl + p` 输入 clangd，先点击 下载language server；然后 加 settings.json , `ctrl + p` 打开工作区设置json，将以下内入加入

```cpp
{
    "clangd.arguments": [
        "--header-insertion=never",
        "--compile-commands-dir=${workspaceFolder}/",
        "--query-driver=**",
    ]
}
```

使用compile_commands.json主要是方便索引文件，特别是td生成的 `inc` 文件，但也可以人为从 `build/tools/mlir/include/mlir/xxx/xxx` 中找到编译出的 `inc`。

### 代码格式

一般使用 `clang-format` 工具(或者基于此的 `lint.sh`)。

安装
```bash
apt-get install clang-format
```

创建`.clang-format`
```bash
BasedOnStyle: LLVM
ColumnLimit: 80
```

格式化
```bash
# 单个文件
clang-format -i path/to/your/file.cpp
# 整个目录
find path/to/your/project -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

---

## Adaptor

只有**operands没有results**的中间态，可以从adaptor中获得很多基础信息

`ConversionPattern` 相比 `RewriterPattern` 需要多传递一个 `adaptor`

1.OpAdaptor的作用：封装了op的operands

2.ConversionPattern和RewritePatter的区别

- ConversionPattern常配合 **applyFullConversion/applyPartialConversion** 使用，用于dialect2dialect的op之间变换

- RewritePattern一般用于优化变换，常配合 **applyPatternAndFoldGreedily** 使用

```cpp
// OpConversionPattern
struct AbsOpToMathAbsConverter : public OpConversionPattern<mhlo::AbsOp> {
  using OpConversionPattern<mhlo::AbsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
...

// OpRewritePattern
struct TransposeSliceLayoutPattern : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mhlo::SliceOp op,
                  OpRewritePattern &rewriter) const override {
```

---

## Analysis

[[MLIR] Analysis](./composition/Analysis.md ':include')

---

## Attribute

```cpp
mlir/include/mlir/IR/Attribute.h
```

常见类型：

- StringAttr
- UnitAttr / IntegerAttr / IndexAttr
- BoolAttr
- ArrayAttr
- DictionaryAttr

常用方法：

1.使用 `OpBuilder` 可以创建这类 `Attr`，例如
rewriter.getI64IntegerAttr 或builder.getI64IntegerAttr。

2.src: AttrTy
- get() 例如从SmallVector<Attribute>变成ArrayAttr
  ```cpp
  SmallVector<Attribute, 8> mappings;
  ArrayAttr tmp = ArrayAttr::get(context, mappings)
  ```

- getName()

- setValue()

- getValue() 对于IntegertAttr会返回APInt，之后一般可以接 `getSExtValue()` ，来将APInt转为**int64_t**

- src : operation*
  - getAttr / getAttrOfType ，一般get完之后要cast到对应的AttrType，例如
    ```cpp
    op->getAttr(getAliasAttrKey()).dyn_cast_or_null<mlir::IntegerAttr>()
    op->getAttrOfType<DenseI64ArrayAttr>
    ```

  - hasAttr / hasAttrOfType

  - setAttr(StringRef name, Attribute value)
    - name可以`constexpr llvm::StringLiteral` 形式定义在头文件中
    - funcOp→setAttr(attrName, IntegerAttr::get(intType, 1));

  - removeAttr

  - func::FuncOp::setResultAttr

### operation、attribute、type关系

| 专用指针      | 通用指针     | 值               |
|-------------|-------------|------------------|
| AddOp       | Operation * | Operation        |
| IntegerType | Type        | TypeStorage      |
| IntegerAttr | Attribute   | AttributeStorage |

---

## Block

```cpp
mlir/include/mlir/IR/Block.h
```

Block 包含 `BlockArgument`（使用getArguements()获得）和 `BlockOperand`

### BlockArgument

继承自 `Value`。

`Block *getOwner()` 返回该arg属于哪个block。

`unsigned getArgNumber()` 返回该arg的index。

### BlockOperand

继承自`IROperand`。

`unsigned getOperandNumber()` 返回该operand的index。

### 使用

1.返回 Block

- `Operation *` -> `getBlock()`
- `Value` -> `getParentBlock()`


2.遍历block
- walk
  ```cpp
  block->walk([&](Operation *op) {...
  ```

- 只遍历同层op
  ```cpp
  Operation &workOp : rootBlock->getOperations()
  ```

---

## Builder

[[MLIR] Builder](./composition/Builder.md ':include')

---

## Bufferize

### bufferization dialect

bufferization：将逻辑计算语义的tensor转为物理内存语义的buffer

- bufferize::AllocTensorOp

申请一块空间，使用给定shape创建一个bufferize allocation。常会传入一个可选的 srcOp，表示从这个srcOp拷贝出的数据，此时就传入的 ValueRange dynamicShape就应为空。

该op主要是帮助bufferization过程提供一个 `handler`，并且这样产生的alloc_tensor op没有不会产生 read-after-write 冲突，也不会alias其他buffer，可以再进行 `in-place bufferize`

### one-shot-bufferize

（copy from大佬）

```cpp
mlir/lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp
```

1） OneShotBufferize pass

对于每个有 `BufferizableOpInterface` 的op都进行bufferize

- 声明：mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td
    - 1. 先基于tensor的SSA use-def链进行原位分析来确认哪些operand可以**in-place bufferize**.（尽量减少alloc和copy, 提高性能）
        - destination-passing style op（继承`DestinationStyleOpInterface` ）： 某一个**operand和dst的buffer可复用**，所以分配了该operand的buffer后，无需再分配dst的buffer：eg: %t0 = tensor.insert %f into %dest[%idx]， buffer(%t0)和buffer(%dest)是完全一致的；
        - 非destination-passing style op：对每个OpOperand产生一个新的buffer allocation, eg：tensor.generate
        - 所有新allocate的buffer后续都会deallocate，不然会内存泄露
    - 2.TensorCopyInsertion：对确定是**out-of-place的operands插入 copies**，insertTensorCopies()函数。
    - 3. 调用bufferize接口bufferize()函数来实现bufferize. bufferizeOp()函数。
    - 4. 函数签名的layout map由`function-boundary-type-conversion`选项单独控制，可选的参数有3种：`infer-layout-map`，`fully-dynamic-layout-map` and `identity-layout-map`， 默认是`infer-layout-map`。无法精确推测时，函数参数类型为fully dynamic layout maps。
    - 5.  `bufferize-function-boundaries` 是一个用来对funcOp、returnOp、callOp进行bufferize的flag
    - 6.funcArg一般可以bufferize，除非有 `bufferization.writable = false`
- 实现：mlir/lib/Dialect/Bufferization/Transforms/Bufferize.cpp
    - struct OneShotBufferizePass {void runOnOperation() override }
        - Configure type converter， 先获得 unknownTypeConversionOption：
            - 若是LayoutMapOption::IdentityLayoutMap， bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace)；
            - 否则，只能是LayoutMapOption::FullyDynamicLayoutMap，bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,memorySpace);
        - Configure op filter. 依据编译选项设置不可bufferize的op
        - 依据编译选项是否激活bufferizeFunctionBoundaries确定调用哪个函数进行bufferize:
            - 若激活了，runOneShotModuleBufferize(moduleOp, opt, &statistics)
            - 反之，runOneShotBufferize(moduleOp, opt, &statistics)
        - createCanonicalizerPass()
        - createCSEPass()
        - createLoopInvariantCodeMotionPass()
- 示例：mlir/test/Dialect/Bufferization/Transforms/one-shot-module-bufferize-out-params.mlir, mlir/test/Dialect/Bufferization/Transforms/one-shot-module-bufferize.mlir

2) transform IR : transform.bufferization.one_shot_bufferize 有很多可选的参数

- layout{IdentityLayoutMap} { bufferize_function_boundaries = true }
- {bufferize_function_boundaries = true }
- 定义：mlir/include/mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.td
- 实现：transform.bufferization.one_shot_bufferize的代码：
    - mlir/lib/Dialect/Bufferization/TransformOps/BufferizationTransformOps.cpp: transform::OneShotBufferizeOp::apply()函数，从transform IR提供的各个参数中获得OneShotBufferizationOptions options，之后主要调用
        - runOneShotModuleBufferize()
            - insertTensorCopies(moduleOp, options)
            - bufferizeOp() 会调用`BufferizableOpInterface::bufferize()`函数来对每个op进行具体的bufferize
        - runOneShotBufferize()
            - insertTensorCopies(target, options)
            - bufferizeOp() 会调用`BufferizableOpInterface::bufferize()`函数来对每个op进行具体的bufferize
- 示例：mlir/test/Dialect/Bufferization/Transforms/transform-ops.mlir

```mlir
// 编译命令：mlir-opt --test-transform-dialect-interpreter
func.func @matmul(%A: tensor<12x9xf32>, %B: tensor<9x6xf32>, %C: tensor<12x6xf32>) -> tensor<12x6xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<12x9xf32>, tensor<9x6xf32>) outs(%C: tensor<12x6xf32>) -> tensor<12x6xf32>
  return %D : tensor<12x6xf32>
}
// use identity layout at function boundaries.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
  transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg1 {bufferize_function_boundaries = true }
}
// result is 连续的memref
func.func @matmul(%arg0: memref<12x9xf32>, %arg1: memref<9x6xf32>, %arg2: memref<12x6xf32>) -> memref<12x6xf32> {
  linalg.matmul ins(%arg0, %arg1 : memref<12x9xf32>, memref<9x6xf32>) outs(%arg2 : memref<12x6xf32>)
  return %arg2 : memref<12x6xf32>
}
// use default at function boundaries.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
  transform.bufferization.one_shot_bufferize %arg1 {bufferize_function_boundaries = true }
}
// result is 非连续的memref(所有func.func的args和返回值均是非连续的）
func.func @matmul(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>, %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>, %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) -> memref<12x6xf32, strided<[?, ?], offset: ?>> {
  linalg.matmul ins(%arg0, %arg1 : memref<12x9xf32, strided<[?, ?], offset: ?>>, memref<9x6xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return %arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>
}
```

---

## Conversion

形式：将写好的pattens加入RewriterPatternSet并设置benefit，再apply

```cpp
void runOnOperation() override {
	RewritePatternSet patterns(&getContext());
	patterns.add<xxxx>(patterns.getContext(), /*benefit*/2)
	if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))));
		return signalPassFailure();
}
```

常见的apply形式:

- `applyPartialConversion` ：如果结果是合法（以`ConversionTarget`参数来判断）则保留，如果非法则报错
- `applyFullConversion` ：调用pattern对目标进行转换，直至IR满足`ConversionTarget`设置的目标合法，pattern必须成功才会产生合法的target
- `applyPatternAndFoldGreedily`：尽可能地多次修改，pattern可以失败

前两种常用于dialect conversion，需要多传入一个`ConversionTarget`参数，greedilyConversion一般用于优化pass

### ConversionTarget

常用定义op

```cpp
MLIRContext &ctx = getContext();
ConversionTarget target(ctx);
target.addIllegalDialect<SparseTensorDialect>();
target.addLegalDialect
target.addDynamicallyLegalDialect
target.addLegalOp
target.addDynamicallyLegalOp
```

例如只对标量op进行转换的pattern
```cpp
target.markUnknownOpDynamicallyLegal([](Operation *op) {
	if (isa<math::MathDialect>(op->getDialect()) &&
			llvm::isa<math::LogOp, math::ExpOp,...>(op)) {
	   return op->getResultTypes().front().isa<ShapedType>();
  }
  return true;
});

RewritePatternSet patterns(&ctx);
patterns.add<xxx>(patterns.getContext());
if(failed(applyParticalCpnversion(getOperation(), target,
																	std::move(patterns))))
	return signalPassFailure();
```

ConversionPattern相比RewriterPattern一般多一个[adaptor](#adaptor)参数，用于访问op的opernads

```cpp
// 常用于op的dialect2dialect下降
struct AbsOpToMathAbsConverter : public OpConversionPattern<mhlo::AbsOp> {
  using OpConversionPattern<mhlo::AbsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

// 常用于op的优化行为，也可以用于dialect2dialect中的op下降
struct TransposeSliceLayoutPattern : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mhlo::SliceOp op,
                  PatternRewriter &rewriter) const override {
```

---

### dialect conversion

```cpp
mlir/lib/Transforms/Utils/DialectConversion.cpp
```

Value ConversionValueMapping::lookupOrDefault(Value from, Type desiredType)

即dialect_a中的op对应转换到dialect_b中，例如vector dialect → gpu dialect

dialect conversion一般包含op conversion和type conversion

### op conversion

1.OpRewritePattern

```cpp
mlir/include/mlir/IR/PatternMatch.h
```

以vector2gpu为例

```cpp
// mlir/lib/Conversion/ArithToSPIRV/ArithToSPIRV.cpp
// namespace内定义许多op conversion patterns
namespace{
struct ConstantCompositeOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OperationConversionPattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                opAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
	...
	}
}
...
void mlir::populateArithToSPIRVPatterns(RewritePatternSet &patterns) {
	patterns.add<ConstantCompositeOpPattern>(patterns.getContext());
	// 可以设置pattern的/*benefit=*/
	// patterns.add<ConstantCompositeOpPattern>(patterns.getContext(), /*benefit=*/2);
	...
}
} // namespace
```

### type conversion

```cpp
mlir/Conversion/LLVMCommon/TypeConverter.h
```

对type对改写一般通过 `typeConverter` ，常配合 `ConversionTarget` 使用。其一般包含三个主要函数

- `addConversion` ：定义type转换规则

例如
```cpp
typeConverter converter;
converter.addConversion([&]ToyIntegerType t) -> std::optional<Integer> {
	return Integer::get(&getContext(), t.getWidth())
}
```

- `addTargetMaterialization` ：sourceType→targetType
- `addSourceMaterialization` ：targetType→sourceType
- `addArgumentMaterialization`

```cpp
static Value materializeToXXXCallback(OpBuilder &builder, Type type, ValueRange values) {
  if (xxx)
    ...
  return nullptr;
}

class MyTypeConvert : public TypeConverter {
public:
  MyTypeConvert() {
    addConversion([](Type type)) -> Type {
      if (isSomeType(type))
        return ...;
      return type;
    });
  }

  addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange values) {
    if (...)
      return builder.create<SomeOp>(type, values);
    return nullptr;
  });

  addSourceMaterialization(materializeToXXXCallback);
  addArgumentMaterialization(materializeToXXXCallback);
}
```

---

## Dataflow

MLIR中的数据流图是由Operation和Value构成的：（use-def chain）

- Value的值要么来自于Operation的result，要么来自于BlockArgument
    - 调用getDefiningOp时，BlockArgument会返回null

- 每个Operation的Operand都是到Value的指针

Operation都包含Results和Operands；Results中包含多个OpResult实例，Operands中包含多个OpOperand实例

<div style="text-align: center;"><img src="./img_MLIR_Note/Untitled.png" alt="Untitled" style="width: 90%;"></div>

- 修改Operand时，实际是修改OpOperand，对应value的use-chain也会被修改

<div style="text-align: center;"><img src="./img_MLIR_Note/Untitled%201.png" alt="Untitled" style="width: 90%;"></div>

### Operation找Value

- getOperands() / getResults()

```cpp
for (auto operand : op.getOperands()) {
	if (auto *def = op.getDefiningOp()) {
	} else {
		// BlockArgument
	}
}
```

- getOpOperands() 用于需要修改operand时

```cpp
IRMapping mapping;
mapping().map(op1.getResults(), op2.getResults());
for (auto &opOperand : op3.getOpOperands()) {
	// 将 op3 的参数里含有 op1 results 的替换为 op2 的
  // lookupOrDefault 指找不到 mapping 就用原来的
  opOperand.set(mapping.lookupOrDefault(opOperand.get()));
}
```

### value找op

- getDefiningOp：可能返回nul

- getUses ：返回OpOperand迭代器，即使用了这个value的OpOperand集合
  - OpOperand &operand : value.getUses()

- getUsers ：返回Operation迭代器，即直接依赖于该value的operation集合
  - user_iterator相当于对use_iterator使用getOwner()
  - use.getOwner() → Operation*

### dataflow framework

见 [[MLIR] Analysis](./composition/Analysis.md ':include') 中 dataflow framework 节

---

## DataType

```cpp
mlir/include/mlir/IR/BuiltinTypes.h
```

从ShapedType使用getElementType()获得

类型：

- FloatType
    - getF32
    - getWidth
- IndexType ：target word-size integer
- IntegerType

用法

- 判断类型
    - isInteger
        - isInteger(unsigned width)
    - isIndex
    - isIntOrIndex
    - isIntOrFloat
- 生成 get
    -  RankedTensorType::get(ArrafRef<int64_t> shapes, elemType)
      例如 RankedTenorType newType = RankedTensorType::get({srcDims[0], 1}), srcType.getElementType)
    - IntegerType::get(op→getContext(), 64);

---

## Debug

```cpp
#include "llvm/include/llvm/Support/Debug.h"
LLVM_DEBUG(llvm::dbgs() << "Original loop:\n"
                        << *region->getParentOp() << "\n");
LLVM_DEBUG(llvm::dbgs() << "Checking op: " << *op << "\n");
```

---

## Dianostic

```bash
mlir/docs/Diagnostics.md
mlir/include/mlir/IR/Diagnostics.h
mlir/lib/IR/Diagnostics.cpp
```

当rewrite-pattern使用op的verify(rewrite出的op是否合法)来判断pattern是否match-and-rewrite成功时，那apply-pattern时的报错就是不必要的，可以通过去除handler的办法消除掉这些不必要的报错

使用
```cpp
auto *context = &getContext();
auto handlerID =
    context->getDiagEngine().registerHandler([](Diagnostic &) { return; });
...
RewritePatternSet patterns(context);
patterns.add<xxx>(patterns.getContext());
(void)applyPatternAndFoldGreedily(getOperation(), std::move(patterns));
...
context->getDiagEngine().eraseHandler(handlerID);
```

---

## Dialect

[[MLIR] Dialect](./composition/Dialect.md ':include')

---

## Diagnostic

```cpp
mlir/include/mlir/IR/Diagnostics.h
```

---

## Dominance

```cpp
mlir/include/mlir/IR/Dominance.h
mlir/lib/IR/Dominance.cpp
```

### DominanceInfo

`class DominanceInfo : public detail::DominanceInfoBase</*IsPostDom=*/false>`

返回一个 region-like 的 dominance

常见函数

下面的 `T` 可以是 `Operation *` 或 `Value *`

- bool dominates(T *a, Operation *b): 判断a是否支配b
  - 如果a是Operation，则返回 `a == b || properlyDominates(a, b)`
  - 如果a是Value，则返回 `(Operation *)a.getDefiningOp() == b || properlyDominates(a, b)`


- Bool properlyDominates(T *a, Operation *b)
  - 如果a是Operation，则直接调用 properlyDominatesImpl
  - 如果a是Value，且a是BlockArgument，则`dominates(blockArg.getOwner(), b->getBlock());`，反之properlyDominates((Operation *)a.getDefiningOp(), b)

- bool hasSSADominance(Block *block) -> hasSSADominance(block->getParent())
- bool hasSSADominance(Region *region)
  - 判断region中的ops是否都满足SSA支配关系
  - 如果region中不满足，则无法分析出dominanceInfo，遍历order需要修改

- DominanceInfoNode *getRootNode(Region *region)
    - 获得给定region的root dominance node，输入的region必须有多block

### DominanceInfoNode

支配树节点 `llvm::DomTreeNodeBase<Block>`

有一个 `SmallVector<DomTreeNodeBase *, 4> Children;`

- begin() / end() 都是以 Children 为对象。下面的代码是CSE pass遇见多block的region的遍历行为

```cpp
  // currentNode->node 就是 DominanceInfoNode，即 llvm::DomTreeNodeBase<Block>
  while (!stack.empty()) {
    auto &currentNode = stack.back();
    // 检查当前node是否被处理
    if (!currentNode->processed) {
      // 处理该节点
      currentNode->processed = true;
      // getBlock() 会返回当前Block
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
    }
    // 遍历该node的子节点
    if (currentNode->node->begin() != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // 如果当前节点和其子节点都被处理了，移除它
      stack.pop_back();
    }
  }
```



---

## Func

所有dialect的funcOp都继承了  `FunctionOpInterface` , 该interface提供许多方法来获得func的信息

- Type getFunctionType()
- Region **&**getFunctionBody()
- BlockArgListType getArguments()

---

## IRMapping

```cpp
mlir/include/mlir/IR/IRMapping.h
```

用法：

- 构造
    - map(Value from, Value to)
    - map(Block *from, Block *to)
    - map(Operation *from, Operation *to)
- lookupOrValue(from, value);
- lookupOrDefault <=> lookupOrValue(from, from);

```cpp
IRMapping mapping;
mapping().map(op1.getResults(), op2.getResults());
for (auto &opOperand : op3.getOpOperands()) {
	// 将 op3 的参数里含有 op1 results 的替换为 op2 的
  // lookupOrDefault 指找不到 mapping 就用原来的
  opOperand.set(mapping.lookupOrDefault(opOperand.get()));
}
```

- lookupOrNull  <=>  lookupOrValue(from, nullptr);

---
## Interface

[[MLIR] Interface](./composition/Interface.md ':include')

---

## llvm

常用

### LogicalResult
- 函数内用 success() / failure() 作为返回
- 外部调用判断 succeeded / failed

```cpp
static LogicalResult getXXX(Value &res, ...) {
  if (xxx) {
    value = newValue
    return success();
  }
  return failure();
}

// 使用
Value res = nullptr;
LogicalResult a = getXXX(res, ...);
if (failed(a)) {
  DBGS() << "Get value failed.";
}
```

但如果不想判断该函数是否 successed，或直接使用返回值(某些数据类型直接返回开销不大，没必要传入引用)，可以考虑使用 `FailureOr<Ty>`

```cpp
static FailureOr<Value> getXXX(...) {
  if (xxx) {
    return newValue;
  }
  return failure();
}

// 使用
FailureOr a = getXXX(res, ...);
if (failed(a))
  DBGS() << "Get value failed.";
Value res = *a
```

### isa
isa : 不能在cast之后使用isa，有类似场景请用dyn_cast
isa_and_nonnull / isa_and_present：允许op为null作为输入，返回null
> 两者作用完全相同，未来的llvm将弃用 `isa_and_nonnull` ，推荐使用 `isa_and_present`

### cast
- cast ：直接转换，失败时报错
- cast_or_null ：允许op为null作为输入，返回null
- dyn_cast ： 尝试转换，失败时返回null，op为null时报错
- dyn_cast_if_present / dyn_cast_or_null ：尝试转换，失败时返回null，op为null时返回null
    ```cpp
    template <class X, class Y> auto dyn_cast_or_null(const Y &Val) {
      return dyn_cast_if_present<X>(Val);
    }
    ```
    例子：
    ```cpp
    Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b,
                                                Location loc,
                                                OpFoldResult ofr) {
      if (auto value = llvm::dyn_cast_if_present<Value>(ofr))
        return value;
      auto attr = dyn_cast<IntegerAttr>(llvm::dyn_cast_if_present<Attribute>(ofr));
      assert(attr && "expected the op fold result casts to an integer attribute");
      return b.create<arith::ConstantIndexOp>(loc, attr.getValue().getSExtValue());
    }
    ```

### all_of / any_of / for_each
```cpp
llvm/include/llvm/ADT/STLExtras.h
```

- all_of ：判断是否所有元素都满足条件
    - 用法
        ```cpp
        if (llvm::all_of(inputs, [](Value input) {
          return input.getType().isa<ShapedType>();
        });
        ```
    - 实现
        ```cpp
        template <typename R, typename UnaryPredicate>
        bool all_of(R &&Range, UnaryPredicate P) {
          return std::all_of(adl_begin(Range), adl_end(Range), P);
        }
        ```

- any_of ：判断是否有元素满足条件
    - 用法
        ```cpp
        	if (llvm::any_of(op->getRegions(), [](Region &r) {
        			return r.getBlocks().size > 1;
        		}))
        	return failure();
        ```
    - 实现
        ```cpp
        template <typename R, typename UnaryPredicate>
        bool any_of(R &&Range, UnaryPredicate P) {
          return std::any_of(adl_begin(Range), adl_end(Range), P);
        }
        ```

- none_of ：判断是否没有元素满足条件

- for_each ：对每个元素执行操作

- llvm::enumerate
    - 返回一个pair，first是index，second是value，直接对元素使用 `.index()` 和 `.value()` 即可
    - 也可以使用 `auto [idx, val] : llvm::enumerate(inputs)` / `auto [idx, val1, val2] : llvm::enumerate(inputs, outputs)`
    - 用法
        ```cpp
        auto isConsecute [](ArrayRef<int64_t> array) -> bool {
          return llvm::all_of(llvm::enumerate(array), [array](auto iter)) {
            return iter.index() + array.front() == iter.value();
          });
        };
        ```

- llvm::zip

	- 遍历时使用 `std::get<0>` 和 `std::get<1>` 来获得值
  ```cpp
  for (const auto &it :llvm::enumerate(llvm::zip(valAVec, valBVec))) {
	    Value aVal = std::get<0>(it.value());
	    Value bVal = std::get<1>(it.value());
	```

  - 比较
  ```cpp
  for (const auto &[operand, arg] :
       llvm::zip(op->getOperands(), body->getArguements())) {
    if (operand != arg)
      return failure();
  }
  ```

### function

- llvm:function_ref 定义inline func，用于传递函数指针

`using function_ref = llvm::function_ref<Fn>`

```cpp
size_t mlir::moveLoopInvariantCode(
    ArrayRef<Region *> regions,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion, // 调用点
    ...

size_t mlir::moveLoopInvariantCode(LoopLikeOpInterface loopLike) {
  return moveLoopInvariantCode(
      loopLike.getLoopRegions(),
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      }, // 定义点，传递了一个lambda函数
```



### Array / Vector / Set / hash
- llvm:ArrayRef
    - **轻量级数组引用，不进行内存分配或拷贝，适用于对已有数组进行访问而不修改的场景，是一个只读工具**
    - 常传入SmallVector或std::vector构造
    - tips: `const &SmallVector` <==> `ArrayRef`

- llvm:SmallVector
    - SmallVector<int64_t> srcDims(2, 1) 表示 初始化了两个元素，每个元素的值都是 `1`。
    - `SmallVector<int64_t, 2>` 表示包含 `int64_t` 元素的 `SmallVector` 类型，其中 `2` 是指定的初始大小
    - **tips:如果能预先知道需要的size，就使用reserve先分配**
    - 其他
        ```cpp
        llvm::reverse()
        llvm::to_vector()
        // SmallVector<int64_t> res{llvm::to_vector(llvm::seq((int64_t)0, size))};
        llvm::make_filter_range()
        llvm::map_range()
        ```

- llvm::SmallVectorImpl
    - SmallVector构造时调用的是 `SmallVector() : SmallVectorImpl<T>(N) {}`
    - 写一个以SmallVector为参数的函数，如果传入的元素个数是固定的，建议使用`SmallVectorImpl` 作为形参，来避免**对堆栈元素的隐式数量进行硬编码**
    > `SmallVector` 有一个参数 `N`表示该堆栈开辟了多少空间(元素)，函数上直接使用 `SmallVectorImpl`作形参能减少拷贝。 `SmallVectorImpl<Value> &`

- llvm::SetVector
    - 即有set的存储行为，又有vector的存储顺序
    - 常用方法
        - insert
        - contains / count
        - erase
        - clear
        - size
        - empty

- llvm:to_vector
    - 将数组类型的对象转为SmallVector，常用来解决用ArrayRef构造SmallVector
    - 用法
      ```cpp
      // 构造一个[0, 1, windowTy.getShape()-1]的数组
      SmallVector<int64_t> dstShape(llvm::to_vector(windowTy.getShape()));
      ```
    - 源码
      ```cpp
      template <typename R>
      SmallVector<ValueTypeFromRangeType<R>> to_vector(R &&Range) {
        return {std::begin(Range), std::end(Range)};
      }

      template <typename RangeType>
      // std::remove_const_t 用于移除模板参数类型的const修饰符
      // std::remove_reference_t 用于移除模板参数类型的引用修饰符
      // decltype 用于推断表达式的类型
      // std::declval 用于创建模板类型的临时值
      using ValueTypeFromRangeType =
          std::remove_const_t<std::remove_reference_t<decltype(*std::begin(
              std::declval<RangeType &>()))>>;
      ```

- llvm::seq
    - 生成一个连续的序列，包含起始值，不包含结束值。 `seq_inclusive` 既包含起始值，也包含结束值。
    - 用法
        - 循环的范围 `for (auto idx : llvm::seq<int>(0, rank))`
        - 创建个连续的`SmallVector<int64_t> res{llvm::to_vector(llvm::seq((int64_t)0, size))};`
    - 源码
      ```cpp
      auto seq(T Begin, T End) {
        return iota_range<T>(Begin, End, false);
      }
      auto seq(T Size) {
        return seq<T>(0, Size);
      }
      ```

- llvm:DenseSet
    - set和map都是基于hash的，都是无序的
    - 常用方法
      - insert(const ValueT &V)
        ```cpp
          // insert是有返回值
          std::pair<iterator, bool> insert(const ValueT &V)
          - first: 插入后的iterator
          - bool: 是否插入
        ```
      - erase(Iterator I)
      - count(const ValueT &V)
      - contains(const_arg_type_t<ValueT> V)
      > `using const_arg_type_t = typename const_pointer_or_const_ref<T>::type;`


- llvm:DenseMap
    - 和std::map类似， <key, value>
    - find(返回iterator) / lookup(返回value或null)
    - contains(返回true/false) / count(返回1/0)
    - std::pair<iterator, bool> insert / try_emplace : 返回值的second为true时，表示原本的map中不能找到key，已新插入一个<key, val>的pair，并以该pair的iterator作为返回值的first
        ```cpp
        launchInfo.insert({candidateOp, replacementIndexes})
        ```

- llvm::DenseMapInfo
    - hash表，只存key，`DenseMapInfo<T*>`
    - 使用 `getHashValue` 来获得hash值，最原始的方法是使用指针地址偏移计算的。但如果要实现自定义的hash，可以继承该类并重载 `getHashValue` 和 `isEqual` 方法
    - 例如：CSE中确定Op是否相同的代码
      ```cpp
      // mlir/lib/Transform/CSE.cpp
      struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
        static unsigned getHashValue(const Operation *opC) {
          return OperationEquivalence::computeHash(
              const_cast<Operation *>(opC),
              /*hashOperands*/OperationEquivalence::directHaseValue,
              /*hashResults*/OperationEquivalence::ignoreHaseValue,
              OperationEquivalence::IngoreLocations);
        }
      }
      static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
        auto *lhs = const_cast<Operation *>(lhsC);
        auto *rhs = const_cast<Operation *>(rhsC);
        if (lhs == rhs)
          return true;
        // 防止lsh/rhs是hash表中的特异值
        if (lhs == getTombstoneKey() || lhs == getEmptyKey()
            || rhs == getTombstoneKey() || rhs == getEmptyKey())
          return false;
        return OperationEquivalence::isEquivalentTo(
            const_cast<Operation>(lhsC), const_cast<Operation>(rhsC),
            OperationEquivalence::IngoreLocations);
      }
      ```

- llvm::ScopedHashTable
    - 需要包含 key, value, keyInfo, AllocatorTy 四个参数，例如 CSE.cpp 中构造该类型:
      ```cpp
      using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
          SimpleOperationInfo, AllocatorTy>
      // SimpleOperationInfo 继承自 DenseMapInfo<Operation *>
      // using AllocatorTy = llvm::RecyclingAllocator<llvm::BumpPtrAllocator, llvm::ScopedHashTableVal<Operation *, Operation *>>;
      ```
    - 实例化 `using ScopeTy = ScopedHashTableScope<K, V, KInfo, AllocatorTy>;`
      ```cpp
      ScopedMapTy::ScopeTy scope(knownValues);
      ```
    - 使用函数
      - V lookup(K key) : 在其中尝试查找key。一般来说 `SimpleOperationInfo` 都会有有自己的 `getHashValue` 和 `isEqual` 函数，这样就能根据key找到对应的value


### make_range

- llvm::map_range
    - 将一个range映射到另一个range
    - 用法
      ```cpp
      // 将 srcDims 中的每个元素都乘以2
      auto res = llvm::map_range(srcDims, [&](int64_t dim) { return dim * 2; });

      // 判断所有的operand的shape都相同
      assert(llvm::all_equal(llvm::map_range(op.getOperandTypes(),
          [](Type t) { return t.cast<TensorType>().getShape(); })));
      ```
    - 源码
      ```cpp
      template <typename ContainerTy, class FuncTy>
      auto map_range(ContainerTy &&C, FuncTy F) {
        return make_range(map_iterator(std::begin(C), F),
                          map_iterator(std::end(C), F));
      }

      template <typename ItTy, class FuncTy>
      inline mapped_iterator<ItTy, FuncTy> map_iterator(ItTy I, FuncTy F) {
        return mapped_iterator<ItTy, FuncTy>(std::move(I), std::move(F));
      }
      ```

- llvm::make_early_inc_range
    - 允许range中的元素被修改，且不影响iterator。例如遍历DenseMap对符合条件的iterator进行erase
    ```
      for (Block *block : llvm::make_early_inc_range(blockVec)) {
        ... // earse/replace ops in block
      }
    ```

### find

- llvm::find

```cpp
llvm::find(a.begin(), a.end(), val)
```

- llvm::find_if

```cpp
llvm::find_if(shapeIndexs, [&](int64_t shapeIndex) {
   return !oneSizeDimIndexsSet.count(shapeIndex);
});
```

tip: 如果需要在循环中查找，建议使用 `DenseSet`, `DenseMap` 类数据结构， `contains`, `find`, `count`等操作开销都小

### switch

- llvm:TypeSwitch
    - 常用在模板输入的pattern中，某些op需要额外的处理，例如构建某些op的时候需要额外set一些属性
      ```cpp
      auto newOp = clone(rewriter, op, TypeRange{newResType}, newOperands);
      auto inWhiteList = llvm::TypeSwitch<Operation *, bool>(newOp)
          .Case([&](linalg::BroadcastOp updateOp) {
            auto srcOp = cast<linalg::BroadcastOp>(op);
            updateOp.setDimensions...
            return true;
          })
          .Case([&](linalg::ReduceOp updateOp) {
            auto srcOp = cast<linalg::ReduceOp>(op);
            updateOp.setDimensions...
            return true;
          })
          .Case([&](linalg::TransposeOp updateOp) {
            auto srcOp = cast<linalg::TransposeOp>(op);
            updateOp.setPermutation...
            return true;
          })
          .Default([&](Operation *noNeedUpdate) { return false; });
      ```


### STL_Extra func

- llvm::count_if
    - 用法
      ```cpp
      llvm::count_if(inputs.getType(), [](Type type) {
        return type.isa<ShapedType>();
      });

      int64_t = numUsesInContainingOp = llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
        return containingOp->isAncestor(op);
      });
      ```
    - 实现
      ```cpp
      template <typename R, typename UnaryPredicate>
      auto count_if(R &&Range, UnaryPredicate P) {
        return std::count_if(adl_begin(Range), adl_end(Range), P);
      }
      ```

- llvm::transform
    - 使用
      ```cpp
      llvm::transform(srcDims, std::back_inserter(resDims)
      [&](int64_t dim) { return dim * 2; });
      ```

- llvm::hasSingleElement

    - 使用
        ```cpp
          auto containingOps = state.getPayloadOps(getContainingOp());
          if (!llvm::hasSingleElement(containingOps)) {
            return emitDefiniteFailure()
                   << "requires exactly one containing_op handle (got "
                   << llvm::range_size(containingOps) << ")";
          }
          Operation *containingOp = *containingOps.begin();
        ```

    - 实现
        ```cpp
        template <typename ContainerTy>
        bool hasSingleElement(ContainerTy &&C) {
          auto B = std::begin(C), E = std::end(C);
          return B != E && std::next(B) == E;
        }
        ```

- drop_begin / drop_end
	```cpp
	template <typename T>
	auto drop_begin(T &&RangeOrContainer, size_t N = 1) {
	  make_range(std::next(adl_begin(RangeOrContainer), N), adl_end(RangeOrContainer));
	}
	auto drop_begin(T &&RangeOrContainer, size_t N = 1) {
	  make_range(adl_begin(RangeOrContainer), adl_prev(adl_end(RangeOrContainer), N));
	}
	```

### setOperation
```cpp
llvm/include/llvm/ADT/SetOperations.h
```
- llvm::set_union(A, B): 计算集合A与集合B的并集，并将结果赋值给集合A
- llvm::set_intersection(A, B): 计算集合A与集合B的交集，并将结果赋值给集合A
- llvm::set_subtract(A, B): 计算集合A与集合B的差集（在A中但不在B中），并将结果赋值给集合A


---

## Matcher

```bash
mlir/include/mlir/IR/Matchers.h
```

### matchPattern

用来确定输入对象符合某种特性(`Pattern`)

- matchPattern(Value, Pattern)
- matchPattern(Operation *, Pattern)
- matchPattern(Attribute, Pattern)

实现形式例如: (对于不同类型的输入，都有一个 `match` 函数)

```cpp
template <typename Pattern>
inline bool matchPattern(Attribute attr, const Pattern &pattern) {
  static_assert(llvm::is_detected<detail::has_compatible_matcher_t, Pattern,
                                  Attribute>::value,
                "Pattern does not support matching Attributes");
  if (!attr)
    return false;
  return const_cast<Pattern &>(pattern).match(attr);
}
```

### Pattern
一些常见的 Pattern
- m_Constant() : 对于operation*而言，需要满足 `op->hasTrait<OpTrait::ConstantLike>()`
- m_AnyZeroFloat() : 是0且是APFloat，则匹配成功
- m_Zero() : 是0且是APInt，则匹配成功
- m_OneFloat() : 是1且是APFloat，则匹配成功
- m_One() : 是1且是APInt，则匹配成功
- m_ConstantFloat(FloatAttr::ValueType *bind_value) : 会把值写入bind_value(binds the constant integer value)
- m_ConstantInt(IntegerAttr::ValueType *bind_value)

```cpp
auto opFoldIsConstantValue = [](OpFoldResult ofr, int64_t value) {
  if (auto attr = llvm::dyn_cast_if_present<Attribute>(ofr))
    return cast<IntegerAttr>(attr).getInt() == value;
  llvm::APInt actual;
  return matchPattern(ofr.get<Value>(), m_ConstantInt(&actual)) &&
         actual.getSExtValue() == value;
};
```


---

## Operand

```cpp
mlir/include/mlir/IR/Operation.h
```

方法（src: Operation *）

- getDpsInputOperand
- getOperands
- getOpOperands：在需要修改operand时用，配合opOperand.set()

---

## OpOperand

```bash
mlir/include/mlir/IR/Value.h
```

每个Operation的Operand都是到Value的指针，这就意味着可以通过修改 OpOperand 来修改 Value。

```cpp
OpOperand a;
Value b = a->get();
unsigned idx = a.getOperandNumber(); // 返回operation中改operand的idx，一般想要获得operand的idx也是通过其OpOperand
Value value;
a.assign(value);
```

Operation都包含Results和Operands；Results中包含多个OpResult实例，Operands中包含多个OpOperand实例

<div style="text-align: center;"><img src="./img_MLIR_Note/Untitled.png" alt="Untitled" style="width: 90%;"></div>

- Operation * 的 getOpOperands() 将返回 `MutableArrayRef<OpOperand>`。相比 getOperands 获得的是 `OperandRange` ，一般不可修改，常常当 `ArrayRef<Value>` 来用。

某些op中一般实现了 `getxxx` 和 `getxxxMutable` 两种方法来获得 operands

```cpp
// scf.forall 的 相关方法
::mlir::Operation::operand_range getDynamicLowerBound();
::mlir::Operation::operand_range getDynamicUpperBound();
::mlir::Operation::operand_range getDynamicStep();
::mlir::Operation::operand_range getOutputs();
::mlir::MutableOperandRange getDynamicLowerBoundMutable();
::mlir::MutableOperandRange getDynamicUpperBoundMutable();
::mlir::MutableOperandRange getDynamicStepMutable();
::mlir::MutableOperandRange getOutputsMutable();
```

还有许多op在 `mlir/include/mlir/Dialect/xxx/IR/xxxOps.h` 中定义有op的一些方法

```cpp
Value getTagMemRef() { return getOperand(0); }
OpOperand &getTagMemRefMutable() { return getOperation()->getOpOperand(0); }
```

- Value 的 `getUses()` 方法返回的就是 `OpOperand`

- replaceUsesWithIf常用的写法
```cpp
// 例如使用 to 替换 from 在 userBlock中的使用
Value from, to;
Block *userBlock;
rewrite.replaceUsesWithIf(from, to, [&](Operation &use) {
  return userBlock == use.getOwner()->getBlock();
});
```

## OpResult

```bash
mlir/include/mlir/IR/Value.h
```

继承自 Value，和 OpOperand 的用法基本一致，基础的方法有:

- Operation *getOwner()
- unsigned getResultNumber()

---

## Operation

```cpp
mlir/include/mlir/IR/OperationSupport.h
mlir/lib/IR/Operation.cpp
```

每个op本质上都是一个 `Operation` 的指针

`Operation` 里面存了 OpName 和所有的 operands, results, attributes 和其它的东西

在不修改 IR 的情况下，每个 `Operation` 有唯一地址 ，如果只是对ir分析，可以使用 `llvm::DenseMap<Operation*, size_t> numberOfReference;` 来统计

<div style="text-align: center;"><img src="./img_MLIR_Note/Untitled%202.png" alt="Untitled" style="width: 90%;"></div>

### 常用方法

- OperationName opName = op->getName();

- 获得父op
    - getParentOp()：返回该operation的最近的上一级op

        如果要判断某个op的父op就用该方法

    - getParentOfType<OpTy>()：返回该operation的最近的上一级的为OpTy的op

        ```cpp
        template <typename OpTy>
        OpTy getParentOfType() {
        	auto *op = this;
        	while (op = op->getParentOp()) {
        		if (auto parentOp = dyn_cast<OpTy>(op))
        			return parentOp;
        		return OpTy;
        	}
        }
        ```

- getParentRegion 返回包含该op的region，也可以返回nullptr

- getBlock() 返回父block，而不是当前op的block

- getBody() 返回当前op内部的block或region

- getOperands()
  - 如果operand没有defineOp，则代表是BlockArgument

- bool isBeforeInBlock(Operation *other) 判断这个op是否在other之前，要求当前op和other都在同一block内

- getResults() / getResult(unsigned idx)

- 转换为目标op
    - cast<AddOp>(op)
    - dyn_cast<AddOp>(op)

- getUses() / getUsers()
相当于这个operation的所有result的use / user（其实就是operation *）的集合

- walk : 遍历 op 所有内部的 innerOp，第一次遍历的是本身

- emitOpError

```cpp
if (xxx) {
	op->emitOpError() << "error info" << "."
	return signalPassFailure();
}
```

### 注册op

例如想新建一个linalg_ext.xxop

参考：linalg.map的注册方法

- 首先在 `td` 中注册

```cpp
// mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

def MapOp : LinalgStructuredBase_Op<"map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Elementwise operations";
  let description = [{
    Models elementwise operations on tensors in terms of arithmetic operations
    on the corresponding elements.

    Example:
    ```
      %add = linalg.map
          ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
          outs(%init: tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
    ```

    Shortened print form is available. Applies to simple maps with one
    non-yield operation inside the body.

    The example above will be printed as:
    ```
      %add = linalg.map { arith.addf }
          ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
          outs(%init: tensor<64xf32>)
    ```
  }];

  let arguments = (ins
    // Input args
    Variadic<TensorOrMemref>:$inputs,

    // Output arg
    TensorOrMemref:$init
  );
  // 把result只限制在tensor语意上，memref时候就没有result
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$mapper);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Implement functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    SmallVector<OpOperand *> getOpOperandsMatchingBBargs() {
      return getDpsInputOperands();
    }

    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      if (isDpsInit(opOperand)) return false;
      return !getMatchingBlockArgument(opOperand).use_empty();
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return nullptr;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

- 然后在 `op.cpp` 中写 build 等方法
```cpp
// mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
void MapOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs, Value init,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, init);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  // 配合 td 中 `let results = (outs Variadic<AnyTensor>:$result);`
  Type initType = init.getType();
  if (llvm::isa<RankedTensorType>(initType))
    result.addTypes(initType);

  if (bodyBuild)
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       inputs, /*outputs=*/{}, bodyBuild);
}
```

### 创建op

使用 `OpBuilder` 来 create

#### 根据op的build函数create

create<OpTy>(…) : 查看

- `/mlir/include/mlir/Dialect/XXX/IR/XXXOps.td`

- `build/tools/mlir/include/mlir/Dialect/XXX/IR/XXXOps.h.inc`中对应op的 `build` 函数

#### 根据operationState来create

create(OperationState state)
```cpp
OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                     newResults, op->getAttrs(), op->getSuccessors());
 Operation *newOp = rewriter.create(state);
```
使用 `OperationState` 可以用来写一些模板函数pattern或者 `TypeConvert` 的naive_pattern，创建op会更加简单

例: 当op的mask来自于特殊情况，将起专为 `scf.if` + `op` 的形式

```cpp
static std::optional<Value> getBaseMaskVal(Value maskVal) {
  if (mask) {
    if (...)
      return ...;
  }
  return std::nullopt;
}

/// access_op(with mask) --> scf.if + access_op(without mask)
template<typename OpTy>
class FoldMaskAccessPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    std::optional<Value> maskBaseVal = getMaskBaseVal(op.getMask());
    if (!maskBaseVal.has_value())
      return failure();

    auto resTypes = op->getResultTypes();
    auto loc = op->getLoc();
    // If else region is empty, it will be fold in canonicalize.
    auto ifOp = rewriter.create<scf::IfOp>(loc,
                                           /*resultTypes*/resTypes,
                                           /*cond*/maskBaseVal.value(),
                                           /*addElseBlock*/true);

    // Then region.
    auto maskIndex = llvm::find_if(op.getOperand(), [&](Value operandVal) {
        return operandVal == op.getMask(); }) - op.getOperand().begin();
    SmallVector<Value> operand;
    operand.reserve(maskIndex);
    operand.append(op->operand_begin(), op->operand_begin() + maskIndex - 1);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    OperationState state(loc, op->getName().getStringRef(), operands,
                         resTypes, op->getAttrs(), op->getSuccessors());
    auto newOp = rewriter.create(state);
    rewriter.create<scf::YieldOp>(loc, newOp->getResults());

    // Else resgion.
    if (!resTypes.empty()) {
      // Fill with zero.
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto zeroVal = rewriter.create<arith::ConstantOp>(
          loc, resTypes.front(), rewriter.getZeroAttr(resTypes.front()));
      rewriter.create<scf::YieldOp>(loc, zeroVal);
      rewriter.replaceOp(op, ifOp);
      return success();
    }

    rewriter.eraseOp(op);
    return success();
  }
};
```

但如果 op 存在  `operandSegmentSizes` 属性，还需要额外修改

```cpp
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes")) {
    auto segments = attr.asArrayRef();
    SmallVector<int32_t> newSegments;
    newSegments.assign(segments.begin(), segments.end());
    for (size_t i = maskIdx; i < segments.size(); ++i) {
      newSegments[i] = 0;
    }
    op->setAttr("operandSegmentSizes",
                builder.getDenseI32ArrayAttr(newSegments));
  }
```

但是直接修改 `operandSegmentSizes` 属性的方法十分危险，且不利于维护，建议使用 `MutableOperandRange`(见 `mlir/include/mlir/IR/ValueRange.h`) 直接丢弃该 operand，同时也会修改 `operandSegmentSizes` 属性

```cpp
    // Then region.
    op.getMaskMutable().clear();
    // For load op, drop the `ohter` operand as well.
    if (llvm::isa<triton::LoadOp>(op))
      op.getOtherMutable().clear();
    auto newOp = rewriter.clone(&op);
    ...
```

---

## OpFoldResult

```cpp
mlir/include/mlir/IR/OpDefinition.h
```

OpFoldResult是一个PointUnion，可以是Value，Attribute，也可能是空的(使用其前记得 `isNull` 函数判断下)

```cpp
class OpFoldResult : public PointerUnion<Attribute, Value> {
  using PointerUnion<Attribute, Value>::PointerUnion;

public:
  void dump() const { llvm::errs() << *this << "\n"; }

  MLIRContext *getContext() const {
    return is<Attribute>() ? get<Attribute>().getContext()
                           : get<Value>().getContext();
  }
};
```

常见用法：

- Attribute / Value 转为 OpFoldResult
  - 直接包起来`OpFoldResult{b.getI64IntegerAttr(1)}`
  - 使用 `getAsOpFoldResult` 函数
    ```cpp
    OpFoldResult getAsOpFoldResult(Value val);
    SmallVector<OpFoldResult> getAsOpFoldResult(ValueRange values);
    SmallVector<OpFoldResult> getAsOpFoldResult(ArrayAttr arrayAttr);
    ```

- 一些op的fold行为
    ```cpp
    // mlir/lib/Dialect/Complex/IR/ComplexOps.cpp
    OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
      // complex.neg(complex.neg(a)) -> a
      if (auto negOp = getOperand().getDefiningOp<NegOp>())
        return negOp.getOperand();

      return {};
    }
    OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
      // complex.log(complex.exp(a)) -> a
      if (auto expOp = getOperand().getDefiningOp<ExpOp>())
        return expOp.getOperand();

      return {};
    }
    ```
- getAsOpFoldResult(ValueRange values)
    遍历values，尝试将value转化为constant Attribute，如果失败则返回value
    ```cpp
    Attribute attr;
    if (matchPattern(value, m_Constant(&attr)))
      return attr;
    return value;
    ```

- std::optional<int64_t> getconstantIntValue(OpFoldResult v)
```cpp
std::optional<int64_t> getconstantIntValue(OpFoldResult v) {
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return std::nullopt;
  }
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(attr)
    return intAttr.getValue().getSExtValue();
  return std::nullopt;
}

```

- bool isConstantIntValue(OpFoldResult ofr, int64_t value)

---

## PDLL

```cpp
mlir/include/mlir/Dialect/PDL/IR/PDLTypes
```

常见用法

`auto pdlOpType = pdl::OperationType::get(context);`

用pdll来实现pattern的match和rewrite

[[MLIR] PDLL](./composition/PDLL.md ':include')

```cpp
// ==== Rewrite Rule ======================
Rewrite tileOp(op: Op, tileSize: Attr);

// ==== Constraint Rule ======================
Constraint tilingLoopSizeLimit(op:Op, loopPosAttr:Attr, lowNumAttr:Attr, highNumAttr:Attr);

// Constraint + Rewrite -> Patterns
// ==== Patterns ======================
Pattern TileParallelofConvOpUseRange with benefit(9) {
  let root = op<linalg.conv_2d_nhwc_fhwc>; // payloadOp
  canTileParallel(root); // Constraint1
  tilingLoopSizeLimit(root, attr<"1">, attr<"513">, attr<"2000">); // Constraint2
  rewrite root with { // rewrite func
    tileOp(root, attr<"[1, 6, 1, 4, 1, 1, 1]">);
  };
}
```

---

## Pass

### 写一个 pass

- Passes.td中定义pass的基本信息（描述、作用对象）
  include/xxx/Transforms/Passes.td  （xxxx一般为project名字，例如iree，一般也会定义相应的namespace `mlir::iree`）

    ```cpp
    def passNamePass : Pass<"pass-flag">, "该pass的作用对象" > { // 作用域可以为 func::FuncOp 或 mlir::ModuleOp
    	let summary = "";
    	let description = [{
    		more detail

    		For example, consider the following input:

        ``` mlir

    	  ````

        After running, we get the expected:

        ``` mlir

      	```
      ]};
      let constructor = "mlir::xxxx::createPassNamePass()";
      let options = [
      	Option<"OptionName", "option-tag", "option-input-type", /*default*/"default-option-input-value",
      				 "Option description.">
      ];
      let dependentDialects = [
      	// 例如：
      	"func::FuncDialect";
      	"linalg::LinalgDialect",
      	"tensor::TensorDialect",
      ];

- Passed.h 中声明pass
  include/xxx/Transforms/Passes.h
  ```cpp
  std::unique_ptr<Pass> createPassNamePass();
  ```

- passName.cpp中定义pass的实现
  lib/xxx/Transforms/PassName.cpp

    ```cpp
    //===- passNamePass.cpp -----------------------------------------*- cpp -*-===//
    //
    // description
    //
    //===----------------------------------------------------------------------===//
    // 头文件，常见的如下
    #inlcude "xxxx/xxx/Transforms/Passes.h"
    #include "mlir/Dialect/xxx" // #include "mlir/Dialect/SCF/IR/SCF.h"
    #include "mlir/IR/BuiltinAttribute.h"
    #include "mlir/IR/BuiltinType.h"
    #include "mlir/IR/Operation.h"
    #include "mlir/IR/Region.h"
    #include "mlir/IR/Type.h"
    #include "mlir/Pass/Pass.h"
    #include "mlir/Support/LLVM.h"

    #define DEBUG_TYPE "pass-flag"

    using namespace mlir;
    using namespace mlir::xxxx;

    namespace{
    // 相关代码runOperation()写在匿名空间，匿名空间可以限制标识符的作用域，防止全局空间污染
    struct PassNamePass : public PassNamePassBase<PassNamePass> {
    	// explicit PassNamePass() = default(option-input-type optionName) {
    	// 	 this->optionName.setValue(optionName);
    	// }
    	explicit PassNamePass() = default;

    	void runOnOperation() override {
    		// 根据td中的作用域来返回，如果pass的td定义的作用域是mlir::ModuleOp,则这里返回moduleOp。
        // 如果pass.td中没有设置，则返回输入ir的top-level op
    		auto targetOp = getOperation();
    		MLIRContext *ctx = funcOp->getContext();
    		...
    		// 也可以使用pattern
    	}

    }
    }; // end struct

    } //namespace

    // std::unique_ptr mlir::xxxx::createPassNamePass(option-input-type optionName)
    std::unique_ptr mlir::xxxx::createPassNamePass(){
    	// return std::make_unique<PassNamePass>(optionName);
    	return std::make_unique<PassNamePass>();
    }
    ```

- passName.mlir中添加对该pass的单元测试
  xxxx/xxx/Transforms/Test/PassName.mlir

    ```cpp
    // RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(passname))' | FileCheck %s

    func.func @example() -> () {
    	...
      return ...
    }
    // CHECK-LABEL: @example
    // CHECK-NEXT:
    // CHECK-NEXT
    ```

> 使用 `mlir-tblgen` 主动生成 `pass.h.inc`
> `mlir-tblgen -gen-op-defs Passes.td -o Passes.h.inc `
> 详细查看 `mlir-tblgen -h | grep gen`

### Pass infrastructure

```cpp
mlir/include/mlir/Pass/Pass.h
mlir/lib/Pass/Pass.cpp
```

- 在pipeline中添加pass

    - addPass
        ```cpp
        // unique_ptr 申明独占资源，防止pass之间抢占资源
        void addPass(std::unique_ptr<Pass> pass);
        ```

        ```cpp
        void mlir::bufferization::buildBufferDeallocationPipeline(
            OpPassManager &pm, const BufferDeallocationPipelineOptions &options) {
          pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
          pm.addPass(createCanonicalizerPass());
          pm.addPass(createOwnershipBasedBufferDeallocationPass(options));
          pm.addPass(createCanonicalizerPass());
          pm.addPass(createBufferDeallocationSimplificationPass());
          pm.addPass(createLowerDeallocationsPass());
	      pm.addPass(createCSEPass());
	      pm.addPass(createCanonicalizerPass());
        }
        ```

    - addNestPass: 制定pass的作用op，常见的有FuncOp、ModuleOp，这个一般在 `pass.td` 中就定义该pass的作用域
        ```cpp
          void addNestedPass(std::unique_ptr<Pass> pass) {
            nest<OpT>().addPass(std::move(pass));
          }
          OpPassManager &nest() {
            return nest(OpT::getOperationName());
          }
        ```

        ```cpp
        void mlir::tosa::addTosaToLinalgPasses(
            OpPassManager &pm, const TosaToLinalgOptions &options) {
          if (!options.disableTosaDecompositions)
            pm.addNestedPass<func::FuncOp>(tosa::createTosaOptionalDecompositions());
          pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
          pm.addNestedPass<func::FuncOp>(tosa::createTosaInferShapesPass());
          pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
          ...
          pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
          pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
        }
        ```

- 保留当前IR的分析信息
    - markAllAnalysesPreserved()
    - markAnalysesPreserved(id)

```cpp
  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed)
    return markAllAnalysesPreserved();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
```

---

## Rank

"Rank" 表示数组的维度数量，而 "Dim" 表示数组在某个具体轴上的大小

- MemRefType 和 RankedTensorType 可以getRank()：因为都是继承自ShapedType

补充点常用的ShapedType函数

- isDynamicDim(unsigned idx)  ←→ getShape()[idx] == ShapedType::kDynamic
- hasStaticShape()
- int64_t getNumDynamicDims()
- getDimSize(unsigned idx)

---

## Range

表示范围

| ValueRange | ValueRange(ArrayRef<Value>) / ValueRange(ArrayRef<BlockArgument>) |
| --- | --- |
| TypeRange | TypeRange(ArrayRef<Type> types) |
| ValueTypeRange | 代表给定value的type |
| OperandRange  | TypeRange(ArrayRef<Operand> types) |
| ResultRange | TypeRange(ArrayRef<OpResult> types) |
| MutableOperandRange | 可以进行修改操作 append / assign / erase  |

---

## SymbolTable

```cpp
mlir/include/mlir/IR/SymbolTable.h
mlir/lib/SymbolTable.cpp
```

使用**SymbolTable trait**来表示Operation的特征表

在 op 的 liveness 相关计算中常见到

```cpp
mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp
```

### SymbolTable用法：

- 构建: SymbolTable(Operation *symbolTableOp)
- getSymbolAttrName { return "sym_name"; }
- lookup()
- getOp() ： 获得该symbolTable对应的Operation
- SymbolTable::SymbolUse
    - getUser() 返回该symbol ref的user operation

### SymbolUse

```cpp
class SymbolUse {
public:
  SymbolUse(Operation *op, SymbolRefAttr symbolRef)
      : owner(op), symbolRef(symbolRef) {}

  /// Return the operation user of this symbol reference.
  Operation *getUser() const { return owner; }

  /// Return the symbol reference that this use represents.
  SymbolRefAttr getSymbolRef() const { return symbolRef; }

private:
  /// The operation that this access is held by.
  Operation *owner;

  /// The symbol reference that this use represents.
  SymbolRefAttr symbolRef;
};
```

---

## Region

```cpp
mlir/include/mlir/IR/Region.h
mlir/include/mlir/Transforms/RegionUtils.h
```

region包含若干个block，一般linalgOp都包含一个region

- bool hasOneBlock() 常用来判断region内只有一个block，取Block的时候用 `a.front()`

- getUsedValuesDefinedAbove(MutableArrayRef<Region> regions, SetVector<Value> &values)
收集在regions中使用，但不在region中的blockArg上定义的Value，将其放入values

- takeBody: 把另外一个region的block占为己有(相当于把另外一个region的block的所有权给拿走了)
`newforallOp.getRegion().takeBody(forallOp.getRegion());`

- getOps() : 获得region内的所有op，有相对次序

`for (Operation &op : region.getOps())`

```cpp
for (Region *region : regions()) {
    std::queue<Operation *>worklist; // 如果不在乎遍历顺序，或者可以按压入顺序来逆序遍历
    for (Operation &op : region.getOps()) {
      worklist.push(&op);
    }
    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop();
      if (op->getParentRegion() != region)
        // 防止某些op提前被修改
        continue;
    }
    ...
}
```

举例：合并多余的barrier op
```cpp
static bool hasReadOrWriteEffect(Operation *op) {
  WalkResult ret = op->walk([&](Operation *innerOp) ->WalkResult {
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(innerOp)) {
      if (interface.hasEffect<MemoryEffects::Read>() ||
          interface.hasEffect<MemoryEffects::Write>())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ret == WalkResult::interrupt();
}

// 下面的op都是随意的给的
std::optional<Attribute> getExecutorAttr(Operation *op) {
  std::optional<Attribute> executor;
  bool hasExecutorAttr =
      llvm::TypeSwitch<Operation *, bool>(op)
          .Case<gpu::BarrierOp, gpu::MemsetOp,
                gpu::PrintfOp> ([&](auto oriOp) {
            executor = gpu;
            return true;
          }
          .Case<gpu::LaunchOp, gpu::AllocOp> ([&](auto oriOp) {
            executor = cpu;
            return true;
          }
          .Case<gpu::SubgroupMmaStoreMatrixOp> ([&](auto oriOp) {
            executor = oriOp.getExecutorAttr();
            return true;
          }
          .Default([&](Operation *other) {
            return false;
          });
  if (!hasExecutorAttr) {
    llvm::SmallDenseSet<Attribute> executorSets;
    WalkResult ret = op->walk([&](Operation *innerOp) -> WalkResult {
      if (innerOp == op || !hasReadOrWriteEffect(innerOp))
        return WalkResult::advance();
      auto currentExecutor = getExecutorAttr(innerOp);
      if (currentExecutor.has_value()) {
        executorSets.insert(currentExecutor.value());
        return WalkResult::advance();
      }
      // 那么就无法判断executor，采取保守
      return WalkResult::interrupt();
    });
    if (ret == WalkResult::advance() && executorSets.size() == 1)
      return *executorSets.begin();
  }
  return executor;
}

bool checkBarriersCanBeCombined(Operation *front,
    Operation *backOp, std::optional<Attribute> &recordExecutor) {
  bool canBeCombined = true;
  auto currentOp = frontOp->getNextNode();
  // Combine consecutive barriers.
  while (currentOp != backOp) {
    // 当barrier之间全为无读写effect的op也可以合并
    if (hasReadOrWriteEffect(currentOp)) {
      auto currentExecutor = getExecutorAttr(currentOp);
      if (!currentExecutor.has_value() ||
          (recordExecutor.has_value() &&
           (recordExecutor.value() != currentExecutor.value()))) {
        canBeCombined = false;
        break;
      }
      if (!recordExecutor.has_value()) {
        recordExecutor = curretExecutor;
      }
    }
    currentOp = currentOp->getNextNode();
  }
}

static void foldBarrierInFunc(gpu::FuncOp func) {
  auto barrierOpVec = llvm::to_vector(func.getOps<gpu::BarrierOp>());
  size_t numOfBarrier = barrierOpVec.size();
  if (numOfBarrier < 2)
    return;

  llvm::SmallDenseSet<Operation *> opsToCombine;
  size_t idxOfFront = 0;
  for (size_t idxOfBack = 1; idxOfBack < numOfBarrier; ++idxOfBack) {
    gpu::BarrierOp frontOp = barrierOpVec[idxOfFront];
    gpu::BarrierOp backOp = barrierOpVec[idxOfBack];
    std::optional<Attribute> recordExecutor;
    bool canBeCombined =
        checkBarriersCanBeCombined(frontOp, backOp, recordExecutor);
    if (recordExecutor.has_value() && canBeCombined = true) {
      size_t idxOfNext =  idxOfBack + 1;
      if (idxOfNext < numOfBarrier) {
        canBeCombined = checkBarriersCanBeCombined(backOp,
                                                   barrierOpVec[idxOfNext],
                                                  recordExecutor);
      }
    }

    if (canBeCombined) {
      opsToCombine.insert(backOp);
    } else {
      // Update the idx of frontOp.
      idxOfFront = idxOfBack;
    }
  }

  IRRewriter rewriter = IRRewriter(func->getContext());
  for (auto barrierOp : opsToCombine) {
    rewriter.eraseOp(barrierOp);
  }
}
```


---

## SideEffect

```cpp
mlir/include/mlir/Interfaces/SideEffectInterfaces.h
mlir/lib/Interfaces/SideEffectInterfaces.cpp
```

`sideeffect` 是一种用于表示函数或操作可能引起的副作用的概念。副作用是指对程序状态的任何更改，这可能包括但不限于内存写入、I/O 操作、全局状态的更改等。

`sideeffect` 通常与函数签名或操作的属性一起使用，以指定可能的副作用。对优化很重要

读写 side-effect 在程序中可能会导致一些问题，例如：

- **并发问题：** 多个线程同时访问和修改共享状态时可能导致竞态条件和数据竞争问题。
- **可变性引入的不确定性：** 当代码的行为依赖于外部状态时，其行为可能不确定，因为外部状态可能会在代码执行过程中发生变化。

因此，在编写代码时，需要谨慎处理读写 side-effect，尽量减少对共享状态的依赖，使用不可变数据结构和函数式编程技术可以帮助减少 side-effect 带来的问题。

一些常见的 "no side-effect" 操作：

1. **纯函数（Pure Functions）：** 纯函数是指没有副作用的函数，它的输出只取决于输入参数，不会影响程序的状态或外部环境。在纯函数中，不会对传入的参数进行修改，并且每次调用都会返回相同的结果。该op 可以被 CSE 和 DCE
2. **访问不可变数据结构：** 如果在代码中只读取不可变数据结构（如不可变数组、元组、字符串等），则这些操作通常不会引起副作用。因为不可变数据结构的内容在创建后是不可更改，因此不会影响其他部分或外部环境。
3. **访问只读变量或常量：** 如果在代码中只读取只读变量或常量，而不对其进行修改，则这些操作也不会引起副作用。只读变量或常量的值在初始化后是不可更改的。
4. **纯函数式编程操作：** 在纯函数式编程范式中，许多操作都是不可变的，因此它们通常不会引起副作用。这包括函数组合、映射、过滤等操作。

[[MLIR] Interface](./composition/Interface.md ':include') 中具体查看 `MemoryEffectOpInterface` 相关用法

---

## TableGen

基础语法 [[LLVM] tablegen](../LLVM/TableGen.md ':include')

### 可变/可选参数

```cpp
def AddOp : ToyOp<"add", [Pure]> {
  let summary = "add operation";
	// Variadic描述可变参数
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
}

def ReturnOp : ToyOp<"return", [Terminator, ReturnLike]> {
  let summary = "return operation"
	// Optional描述可选参数，在对应的cpp中也用optional声明该变量
  let arguments = (ins Optional<AnyInteger>:$data);
}

def HWRegOp : ToyOp<"reg"> {
  let summary = "hardware register";
  let arguments = (ins I1:$clock, AnyInteger:$input, Optional<I1>:$reset, UnitAttr:$is_public);
  let results = (outs AnyInteger:$result);
	// [{}] 来表示长文本
	// (`a`, $a)表示分组，当^存在时，只有对应的 Optional 或 UnitAttr 存在的时候，才会输出这个分组。
  let assemblyFormat = [{
    (`public` $is_public^)? $input
    `clock` $clock
    (`reset` $reset^)?
    attr-dict `:` functional-type($input, results)
  }];
}
```

### 输出格式

```cpp
def AddOp : ToyOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
	// 原本 %0 = “toy.add”(%a, %b) : (i32, i32) -> (i32)
	let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
  // 现在下面的也可行 %0 = toy.add %a, %b : i32, i32 -> i32
}
```

### OpVerifier

首先在td中声明 `hasVerifier=true`

```cpp
def SubOp : ToyOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let hasVerifier = true;
}
```

然后在toy.cpp中写该op的verifier

```cpp
using namespace mlir;
LogicalResult SubOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return this->emitError() << "Lhs Type " << getLhs().getType()
      << " not equal to rhs " << getRhs().getType();
  return success();
}
```

### builder

```cpp
def SubOp : ToyOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let builders = [
    OpBuilder<
      (ins "mlir::Value":$lhs, "mlir::Value":$rhs),
      "build($_builder, $_state, lhs.getType(), lhs, rhs);"
    >
  ];
  let hasVerifier = true;
}
```

例如： `fuse-into-containing-op` 的定义

```cpp
def FuseIntoContainingOp :
    Op<Transform_Dialect, "structured.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface,
          ["allowsRepeatedHandleOperands"]>,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fuse a producer into a containing operation.";
  let description = [{
    ...
  }];
  let arguments = (ins TransformHandleTypeInterface:$producer_op,
                       TransformHandleTypeInterface:$containing_op);
  let results = (outs TransformHandleTypeInterface:$fused_op,
                      TransformHandleTypeInterface:$new_containing_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict "
                       " `:` functional-type(operands, results)";
  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];
}
```

- `FuseIntoContainingOp` 继承自 `Op`，这是所有op的基类，实现了一些共有的方法
  ```cpp
  template <typename ConcreteType, template <typename T> class... Traits>
  class Op : public OpState, public Traits<ConcreteType>... {
  public:
    /// Inherit getOperation from `OpState`.
    using OpState::getOperation;
    using OpState::verify;
    using OpState::verifyRegions;
    ...
  ```

- `DeclareOpInterfaceMethods` 用来声明一个op继承了某个interface，并需要对其重写

使用 [DeclareOpInterfaceMethods](https://reviews.llvm.org/D79030?id=261051) 声明的interface时需要**被重写覆盖**，而其他都是默认继承（采用默认实现）。

- `builders` 用来声明一个op的builder。这里只是声明好了接口，要在其对应的文件中写一下实现
  ```cpp
  void transform::FuseIntoContainingOp::build(OpBuilder &builder,
                                              OperationState &result,
                                              Value producerOp,
                                              Value containingOp) {
    result.addOperands({producerOp, containingOp});
    auto resultType = transform::AnyOpType::get(builder.getContext());
    result.addTypes({resultType, resultType});
  }
  ```

---

## tensor

---

## test

当我们增加一个新feature后，需要添加对应的测试文件。

以[mlir/test/Dialect/Linalg/decompose-ops.mlir](https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/Linalg/decompose-ops.mlir)为例：

(0) 文件开头 `// RUN: opt_name -pass-name -split-input-file %s | FileCheck %s`

`FileCheck` 后可以所使用 `check-prefix` 来自定义 `CHECK` 的头

```mlir
// RUN: mlir-opt -test-linalg-decompose-ops -cse -split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-linalg-decompose-ops=remove-dead-args-and-results -cse -split-input-file %s | FileCheck %s --check-prefix=CANONICALIZECHECK
```

(1) `CHECK` : 最通用

```mlir
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : () -> memref<32xf32>
// 检查使用它时 %[[ALLOC0]]
```

(2) `CHECK-LABEL` : 一般后接func名

```mlir
// CHECK-LABEL: func @simple_op(
```

(3)`CHECK-SAME` : 上一个CHECK的延伸，

```mlir
// CHECK-LABEL: func.func @forward_no_view(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: i1

// CHECK:   %[[GENERIC1:.+]]:3 = linalg.generic
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP0]], #[[MAP3]]]
// CHECK-SAME:       ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:       outs(%[[INIT1]], %[[INIT2]], %[[INIT1]] :
```

(4) `CHECK-NEXT` : 检查紧挨上一个ir的ir

(5) `CHECK-DAG` : 不考虑检查的顺序

- 测试方法

```bash
build/bin/llvm-lit mlir/test/Dialect/Linalg/decompose-ops.mlir -a
```

综合例

```mlir
// CHECK-LABEL: func.func @forward_no_view(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: memref<32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: i1
// CHECK:   %[[ALLOC0:.*]] = memref.alloc() : () -> memref<32xf32>
// CHECK-NOT: memref.copy
// CHECK:   cf.cond_br %[[ARG1]], ^bb1(%[[ALLOC0]], %[[ARG0]] : memref<32xf32>, memref<32xf32>), ^bb2(%[[ARG0]] : memref<32xf32>)
// CHECK: ^bb1(%[[VAL0:.*]]: memref<32xf32>, %[[VAL1:.*]]: memref<32xf32>):
// CHECK:   %[[ALLOC1:.*]] = memref.alloc() : () -> memref<32xf32>
// CHECK:   vector.xxx %[[ALLOC1]], %[[VAL1]], []) : (memref<32xf32>, memref<32xf32>) -> ()
// CHECK:   cf.br ^bb2(%[[ALLOC1]] : memref<32xf32>)
```

---

## Trait

### OpTrait

```cpp
mlir/include/mlir/IR/OpDefinition.h
```

用 `mightHaveTrait` 、 `hasTrait` 来判断

- SameTypeOperands : 所有operand type相同
```cpp
class Arith_CompareOp<string mnemonic, list<Trait> traits = []> :
    Arith_Op<mnemonic, traits # [Pure, SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "::getI1SameShape($_self)">]> {
  let results = (outs BoolLike:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}
```

- SameOperandsAndResultType：操作数和返回值有相同的类型，使用后 assemblyFormat 里就只需要写任何某一个操作数的类型
```cpp
class Math_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> :
    Math_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

- InferTypeOpInterface ：通过输入和 attr 的类型推断返回值类型，自己写推断函数

- InferTypeOpAdaptor ：与上一个相似，但封装了一个 Adaptor，写起来会更简单

- ConstantLike : 代表该op是个constant op
```cpp
def LLVM_ConstantOp
    : LLVM_Op<"mlir.constant", [Pure, ConstantLike]>,
      LLVM_Builder<[{$res = getLLVMConstant($_resultType, $value, $_location,
                                            moduleTranslation);}]>
{
  let summary = "Defines a constant of LLVM type.";
  ...
```

- IsTerminator : 表示该op是一个block的最后一个操作(terminator operations)

一般pass处理op时都要避免处理带有该 trait 的op

```cpp
if (op->hasTrait<OpTrait::IsTerminator>()) {
  return;
}
```

- IsIsolatedFromAbove ：表示该op不会读取或修改其父操作的任何值，有这个trait的op是不能被schedule的

找到op的符合条件的parentOp作为基点来计算liveness，带有该 trait 的op一般可以理解为是 `isolateOp`
```cpp
Liveness liveness(op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());
Block *block = op->getBlock();
const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
```

---

## transform dialect

### transform Interface

```cpp
mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

transform IR: 实现 transformOpInterface 和相关的数据结构的op

payload IR: transfromations apply的对象

transform IR 应用在 payload IR(operations) 对应的 values 上，这些value又称为 handle。这些tansform IR也能应用在 Attribute 上，例如对module内的op进行排序，以 op-indexing 来作为handle。

#### apply方法

transform op在应用时，一般调用 apply方法，该方法需要传入三个元素

```cpp
DiagnosedSilenceableFailure
transform::FuseIntoContainingOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
```

- `transform::TransformRewriter`

```cpp
class TransformRewriter : public RewriterBase {
protected:
  friend class TransformState;
public:
  // notify给定的op已经被另一个op替换了,应该修改handle和payload(ops/val)之间的map
  LogicalResult notifyPayloadOperationReplaced(Operation *op,
                                               Operation *replacement);
```

- `transform::TransformResults`

```cpp
class TransformResults {
  friend class TransformState;
public:
  // 各种set方法...
```

常用 `RaggedArray` 数据结构， `RaggedArray` 表示2D数组，每行元素连续，每列元素并不固定

- `transform::TransformState`

    - Operation *getTopLevel :  topLevel包含all payload IR，一般来说是一个moduleOp。当transform ir应用在全局使用

    - getPayloadOps : 返回给定operand的对应payloadOps。当transform ir应用在特定的handle使用
      - 定义
      ```cpp
      auto getPayloadOps(Value value) const {
          ArrayRef<Operation *> view = getPayloadOpsView(value);
      ```
      - 使用
      ```cpp
      auto targetOps = state.getPayloadOps(getTarget())
      auto producerOps = state.getPayloadOps(getProducerOp());
      auto containingOps = state.getPayloadOps(getContainingOp());
      ```

    - `ArrayRef<Attribute> getParams(Value value)` : 返回传入transform ir中，给定operand对应的参数(都是以Attribute的形式传入的，例如tile_size)

        ```cpp
          auto tileSizeIntAttr = tileSizeAttr.dyn_cast_or_null<IntegerAttr>();
          if (!tileSizeIntAttr)
            return failure();
          auto tileSize = tileSizeIntAttr.getInt();
        ```

- getHandlesForPayloadOp

-  getHandlesForPayloadValue 传入的value是op的opResult

### linalg transformOp

```cpp
mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp
```

常见op（详细请学习[https://mlir.llvm.org/docs/Dialects/Transform/](https://mlir.llvm.org/docs/Dialects/Transform/)）

- transform.structured.match
其他transform ir使用前一般都要先match
ir中可能包含多个同名op，所以opIndexing来锁定handle

- transform.structured.tile_reduction_using_for

对reduction轴的tile

继承了 `TilingInterface` 的op都可以进行tile，每个可以tile的维度属于parallel或reduction


- transform.structured.tile_reduction_using_forall

- transform.structured.tile_using_forall

个人理解：forall region内的计算可以并行，for region内的计算不能并行

- transform.structured.fuse_into_containg

要求producer op一定是 `linalg::::GenericOp` ，常见的linalgOp都可以表示为linalg.generic

---

## Type

Value 必然包含 Type，Type 也可以作为 Attribute 附加在 Operation 上

### 常见类型

- ShapedType
    - ShapedType::kDynamic 用于判断某个数不是 `?`
    - isDynamicDim(i)
    - 当Type满足 `!llvm::isa<BaseMemRefType, TensorType>(type)` 时，认为该type是个scalar

- TensorType
  - kind: RankedTensorType / UnrankedTensorType
  - function:
    - hasRank() -> bool
    - getShape() -> ArrayRef<int64_t>
    - getElementType() -> Type
    - getElementTypeBitWidth()
    - clone(ArrayRef<int64_t> shape, Ttpe elemType) -> RankedTensorType

- BaseMemRefType
  - kind: MemRefType / UnrankedMemRefType
  - function: 大部分和 TensorType 相同，继承自 ShapedType的
    - clone(ArrayRef<int64_t> shape, Ttpe elemType) -> MemRefType
    - getMemorySpace() -> Attribute
    - getMemorySpaceAsInt -> unsigned

- MemRefType
    - getLayout() -> MemRefLayoutAttrInterface
      - isIdentity() : result type has no offset.
    - 获得dim
      ```cpp
          SmallVector<Value, 4> dynamicOperands;
          for (int i = 0; i < memrefType.getRank(); ++i) {
            if (!memrefType.isDynamicDim(i))
              continue;
            Value dim = rewriter.createOrFold<memref::DimOp>(loc, op.getInput(), i);
            dynamicOperands.push_back(dim);
          }
      ```
    - getStridesAndOffset(MemRefType t, SmallVectorImpl<int64_t> **&**strides, int64_t **&**offset)
    - canonicalizeStridedLayout(MemRefType t) -> MemRefType : 标准化t的layout格式，如果能canonicalize成静态的就没问题，否则返回MemRefType的layout将是affineMap的形式

例： bufferize时创建memref

```cpp
auto bufferType = cast<MemRefType>(buffer.getType());
MemRefType resultType;
if (bufferType.getLayout().isIdentity()) {
  // Standard layout: result type has no offset.
  MemRefLayoutAttrInterface layout;
  resultType = MemRefType::get({}, tensorResultType.getElementType(),
                               layout, bufferType.getMemorySpace());
} else {
  // Source memref has a layout map: result type has the same offset as
  // the source type.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(bufferType, strides, offset)))
    return failure();
  resultType = MemRef::get(
      {}, tensorResultType.getElementType,
      StrideLayoutAttr::get(op->getContext(), offset, {}),
      bufferType.getMemorySpace());
}
```

### 常用方法
src -> type (Value::getType())

- dyn_cast<MemRefType>() / dyn_cast<RankedTensorType>
- ShapedType
    - getElementTypeBitWidth
    - getRank
    - getShape: 当该type为ranked返回 ArrayRef<int64_t> ，否则assert
    - isDynamicDim / getNumDynamicDims / getDynamicDimIndex

- mlir/include/mlir/IR/TypeUtilities.h 中的一些函数
  - verifyCompatibleShape(Type lhs, Type rhs) : 比较两个Type的shape是否一致，不关心elemType

---

## tiling

```cpp
mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp
```

常见概念

- TilingInterface

对于有该interface的op可以cast成该interface `llvm::cast<TilingInterface>(op)`

​	- getLoopIteratorTypes：每个元素为utils::IteratorType，表示为utils::IteratorType::parallel或utils::IteratorType::reduction

​	- getIterationDomain：每个元素是一个Range

```cpp
if (auto intAttr = range.size.dyn_cast<Attribute>()) {
	tileSize = std::min(setTileSize, intAttr.cast<IntegerAttr>().getInt());
}
```

---

## Value

`value` 只可能表现为 `BlockArgument` 和 `OpResult` 两种形式，所以从 `value` 找其对应 `operation` 的方法：

- getDefiningOp： BlockArgument 返回 null
- getOwner()
    - OpResult ：返回拥有这个result的Operation
    - BlockArgument ：返回拥有这个blockarg的Block

```cpp
 	// Try to get a memory effect interface for the parent operation.
	Operation *op;
  if (BlockArgument arg = dyn_cast<BlockArgument>(value))
		// getParentOp会返回包含该block的最近Operation
    op = arg.getOwner()->getParentOp();
  else
    op = cast<OpResult>(value).getOwner();
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return failure();
```

- getUses()：返回 OpOperand 的迭代器，返回使用了这个value的OpOperand集合
`OpOperand &operand : value.getUses()`
- getUsers()：返回 Operation 的迭代器 ，返回仅包括直接依赖于该value的其他operation
user_iterator相当于对use_iterator使用getOwner()
use.getOwner() —> Operation*

修改value

- replaceAllUseWith(Value newValue)
- replaceAllUsesExcept(Value newValue, Operation *exceptedUser)

## Visitor

```bash
mlir/include/mlir/IR/Visitors.h
mlir/bin/IR/Visitors.cpp
```

用法

```cpp
Operation a;
WalkResult ret = a.walk([&](Ty opOfTy) -> WalkResult {
  ...
});
```

- `Ty` : 一般为 `Operation *`, `Block *`, `Region *`, `FuncOpInterface`, 直接是opname (比如 memref.alloc)

- `WalkResult` : Interrupt, Advance, Skip

- `WalkOrder` : 可以设置 walk 的方向， `PreOrder` 或  `PostOrder`，不设置时默认 `PreOrder`

- `wasInterrupted()` ,  `wasSkipped()` 判断 walk 的结果(返回值)

---

## clang code style

mlir的代码一般都得准守clang的format，简单的话可以使用 `git-clang-format` 工具

当然也有相关的脚本，可以参考iree 的 [build_tools/scripts/lint.sh](https://github.com/iree-org/iree/blob/main/build_tools/scripts/lint.sh) 去魔改

- 设置代码界限标识

一般clang format以80字符为换行界限，所以可以在vscode中设置显示提示竖线

1. 在设置中搜索 `editor.rulers`
2. 在 `setting.json` 中添加

```bash
    "[cpp]": {
        "editor.rulers": [
            { "column": 80, "color": "#ff0000" }
        ]
    },
```

- 人为控制clang format开关

可以通过注释来实现

```cpp
  // clang-format off
  patterns.add<
      AffineVectorLoadLowering,
      AffineVectorStoreLowering>(patterns.getContext());
  // clang-format on
```



---

## code tips

### 好用的pass

- `mlir-print-ir-after-all` : 打印每次pass后的结果
- `mlir-timing` : 输出每个pass的耗时和百分比

### coding优化

#### 重开销类型返回 -> 引用传递

如果需要从函数中获得一个重开销数据结构的对象，那么函数使用引用传递，不要返回类型，会增加拷贝开销

```cpp
SmallVector<Operation *> func(xxx);

- >

void func(xxx, SmallVector<Operation *> &res);
```

Operation* / Value 这类都是比较重开销的数据结构，但如果是 S mallVector<int64_t> 就没必要引用传递，直接使用返回目标类型

#### 不会被修改的SmallVector -> ArrayRef

- 传递一个不会被修改的SmallVector对象可以使用ArrayRef作为形参数

```cpp
// const llvm::SmallVector<Ty> & -> ArrayRef<Ty>
void func(const SmallVector<Operation *> &input);

->

void func(ArrayRef<Operation> input);
```

#### 循环变量类型static_cast

`for` 循环中的循环变量类型使用 `size_t`，其常用来表示 **一个数组的下标或对象的大小**。在下面的代码中如果循环变量 `i`的类型使用 `int64_t`则会出现warning。

```cpp
for (size_t i = 0; i < a.size(); ++i)
```
- 如果有需要使用 `i` 和 `int64_t`类型比较的时，就用 `static_cast<int64_t>(i)`
- 也可以一开始就将循环变量定义为 `int64_t` （推荐）

```cpp
for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i)
```

- 迭代变量i推荐使用 `++i`
> `i++` 可能会产生一个临时变量来保留前值，而 `++i` 并不会有这样的开销。虽然现代编译器一般能消除此类影响。

#### 循环中的vec查找行为 -> set

如果需要在循环中查找，建议使用 `DenseSet`, `DenseMap` 类数据结构， `contains`, `find`, `count`等操作开销都小（也可以llvm::sort后查找）

```cpp
for(int64_t i = 0; i < static_cast<int_64>(a.size()); ++i) {
    if (llvm::find(visitedIndexesVec, i) != visitedIndexesVec.end()) {
        ...

// 使用DenseSet 数据结构
for(int64_t i = 0; i < static_cast<int_64>(a.size()); ++i) {
    if (visitedIndexesSet.contains(i)) {
        ...

// sort后顺序访问
llvm::sort(visitedIndexesVec);
for(int64_t i = 0, index = 0; i < static_cast<int_64>(a.size()); ++i) {
    if (index < static_cast<int64_t>(visitIndexesVec.size()) &&
        visitedIndexesVec(index) == i) {
        index++;
    }
```

#### 常用reverse来为SmallVector预分配size

防止每次压入后动态扩张size带来的开销

如果预先不确定需要用多大，也可以先分配4

```cpp
SmallVector<Operation *, 4>
```



### 常用code

#### 重排序  `applyPermutationToVector`

```cpp
#include "mlir/Dialect/Utils/IndexingUtils.h"
auto srcShape = srcType.getShape();
SmallVector<int64_t> newShapes(srcShape.begin(), srcShape.end());
// New dimension information after translation.
applyPermutationToVector(newShapes, permutation); // permutation是shape的新序列
```

#### lambda函数编程习惯

```cpp
// [] : 捕获列表，可以是值捕获、引用捕获或者不捕获任何变量
[capture clause](parameters) -> return_type {
    // Lambda函数体
    // 可以访问外部变量，参数等
    return expression; // 可选
};
```

用 `[&]` 可以捕获外面的值，如果lambda函数内使用外面的值较少，可以直接加在 `[]` 内

最好指定输出格式

```cpp
auto getReassociations = [&](const DenseSet<int64_t>& dimIndexSet) -> SmallVector<ReassociationIndices> {
// `const SmallVector<int64_t>&` -> `ArrayRef<int64_t>`
auto getNewPermutation = [](const SmallVector<int64_t>& relativeOrder) -> SmallVector<int64_t> {
```

```cpp
llvm::for_each(relativeOrder, [](int64_t i) {llvm::outs() << i << " ";});

llvm::all_of(llvm::zip(array, array.slice(1)), [](const auto& pair) {
   return std::get<0>(pair) <= std::get<1>(pair);
});

llvm::find_if(shapeIndexs, [&](int64_t shapeIndex) {
   return !oneSizeDimIndexsSet.count(shapeIndex);
});
```

#### 创建一个包含dynamic shape的tensor.empty

```cpp
      SmallVector<OpFoldResult> newShapes;
      for (const auto &shape : llvm::enumerate(collapseResShape)) {
        if (ShapedType::isDynamic(shape.value())) {
          newShapes.emplace_back(rewriter.createOrFold<tensor::DimOp>(
              loc, collapseRes, shape.index()));
          continue;
        }
        newShapes.emplace_back(rewriter.getIndexAttr(shape.value()));
      }
			rewriter.create<tensor::EmptyOp>(loc, newShapes, srcType.getElementType());
```

#### 两维计算展开高维为循环
compute(memref<axbxf32>) -> for a { compute(memref<1xbxf32>) }

获得memref的offset 、size、stide的方法间 [offset-size-stride](#offset--stride--size) 节


下面以第一种方法获得的offset、size、stride为例，来处理

```cpp
// 创建for循环
Value lowBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
Value upperBound = rewriter.create<memref::DimOp>(loc, src, 0);
Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
scf::ForOp outerLoop = rewriter.create<scf::ForOp>(loc, lowBound, upperBound, one);
// 创建内部计算
Block *block = outerLoop.getBody();
rewriter.setInsertionPointToStart(body);
Value iter = body->getArgument(0);
// dst
auto createViewVal = [&](Value target) {
  // 不能支持dynamic stride
  Value innermostStride = rewriter.create<xxx::StrideOp>(loc, target, 0);
  Value strideIter = rewriter.create<arith::MulIOp>(loc, iter, innermostStride);
  Value offset = rewrite.create<xxx::OffsetOp>(loc, target);
  Value finalOffset = rewriter.create<arith::AddIOp>(loc, offset, strideIter);
  Value leftDim = rewriter.create<memref::DimOp>(loc, src, 1);
  return rewriter.create<xxx::MemrefCastOp>(
      loc, target, getElementTypeOrSelf(target.getType()),
      /*offset*/finalOffset,
      /*size*/ValueRange({one, leftDim}),
      /*stride*/ValueRange({one}));
}
for (Value operand : op.getOperands()) {
  Value newOperand = createViewVal(operand);
  newOperands.push_back(newOperand);
}
auto newOp = clone(rewriter, op, TypeRange{newResType}, newOperands);
rewriter.replaceOp(op, newOp->getResults());
```

#### 判断输入是否为升序

```cpp
    auto isIncremental = [](llvm::ArrayRef<int64_t> array) {
        if (array.size() <= 1) {
            return true;
        }
        return llvm::all_of(llvm::zip(array, array.slice(1)),
            [](const auto &pair) {
          return std::get<0>(pair) <= std::get<1>(pair);
        });
    };
```

#### 输出SmallVector<int64_t> a的元素相对大小次序

输入一个SmallVector<int64_t> a，要求输出一个SmallVector<int64_t> res，res记录了a中元素的相对大小顺序

输入 a = {3, 6, 4, 0}
则输出 b={1, 3, 2, 0}

```cpp
auto getNewOrder = [](const SmallVector<int64_t> &relativeOrder)
    -> SmallVector<int64_t> {
  SmallVector<int64_t> sortOrder = relativeOrder;
  llvm::sort(sortOrder);
  SmallVector<int64_t> res;
  llvm::transform(relativeOrder, std::back_inserter(res),
                  [&](int64_t num) {
                    return llvm::find(sortOrder, num) - sortOrder.begin();
                  });
  return res;
};
```

#### 定义一个driver来递归地处理func中符合条件的op

例：收集一起launch(或其他special)的op

```cpp
namespace {
class WrapDriver {
public:
  void processor(Operation *op);
private:
  /// The map between candidateOp and its result indexes that returned from
  /// scf.forall.
  llvm::DenseMap<Operation*, llvm::DenseSet<unsigned>> opsLaunchInfo;

  /// Operations can be wrapped using one loop.
  llvm::SmallVector<Operation *> opsToWrapTogather;

  /// Record the visited ops.
  llvm::DenseSet<Operation*> visited;

  /// The result number of scf.forall.
  unsigned replacementCount;

  bool checkOpIfNeedLaunch(Operation* op);

  LogicalResult processOnOp(Operation* op);

  void createWrapToLaunch();
};

} // namespace

//===----------------------------------------------------------------------===//
// Check if the op is a candidate.
//===----------------------------------------------------------------------===//

bool WrapDriver::checkOpIfNeedLaunch(Operation* op) {
  // Check if the op is a candidate to be wrapped.
  // 1. op needs to be a top level op.
  op.getParentOp() == rootOp(func::FuncOp)
  // 2. operands must has tensor semantic
  op.getNumresults() != 0
  llvm::all_of(op->getOperandTypes(), [&](Type operandtype){
    return operandtype.isIntOrIndexOrFloat() ||
    llvm::isa<RankedTensorType>(operandtype);
  });
  // The result value can only be scalar or ranked tensor with static shape.
}

//===----------------------------------------------------------------------===//
// Process on the candidate op.
//===----------------------------------------------------------------------===//

/// Use opsToWrapTogather and replacementCount to create a loop to wrap ops.
void WrapDriver::createWrapToLaunch() {
  // ...
}

void WrapDriver::processOnOp(Operation* op) {
  if (visited.contains(op)) {
    // The op has been visited.
    return;
  }
  // Mark the op has been visited.
  visited.insert(op);

  for (auto operand : op->getOperands()) {
    if (auto *operandOp = operand.getDefiningOp()) {
      if (checkOpIfNeedLaunch(operandOp)) {
        if (failed(processOnOp(operandOp))) {
          visited.earse(op);
          return failure();
        }
        continue;
      }
      // 则需遍历candidate的输入operand，如果operand的defineOp不存在 或 存在且在scf.forall之前，
      //则该op加入opsToWrapTogather是合法。反之（存在且在之后）将后续op的visited都清除。
      if (!opsToWrapTogather.empty() && opsToWrapTogather.front()->isBeforeInRegion(operandOp)) {
        visited.earse(op);
        return failure();
      }
    }
  }

  opsLaunchInfo.try_emplace(op);
  opsToWrapTogather.push_back(op);

  llvm::DenseSet<unsigned> replacementIndexes;
  for (const auto &[idx, resVal] : llvm::enumerate(op->getResults())) {
    for (auto *candidateOp : resVal.getUsers()) {
      if (!checkOpIfNeedLaunch(candidateOp) ||
          failed(processOnOp(candidateOp))) {
        replacementIndexes.insert(idx);
      }
    }
  }
  opsLaunchInfo[op] = replacementIndexes;
  replacementNum += replacementIndexes.size();
  return success();
}

//===----------------------------------------------------------------------===//
// Collect the candidate op from input.
//===----------------------------------------------------------------------===//

void WrapDriver::processor(func::FuncOp funcOp) {
  llvm::SmallVector<Operation*> worklist;
  funcOp->walk([&](Operation* workOp){
    if (checkOpIfNeedLaunch(workOp)) {
     worklist.push_back(workOp);
    }
  });

  for (auto *candidateOp : worklist) {
    if (visited.contains(candidateOp))
      continue;
    opsToWrapTogather.clear();
    replacementCount = 0;
    if (failed(processOnOp(candidateOp)))
      continue;
    createWrapToLaunch();
  }
}


namespace {
	void runOnOperation() override {
		WrapDriver driver;
    driver.processor(getOperation());
	}
} // namespace
```
