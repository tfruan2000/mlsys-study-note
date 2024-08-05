# Interface

## AttrInterface

- ElementsAttrInterface
    - DenseIntOrFPElements
    - DenseStringElements
    - DenseResourceElements
    - SparseElements
- MemRefLayoutAttrInterface
- TypeAttrInterface


## DialectInlinerInterface

(像 `inliner` 和 `canonicalize` 这样的函数，每个dialect都需要支持上， `inline` 有统一的实现可以继承，而 `canonicalize` 需要编写相关的fold函数即可)

为自定义的Dialect继承该interface以实现inliner的操作，然后在额外重载一点函数就行，例如 `isLegalToInline`

```cpp
struct AffineInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
```

在调inline这个pass时，遍历每个op的时候会使用 `getInterfaceFor`函数获得该op所属dialect重载的inline相关interface和函数

```cpp
bool InlinerInterface::isLegalToInline(Operation *op, Region *dest,
                                       bool wouldBeCloned,
                                       IRMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(op))
    return handler->isLegalToInline(op, dest, wouldBeCloned, valueMapping);
  return false;
}
```

## DestinationStyleOpInterface
linalOp都包含该interface

```cpp
OpOperandVector getDpsInputOperands()
OpOperandVector getDpsInitOperands()
	其中每个元素为 `OpOperand *` ，使用opOperand->get()即可得到alue
```

- BufferizableOpInterface：在oneShotBufferize pass会对有该inferface的op进行bufferize

- TilingInterface：对于有该interface的op可以cast成该interface `llvm::cast<TilingInterface>(op)`
    - getLoopIteratorTypes：每个元素为utils::IteratorType，表示为utils::IteratorType::parallel或utils::IteratorType::reduction
    - getIterationDomain：每个元素是一个Range

        ```cpp
        if (auto intAttr = range.size.dyn_cast<Attribute>()) {
        	tileSize = std::min(setTileSize, intAttr.cast<IntegerAttr>().getInt());
        }
        ```

- hasPureTensorSemantics
  所有operand都不为memref，至少有一个为tensor

- hasPureBufferSemantics

- isScalar
`!llvm::isa<BaseMemRefType, TensorType>(opOperand->get().getType());`


## TilingInterface

对于有该interface的op可以cast成该interface `llvm::cast<TilingInterface>(op)`。要求被tile的op都要实现该interface。

自定义的Dialect以及op并新增`TilingTnterface`可以参考Triton-Linalg中的[LinalgExtOpTilingInterface](https://github.com/Cambricon/triton-linalg/blob/master/lib/Dialect/LinalgExt/IR/LinalgExtOps.cpp)

- TilingResult类

```cpp
struct TilingResult {
    SmallVector<Operation *> tiledOps;
    SmallVector<Value> tiledValues; // 来自
}
```

- getLoopIteratorTypes：每个元素为utils::IteratorType，表示为utils::IteratorType::parallel或utils::IteratorType::reduction

- getIterationDomain：每个元素是一个Range

```cpp
if (auto intAttr = range.size.dyn_cast<Attribute>()) {
	tileSize = std::min(setTileSize, intAttr.cast<IntegerAttr>().getInt());
}
```

## MemoryEffectOpInterface

- getEffects
- hasEffect
- hasNoEffect

### EffectInstance

`EffectInstance` 是描写 `MemoryEffects` 行为的对象，有四种
- Allocate
- Free : 对alloc resource的free行为
- Read
- Write

例如判断某个op是否有read/write effect

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
  return ret.wasInterrupted();
}
```

`MemoryEffects::EffectInstance` 的常用方法

- Value getValue(): 返回这个effect是apply到哪个value，当不知道是apply到谁时就返回 nullptr
```cpp
copy A -> B
copy op 有 read 和 write，其中 read 的 apply value 是A(读A), write 的 apply value 是B(写B)
```

- EffectT *getEffect() 返回四种类型

- int getStage() : 返回这个 effect的发生阶段，值越小说明其发生更早。例如 `copy A -> B` 中 read 比 write 早，那么 `Read` effect 的 stage 就比 `Write` effect的 stage 小

- bool getEffectOnFullRegion() : 返回该 side effect 是否作用于region内的每个value，一般是带 region 内有计算的op，比如 linalg.generic / linalg.map / linalg.reduce

### 常用方法

- isPure(Operation *op)

- op.getEffect()
一般传入一个 `SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4>`
```cpp
void mlir::MemoryEffectOpInterface::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffects(getImpl(), getOperation(), effects);
  }
```

当然，更直接的是判断 `Operation *` 能否转为 `MemoryEffectOpInterface`，一般在 op 的 `td` 中标识该op是否可以有该interface

```cpp
class LinalgStructuredBase_Op<string mnemonic, list<Trait> props>
  : Op<Linalg_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
...
```

以及，需要额外注意带有 `MemWriteAt<0, PartialEffect>`的 op，因为如果writeeffect是full，那么后一次写就能覆盖前一次，但如果是partial，就不能因为有后一次写而删除前一次写。

```cpp
def LoadOp : MemRef_Op<"load", [...] {
  ...
  // 这意味着是部分读，只读一部分数据
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemReadAt<0, PartialEffect>]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

  ...
}
```

使用上

```cpp
if (auto memEffect = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffect.getEffects(effects); // 这样effects中就会收集到op的MemoryEffects.
}
```

- isMemoryEffectFree(Operation *op)
    - NoMemoryEffect
    - HasRecursiveMemoryEffects 且 所有 nested ops 都是 MemoryEffectFree

- hasEffect(Operation *op, Value value = nullptr) : 判断op是否对value有副作用，如果没提供value，则判断op是否有副作用
```cpp
template <typename... EffectTys>
auto memOp = dyn_cast<MeoryEffectOpInterface>(op);
if (!memOp)
  return false;
SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
memOp.getEffects(effects);
return llvm::any_of(effects, [&](MemoryEffects::Effect effect) {
  if (value && effect.getValue() != value)
    return false;
  return isa<EffectTys...>(effect.getEffect());
});
```

- onlyHasEffect
例如判断只有read effect
```cpp
if (!isMemoryEffectFree(op)) {
  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>())
    return failure();
}
```


- isOpTriviallyDead(Operation *op) : 当op没有使用者且不会引起副作用时，op就是trivially dead
```cpp
bool mlir::isOpTriviallyDead(Operation *op) {
  return op->use_empty() && wouldOpBeTriviallyDead(op);
}
```

- getEffectsRecursively(Operation *rootOp) : 获得该root op和其nest op（一般指root op的region内的所有op）的memory Effect

```cpp
std::optional<llvm::SmallVector<MemoryEffects::EffectInstance>>
mlir::getEffectsRecursively(Operation *rootOp)
```

- bool isSpeculatable(Operation* op)

判断一个op是否是用于判断相关的语意，会先尝试判断op是否有 `ConditionallySpeculatable`的OpInterface

然后会根据 `conditionallySpeculatable.getSpeculatability()` 来判断

```cpp
switch(conditionallySpeculatable.getSpeculatability()) {
  case Speculation::RecursivelySpeculatable:
    // 遍历op->getRegions()中的所有op，判断
  case Speculation::Speculatable:
    return true;
  case Speculation::NotSpeculatable:
    return false;
}
```

## BranchOpInterface

```bash
mlir/include/mlir/Interfaces/ControlFlowInterfaces.td
```

所以分支类型的op都继承了该interface，例如`cf.br`、`cf.cond_br`等。对它们的处理需要额外考虑其 `successors`。

例子：collect blocks

```cpp
SmallVector<Block *, 4> blocks;
if (isa<RegionBranchOpInterface>(op)) {
  // When the op is a `RegionBranchOpInterface`, like an `scf.for` or an
  // `scf.index_switch` op, its branch operand controls the flow into this
  // op's regions.
  for (Region &region : op->getRegions()) {
    for (Block &block : region)
      blocks.push_back(&block);
  }
} else if (isa<BranchOpInterface>(op)) {
  // When the op is a `BranchOpInterface`, like a `cf.cond_br` or a
  // `cf.switch` op, its branch operand controls the flow into this op's
  // successors.
  blocks = op->getSuccessors();
}
```

使用：

- mlir::SuccessorOperands getSuccessorOperands(unsigned index)

`Operation` 的 `Successors` 可以理解成该op的后续 `Block` 例如下面的ir中，`cf.br` 的 `Successors` 就是 `{^bb3}`。

> `RegionSuccessor` 表示后续的 `region`

`getSuccessorOperands` 函数会返回 `branchOp` 传给 `Successors` 中第 `index` 个 `successor` 的 `operand`，下面的ir中，`cf.br`的`getSuccessorOperands(0)`会得到 `{%2}`

```mlir
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```

例如获得 `BranchOpInterface` 对应的 `successors` 和 `successorOperands`

```cpp
// terminator -> Operation *
BranchOpInterface branchOp = dyn_cast<BranchOpInterface>(terminator);
if (!branchOp)
  return;

for (unsigned succI = 0, usccE = terminator->getNumSuccessors(); succI < succE; ++succI) {
  SuccessOperands successOperands = branchOp.getSuccessorOperands(succI);
  Block *successor = terminator->getSuccessor(succI);
}
```

- std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(unsigned operandIndex)

返回 `BranchOpInterface` 对应的第 `operandIndex` operand 所对应的 `successor BlockArg`。下面这段ir中`cf.br`的`getSuccessorBlockArgument(0)`会得到 `{%3}`

```mlir
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```

跨 block 的RE：首先需要获取 `%t` 的所有 `aliasChilds`。从 `%t` 开始找 `uses`，若 `use` 的 `owner` 是 `viewLike`，则继续遍历 `viewLike` 的 `result`；若 `use` 的 `owner` 是 `BranchOpInterface`，则使用 `getSuccessorBlockArgument` 获得对应的 `blockArg`，再继续往前找，直到%1。再获得 `%1` 到 `%s` 的 `viewLike` 链条，一步步找 `defineOp`，如果遇见 `blockArg` 的情况再利用 `aliasChilds` 中的对应关系 `*(llvm::find(%val) + 1)` 来跨 block。

> 获取 `aliasChilds` 的方法类似 `LocalAliasAnalysis` 中的 `collectUnderlyingAddressValues`。区别在于，此时我只需要获得 `%arg` 对应的 `BranchOpInterface` 的 `operand`，而不需要考虑其他冗余的 `operand`。

```mlir
  copy %s -> %t
  cf.br ^bb1(%t : tensor<*xf32>)
^bb1(%1: tensor<*xf32>):
  use %1
```

## RegionBranchOpInterface

```bash
mlir/include/mlir/Interfaces/ControlFlowInterfaces.td
```

带region的且会自动跳转region的op，比如 `scf.if`、`scf.for`。对它们的处理需要考虑它们 `region` 内的op。

> 像 `linalg` 许多op虽然有region，但并不属于 `RegionBranchOpInterface`

```cpp
SmallVector<Operation &, 4> ops;
if (isa<RegionBranchOpInterface>(op)) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region)
        for (Operation &inner : block)
          ops.push_back(&inner);
  }
}
```

- RegionBranchPoint

这个 class 表示 `RegionBranchOpInterface` 中 `branch`(理解为跳转) 的点，一般是

`parentOp` -> 即这个 `RegionBranchOpInterface`，常用 `RegionBranchPoint::parent()` 获得

`Region *` -> 这个 `RegionBranchOpInterface` 中的 `region` 即可，常用 `block->getParent()` 获得

- RegionSuccessor

表示 `a successor of a region`，可以是 `RegionBranchOpInterface`(parentOp) 本身，也可以是其 `region`。

`getSuccessor()` 获得 `Region *`， `getSuccessorInputs()` 获得该 `region` 的 input。

- void getSuccessorRegions(mlir::RegionBranchPoint point, mlir::SmallVectorImpl<RegionSuccessor> successors)

返回 `RegionBranchOpInterface` 的 `RegionSuccessor`，例如 `scf.if` 就会返回 `then region` 和 `else region`。

```cpp
SmallVector<RegionSuccessor, 2> successors;
// pred -> RegionBranchPoint
branch.getSuccessorRegions(pred, successors);
```

- OperandRange getEntrySuccessorOperands(mlir::RegionBranchPoint point)

对于下面的ir，使用 `getEntrySuccessorOperands(RegionBranchPoint::parent())` 会获得 `{%init}`。

```mlir
%0 = scf.for %i = %arg0 to %arg0 step %c2 iter_args(%arg = %init) -> (i32) {
  %1 = "test.op"(%i, %arg) : (index, i32) -> i32
  scf.yield %1 : i32
}
```

用法

```cpp
SmallVector<RegionSuccessor> successors;
SmallVector<Attribute> operands(op->getNumOperands(), nullptr);
branch.getEntrySuccessorRegions(operands, successors);

for (RegionSuccessor &successor : successors) {
  OperandRange operands = branch.getEntrySuccessorOperands(successor);
  ValueRange inputs = successor.getSuccessorInputs();
```

## LoopLikeOpInterface

在 `Ops.td` 中定义op时使用，例如

```bash
def AffineForOp : Affine_Op<"for",
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     ImplicitAffineTerminator, ConditionallySpeculatable,
     RecursiveMemoryEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface,
     // 定义函数
     ["getSingleInductionVar", "getSingleLowerBound", "getSingleStep",
      "getSingleUpperBound", "getYieldedValuesMutable",
      "replaceWithAdditionalYields"]>,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
     ["getEntrySuccessorOperands"]>]> {
```

相关op

```cpp
scf.for : 使用 scf.yield 返回
scf.forall : 使用 scf.forall.in_parallel + tensor.parallel_insert_slice
scf.paralle
scf.while
affine.for
affine.parallel
```

方法

- 下界、上界、step

    - `std::optional<::mlir::OpFoldResult> getSingleLowerBound`

    - `std::optional<::mlir::OpFoldResult> getSingleUpperBound`

    - `std::optional<::mlir::OpFoldResult> getSingleStep`

- `mlir::Block::BlockArgListType getRegionIterArgs`
- `mlir::ValueRange getYieldedValues()`

返回yield给下一个iteration的值，可以返回为 {}

- `std::optional<::mlir::ResultRange> getLoopResults()`
- `void moveOutOfLoop(op)` 将op移出loop
- `bool isDefinedOutsideOfLoop(mlir::Value value)`

判断输入value是否在loop region外定义

-  FailureOr<LoopLikeOpInterface> replaceWithAdditionalYields(RewiterBase &rewriter, ValueRange newInitOperands, bool replaceInitOperandUsesInLoop,  NewYieldValuesFn newYieldValuesFn)

```cpp
if (extractionOp && insertionOp) {
  // Create a new loop with an additional iter_arg.
  NewYieldValuesFn newYieldValuesFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
    return {insertionOp.getSourceOperand().get()};
  };
  // 新的yield对应的bbArg的initOp是extractionOp.getResult()
  // 再以insertOp的source作为该yeild的operand
  FailureOr<LoopLikeOpInterface> newLoop =
      loopLike.replaceWithAdditionalYields(
          rewriter, extractionOp.getResult(),
          /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
```

- OpResult getTiedLoopResult(BlockArgument bbArg) / OpResult getTiedLoopResult(OpOperand *opOperand)
    获得 `loopResults`（由getLoopResults()获得） 中该bbArg对应的值。

```cpp
// %a 对应 %res#0, %b 对应 %res#1
%res:2 = scf.forall(%arg3) in (1) shared_outs(%a = %empty0, %b =%empty1) -> (tensor<?xf32>, tensor<?xf32>)
```

- OpOperand *getTiedLoopInit(BlockArgument bbArg)

获得这个bbArg对应的loopInit，例如上面代码中 `%a` 对应的 loopInit 就是 `%empty0`



## SubsetOpInterface

用法

- `bool operatesOnEquivalentSubset(SubsetOpInterface candidate, function_ref<bool(Value, Value)> equivalenceFn)`
    判断this op 是否和 candidate 一起负责一个subset
- `bool operatesOnDisjointSubset(SubsetOpInterface candidate, function_ref<bool(Value, Value)> equivalenceFn)`

```cpp
// 获得loop中的 SubsetExtractionOpInterface 和 SubsetInsertionOpInterface 对应
  SmallVector<SubsetExtractionOpInterface> extractions;
  SmallVector<SubsetInsertionOpInterface> insertions;
  void insertExtractionOp(SubsetExtractionOpInterface extractionOp) {
    for (auto it : llvm::enumerate(insertions)) {
      if (!it.value())
        continue;
      auto other = cast<SubsetOpInterface>(it.value().getOperation());
      if (other.operatesOnEquivalentSubset(extractionOp, isEquivalent)) {
        extractions[it.index()] = extractionOp;
        return;
      }
    }
    // There is no known equivalent insertion op. Create a new entry.
    extractions.push_back(extractionOp);
    insertions.push_back({});
  }

  void insertInsertionOp(SubsetInsertionOpInterface insertionOp) {
    for (auto it : llvm::enumerate(extractions)) {
      if (!it.value())
        continue;
      auto other = cast<SubsetOpInterface>(it.value().getOperation());
      if (other.operatesOnEquivalentSubset(insertionOp, isEquivalent)) {
        insertions[it.index()] = insertionOp;
        return;
      }
    }
    // There is no known equivalent extraction op. Create a new entry.
    extractions.push_back({});
    insertions.push_back(insertionOp);
  }
```

子interface

- SubsetExtractionOpInterface

    - `OpOperand getSourceOperand()`
- SubsetInsertionOpInterface

    - `OpOperand getSourceOperand()`

    - `OpOperand getDestinationOperand()`

    - `OpResult getUpdatedDestination()` : 返回该op的result_tensor

        ```cpp
        OpResult detail::defaultGetUpdatedDestination(Operation *op) {
          auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
          assert(dstOp && "getUpdatedDestination must be implemented for non-DPS ops");
          auto insertionOp = cast<SubsetInsertionOpInterface>(op);
          return dstOp.getTiedOpResult(&insertionOp.getDestinationOperand());
        }
        ```


相关使用：将extractOp和insertOp提升到loop外

```cpp
// mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp
// Hoist the extraction/insertion ops.
iterArg = loopLike.getRegionIterArgs()[iterArgIdx];
OpResult loopResult = loopLike.getTiedLoopResult(iterArg);
OpResult newLoopResult = loopLike.getLoopResults()->back();
rewriter.moveOpBefore(extractionOp, loopLike);
rewriter.moveOpAfter(insertionOp, loopLike);
// insertOp外提后，就需要使用loop内还有的value来替换yield的输出
rewriter.replaceAllUsesWith(insertionOp.getUpdatedDestination(),
                            insertionOp.getDestinationOperand().get());
// extractOp提到loop外后，就需要使用对应的loopInitOp作为sourceOp
extractionOp.getSourceOperand().set(loopLike.getTiedLoopInit(iterArg)->get());
// 使用insertOp的result替换func.return的使用
rewriter.replaceAllUsesWith(loopResult, insertionOp.getUpdatedDestination());
// insertOp提出loop后，source要变成newLoopResult，dst要变成loopResult
insertionOp.getSourceOperand().set(newLoopResult);
insertionOp.getDestinationOperand().set(loopResult);
```

以下面的代码为例

```cpp
iterArg : %t
loopResult : %5#0
newLoopResult : %5#1
loopLike.getTiedLoopInit(iterArg)->get() : %arg0
对extractOp而言
- getSourceOperand() -> %t
对insertOp而言
- getSourceOperand() -> %2
- getDestinationOperand() -> %t
- getUpdatedDestination() -> %3
```

```cpp
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg0) -> (tensor<?xf32>) {
    %standalone = tensor.extract_slice %t[9][5][1] : tensor<?xf32> to tensor<5xf32>
    "test.foo"(%standalone) : (tensor<5xf32>) -> ()
    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    %3 = tensor.insert_slice %2 into %t[%sub][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %3 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>

// convert to
    %extracted_slice = tensor.extract_slice %arg0[0] [5] [1] : tensor<?xf32> to tensor<5xf32>
    %5:2 = scf.for %arg1 = %0 to %1 step %2 iter_args(%arg2 = %arg0, %arg3 = %extracted_slice) -> (tensor<?xf32>, tensor<5xf32>) {
      %extracted_slice_0 = tensor.extract_slice %arg2[9] [5] [1] : tensor<?xf32> to tensor<5xf32>
      "test.foo"(%extracted_slice_0) : (tensor<5xf32>) -> ()
      %6 = "test.foo"(%arg3) : (tensor<5xf32>) -> tensor<5xf32>
      scf.yield %arg2, %6 : tensor<?xf32>, tensor<5xf32>
    }
    %inserted_slice = tensor.insert_slice %5#1 into %5#0[%4] [5] [1] : tensor<5xf32> into tensor<?xf32>
    return %inserted_slice : tensor<?xf32>
```

## ViewLikeOpInterface

tensor.expand_shape, tensor.collapse_shape,tensor.insert_slice, tensor.extract_slice,
memref.expand_shape, memref.collapse_shape,
memref.view, memref.reshape, memref.reshape, memref.reinterpret_cast, memref.cast 等

这些的src都是单operand，但是offset / size / stride属性是 `OperandRange`

- Value getViewSource()

- 判断两个view是否相同
  ```cpp
  static bool isSameView(ViewLikeOpInterface a, ViewLikeOpInterface b) {
    if (a->getName() != b->getName() ||
        a->getAttrs() != b->getAttrs() ||
        a->getNumOperands() != b->getNumOperands()) {
      return false;
    }
    for (auto [operand1, operand2] : llvm::zip(a.getOperands(), b.getOperands())) {
      if (operand1 == a.getViewSource() && operand2 == b.getViewSource())
        continue;
      if (operand1 != operand2)
        return false;
    }
    return true;
  }
  ```


## OffsetSizeAndStrideOpInterface

也属于 `ViewLikeOpInterface` ，可以通过 `llvm::cast<OffsetSizeAndStrideOpInterface>(op)` 获得

- mlir::OperandRange getOffsets() / getSizes() / getStrides()

- llvm::ArrayRef<int64_t> getStaticOffsets() / getStaticSizes() / getStaticStrides()

- llvm::SmallVector<mlir::OpFoldResult, 4> getMixedOffsets() / getMixedSizes() / getMixedStrides()

- bool isDynamicOffset(idx) / isDynamicSize(idx) / isDynamicStride(idx)

- int64_t getStaticOffset(idx) / getStaticSize(idx) / getStaticStride(idx)

- mlir::Value getDynamicOffset(idx) / getDynamicSize(idx) / getDynamicStride(idx)

- isSameAs(::mlir::OffsetSizeAndStrideOpInterface, ::llvm::function_ref<bool, (::mlir::OpFoldResult, ::mlir::OpFoldResult)>)
```cpp
auto isSame = [](OpFoldResult, OpFoldResult) { return a == b};
if (prevInsertOp.isSameAs(insertOp, isSame))
```

## AllocOpInterface

常见的alloc op都继承了该interface，常见 memref.alloc

常用
- Value getSource()
- Value getTarget()

## FunctionOpInterface

各种 func op都继承了该interface，常见 func.func

常见于pass中的遍历处理，在 `.td` 中不设置其target，直接在 `runOnOperation()` 中锚定为 funcOp

```cpp

void processOnFunc(FunctionOpInterface func) {};

namespace {
struct ...
  void runOnOperation() override {
    Operation *input = getOperation();
    input->walk([](FunctionOpInterface func) {
      processOnFunc(func);
    })
  }
}
```

## CallOpInterface

调用 function 的 op，例如

```cpp
// mlir/include/mlir/Dialect/SPIRV/IR/SPIRVControlFlowOps.td
def SPIRV_FunctionCallOp : SPIRV_Op<"FunctionCall", [
    InFunctionScope, DeclareOpInterfaceMethods<CallOpInterface>]> {
```