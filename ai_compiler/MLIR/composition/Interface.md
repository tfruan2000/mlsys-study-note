# Interface


## AttrInterface

- ElementsAttrInterface
    - DenseIntOrFPElements
    - DenseStringElements
    - DenseResourceElements
    - SparseElements
- MemRefLayoutAttrInterface
- TypeAttrInterface

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

## TilingInterface

对于有该interface的op可以cast成该interface `llvm::cast<TilingInterface>(op)`

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

## BranchOpInterface

所以分支类型的op都继承了该interface，例如scf.if，scf.for等

- mlir::SuccessorOperands getSuccessorOperands(unsigned index)

获得后继操作数，例如下面的例子中`invoke`的后继操作数是`^error`的 `%e`

```llvm
invoke %function(%0)
  label ^success ^error(%1 : i32)

^error(%e: !error, %arg0: i32):
```

例子：

```cpp
/// mlir/lib/Dialect/Linalg/Transforms/Detensorize.cpp
/// The result is a map from a branch op to a subset of indices of its operands.
/// The indices specify which of the branch op's operands should be detensored.
static DenseMap<Operation *, DenseSet<int>> computeBranchOpDetensoring(
    const DenseSet<BlockArgument> &blockArgsToDetensor) {
  DenseMap<Operation *, DenseSet<int>> detensorableBranchOps
  for (auto blockArgumentElem : blockArgsToDetensor) {
    Block *block = blockArgumentElem.getOwner()
    for (PredecessorIterator pred = block->pred_begin();
         pred != block->pred_end(); ++pred) {
      BranchOpInterface terminator =
          dyn_cast<BranchOpInterface>((*pred)->getTerminator());
      auto blockOperands =
          terminator.getSuccessorOperands(pred.getSuccessorIndex())
      if (blockOperands.empty() ||
          blockOperands.isOperandProduced(blockArgumentElem.getArgNumber()))
        continue
      detensorableBranchOps[terminator].insert(
          blockOperands.getOperandIndex(blockArgumentElem.getArgNumber()));
    }

  return detensorableBranchOps;
}
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

## RegionBranchOpInterface

分支类的op，比如 scf.if

- getSuccessorRegions

```cpp
    SmallVector<RegionSuccessor, 2> successors;
    branch.getSuccessorRegions(pred, successors);
```

## AllocOpInterface

常见的alloc op都继承了该interface，常见 memref.alloc

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