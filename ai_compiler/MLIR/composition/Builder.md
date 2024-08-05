# Builder

```cpp
mlir/include/mlir/IR/Builders.h
mlir/lib/IR/Builders.cpp
```

`Builder` 用于创建新的 MLIR 操作，例如各种 `Type`, `Attr`, `Affine Expressions` 等

## OpBuilder

OpBuilder 继承自 Builder 类，**额外提供了struct Listener和class InsertPoint**

### InsertPoint

```cpp
Listener *getListener() const { return listener; }
void clearInsertionPoint();
InsertPoint saveInsertionPoint();

// insertionPoint设在block内的iterator处
void setInsertionPoint(Block *block, Block::iterator insertPoint);

// insertionPoint设到op前面，本质上还是找到op在block内的iterator
void setInsertionPoint(Operation *op) {
  setInsertPointPoint(op->getBlock(), Block::iterator(op));
}

// insertionPoint设到op后面
void setInsertionPointAfter(Operation *op) {
  setInsertPointPoint(op->getBlock(), ++Block::iterator(op));
}

// insertionPoint设到value后面
void setInsertionPointAfterValue(Value val) {
  if (Opeartion *op = val.getDefiningOp()) {
    setInsertionPointAfter(op);
  } else {
    auto blockArg = llvm::cast<BlockArguement>(val);
    setInsertionPointToStart(blockArg.getOwner());
  }
}

// insertionPoint设到block开头
void setInsertionPointToStart(Block *block);

// insertionPoint设到block结尾
void setInsertionPointToEnd(Block *block);
```

### create

```cpp
Block *createBlock(Region *parent, Region::iterator insertPt = {},
                   TypeRange argTypes = std::nullopt,
                   ArrayRef<Location> locs = std::nullopt);
// createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
Operation *insert(Operation *op);
Operation *create(const OperationState &state);
```

- OpTy create(loc, Args &&..args);
先创建  `OperationState` 对象，再调用 `OpTy::build` 方法创建 `Operation` 对象
- createOrFold
返回值是 Value （也可以直接作为 OpFoldResult 使用)
创建op后立即尝试fold，一般在创建某些有xxxOp.cpp中有opFoldPattern的op时使用，例如一些arith dialect 中的op 以及 memref.dim
> 参见: mlir/lib/Dialect/Complex/IR/ComplexOps.cpp

### clone

```cpp
Operation *clone(Operation &op, IRMapping &mapper);
Operation *clone(Operation &op);
Operation *cloneWithoutRegions(Operation &op, IRMapping &mapper) {
  return insert(op.cloneWithoutRegions(mapper));
}
Operation *cloneWithoutRegions(Operation &op) {
  return insert(op.cloneWithoutRegions());
}
```

例：使用linalg.reduce的region创建一个linalg.map

```cpp
// op 是 linalg.reduce
Value emptyOp = rewriter.create<tensor::EmptyOp>(
    loc, initDims, dstType.getElementType());
auto mapOp = rewriter.create<linalg::MapOp>(
    loc, ValueRange(op.getDpsInputs()), emptyOp,
    [&](OpBuilder &b, Location loc, ValueRange args) {});

// 下面的代码等价于 rewriter.inlineRegionBefore(op->getRegion(0), mapOp->getRegion(0), mapOp->getRegion(0)->begion());

Block *opBody = op.getBody();
llvm::SmallVector<Value> bbArgs;
for(Operation *opOperand : op.getOpOperandsMatchingBBargs()) {
  bbArgs.emplace_back(opBody->getArgument(
      opOperand->getOperandNumber()));
}
Block *mapOpBody = mapOp.getBlock();
SmallVector<BlockArgument> mapOpBbargs;
for (OpOperand *opOperand : mapOp.getOpOperandsMatchingBBargs()) {
  mapOpBbargs.emplace_back(mapOpBody->getArgument(opOperand->getOperandNumber());
}
assert(mapOpBbargs.size() == bbArgs.size());
IRMapping bvm;
for (auto [bbarg, newBBarg] : llvm::zip(bbArgs, mapOpBbargs)) {
  bvm.map(bbarg, newBBarg);
}
rewriter.setInsertionPointToStart(mapOpBody);
for (Operation &operation : *reduceOpBody) {
  rewriter.clone(operation, bvm);
}
```

## Listener

Listener用于hook到OpBuilder的操作，Listener继承自 ListenerBase，ListenerBase有两种 kind

```cpp
// Listener() : ListenerBase(ListenerBase::Kind::OpBuilderListener)
struct ListenerBase {
  enum class Kind {
    OpBuilderListener = 0,
    RewriterBaseListener = 1
  };
  ...
}
```

Listener常用两个函数为 `notifyOperationInserted(Operation *Op)` 和 `notifyBlockCreated(Block *block)`。自定义rewriter时，一般需要 `override` 这两个函数。

## RewriterBase

```cpp
mlir/include/mlir/IR/PatternMatch.h
mlir/lib/IR/PatternMatch.cpp
```

继承自 OpBuilder，且将 Listener 设置为 RewriterBaseListener

```cpp
class RewriterBase : public OpBuilder {
public:
  struct Listener : public OpBuilder::Listener {
    Listener() : OpBuilder::Listener(Kind::RewriterBaseListener) {}
  };
}
```

常用函数：

1.notify ： 在正式对op修改前都需要调用notify，以便listener监听

- notifyOperationModified : in-place 修改

- notifyOperationReplaced : 调用 replaceOp时触发
  ```cpp
  if (auto *listener = dyn_cast_if_present<RewriteBase::Listener>(rewriter.getListener())) {
    listener->notifyOperationReplaced(op, existing);
  }
  rewriter.replaceAllUsesWith(op->getResults())
  opsToErase.push_back(op);
  ```
- notifyOperationErased : 调用 earseOp时触发

2.modifyOpInPlace : 会调用 `startOpModification` 和 `finalizeOpModification`

```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

3.replaceAllUsesWith

### ForwardingListener

可以将所有 `notify` 发送给另外一个 `OpBuilder::Listener`，用于创建监听链条

```cpp
struct ForwardingListener : public RewriterBase::Listener {
  ForwardingListener(OpBuilder::Listener *listener) : listener(listener) {}
```

### IRRewriter

继承自 `RewriteBase`，当 `PatternRewriter` 不可用时才使用

```cpp
class IRRewriter : public RewriterBase {
public:
  explicit IRRewriter(MLIRContext *ctx, OpBuilder::Listener *listener = nullptr)
      : RewriterBase(ctx, listener) {}
  explicit IRRewriter(const OpBuilder &builder) : RewriterBase(builder) {}
};
```

## PatternMatch

### PatternBenefit

一般配合 `Pattern` 使用，表示一个pattern的benefit，benefit越高越先apply

```cpp
patterns.add<DoWhileLowering>(patterns.getContext(), /*benefit=*/2);
```

benefit的取值范围为 **0到65535**

### Pattern

```cpp
class Pattern {
  /// This enum represents the kind of value used to select the root operations
  /// that match this pattern.
  enum class RootKind {
    Any,
    OperationName,
    InterfaceID,
    TraitID
  };
 ...
```

有match、rewrite、matchAndRewrite这些函数，也会设置 `PatternBenefit` (默认为1)

### RewritePattern

继承自pattern

```cpp
  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    return failure();
  }
```

一些子类：

1.OpOrInterfaceRewritePatternBase

- **OpRewritePattern** : 使用 SourceOp::getOperationName() 来match

- **OpInterfaceRewritePattern** : 使用 SourceOp::getInterfaceID() 来match
```cpp
struct AddOpPat : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter & rewriter) const override{

static EraseDeadLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override{
```

2.OpTraitRewritePattern

-  使用 TypeID::get<TraitType>() 来match

- 例如某些elementwiseTrait : `OpTraitRewritePattern<OpTrait::Elementwise>`

### RewritePatternSet

```cpp
RewritePatternSet(MLIRContext *context,
                  std::unique_ptr<RewritePattern> pattern)
    : context(context) {
  nativePatterns.emplace_back(std::move(pattern));
}
```

1.新建pattern

所以一般新建 `RewritePatternSet` 对象时都得传入 context

```cpp
RewritePatternSet patterns(&getContext());
```

然后再一些函数来归类pattern

```cpp
populateAffineToStdConversionPatterns(patterns);
void mlir::populateAffineToStdConversionPatterns(RewritePatternSet &patterns) {
    ...
}
```

也可以通过[PDLL](./PDLL.md ':include')来写pattern(包含constrict和rewrite)
```cpp
RewritePatternSet(PDLPatternModule &&pattern)
    : context(pattern.getContext()), pdlPatterns(std::move(pattern)) {}
```

2.add : 向set中添加pattern

```cpp
add(LogicalResult (*implFn)(OpType, PatternRewriter &rewriter),
    PatternBenefit benefit = 1, ArrayRef<StringRef> generatedNames = {})
```

3.clear : 清空set中的pattern

### PatternRewriter

继承自 `RewriteBase`， 用于重写（transform）现有 MLIR 操作的工具。它提供了一组方法，允许用户在遍历操作并修改它们时进行规则匹配和替换。在rewrite pattern中才使用

- `PatternRewriter &rewriter`

- `ConversionPatternRewriter &rewriter` : 相比pattern rewriter要多传入一个adaptor，详细见 [Conversion](../MLIR_Note.md ':include')节

常用操作

1.设置插入点（与builder同）
- setInsertionPoint(Operantion *)
- setInsertionPointAfter

2.block
getBlock()

3.创建
- create<OpTy>(…)
- create(OperationState)
  ```cpp
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, op->getAttrs(), op->getSuccessors());
  Operation *newOp = rewriter.create(state);
  ```

4.替换
- replaceOp(Operation *op, Operation *newOp)

- replaceOp(Operation *op, ValueRange newValues())

例如getResults()作为ValueRange输入

- replaceAllOpUsesWith(Operation \*from, ValueRange to) / replaceAllOpUsesWith(Opeation \*from, Operation \*to )

- replaceUsesWithIf(Value from, Value to, func_ref) / replaceUsesWithIf(ValueRange from, Value to, func_ref) / replaceUsesWithIf(Operation \*op, Value to, func_ref)
  ```cpp
  // 替换forallOp外的使用
  rewriter.replaceAllUsesWithIf(workOp->getResult(0), forallOp->getResults(idx)
    [&](OpOperand use) {return !forallOp->isProperAncestor(use.getOwner())
  // 仅替换当前op的使用
  rewriter.replaceUsesWithIf(emptyOp->getResult(), newEmptyOp->getResult(),
      [&](OpOperand use) { return use.getOwner() == op; });
  ```

- replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser) 本质是使用 `replaceUsesWithIf` 来实现
  ```cpp
  rewriter.replaceUsesWithIf(from, to,
      [&](OpOperand use) { return use.getOwner() != exceptedUser; });
  ```

5.消除
- earseOp(Operation *op) : 如果要在pattern中删除op，最好使用 `rewriter.earseOp`，使用op自带的 `erase` 函数代码运行时会在debug模式出问题

- earseBlock(Block *block)

示例

```cpp
struct AddOpPat : public OpRewritePattern<AddOp> {
	using OpRewritePattern<AddOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(AddOp op,
		PatternRewriter & rewriter) const override{
	xxx
	return success();
}
};

class AddOpPatPass : public impl::AddOpPatPassBase<AddOpPatPass> {
	explicit AddOpPatPass() = default;
	void runOnOperation() override {
		RewriterPatternset patterns(&getContext());
		patterns.add<AddOpPat>(patterns.getContext());
		if (failed(applyPatternAndFlodGreedily(getoperation(), std::move(patterns))))
			return signalPassFailure();
	};
}

std::unique_ptr<pass> mlir::createAddOpPatPass() {
	return std::make_unique<AddOpPatPass>;
}
```
