# [MLIR]Analysis

## Analisys Manager

```bash
mlir/include/mlir/Pass/AnalysisManager.h
```
`Analyses` 是独立于其他设施的数据结构，可以将相关的信息 perserve 起来。

例如 `Transforms/CSE.cpp` 中就将一些 `Analyses` 信息保存给下一次分析。
```cpp
  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed)
    return markAllAnalysesPreserved();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
```

但使用 `markAnalysesPreserved` 在 `pass` 间传递信息的行为是不可取的，因为该功能只是为了减少编译时间，要在 `pass` 间传递信息最合理的方法是设计一套 `Attribute` 挂在 op上。

## Dataflow Framework

```bash
mlir/include/mlir/Analysis/DataFlowFramework.h
mlir/lib/Analysis/DataFlowFramework.cpp
```

### ChangeResult

```cpp
enum class [[nodiscard]] ChangeResult {
  NoChange,
  Change,
};
```

> `[[nodiscard]]` 来标记函数的返回值不应该被忽略。也就是说，当调用一个被标记为 `[[nodiscard]]` 的函数时，
> 如果返回值没有被使用，编译器会发出警告。

### ProgramPoint

ProgramPoint 是一个 `PointerUnion`，可以是 `Operation *, Value, Block *`

### DataFlowSolver

实现 child data-flow analyses，使用的是 fixed-point iteration 算法。
一直维护 `AnalysisState` 和 `ProgramPoint` 信息。

数据流分析的流程：

(1) 加载并初始化 children analyses
例如
```cpp
std::unique_ptr<mlir::DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<mlir::DataFlowSolver>();
  solver->load<mlir::dataflow::DeadCodeAnalysis>();
  solver->load<mlir::dataflow::SparseConstantPropagation>();
  ...
  return solver;
}
```

(2) 配置并运行分析，直到达到设置的 fix-point

```cpp
if (failed(solver->initializeAndRun(root))) {
  LLVM_DEBUG(llvm::dbgs() << " - XXX analysis failed.\n");
  return failure();
}
```

(3) 从 solver 中 query analysis state results

```cpp
// lookupState 可能返回 null
const auto analysisState = solver->lookupState<xxxxx>(op)
```

## Liveness

```bash
mlir/include/mlir/Analysis/Liveness.h
mlir/bin/Analysis/Liveness.cpp
```

- 对 op ->  Liveness(Operation *op)

- 对 block -> liveness.getLiveness(block) -> LivenessBlockInfo

## AliasAnalysis

## LocalAliasAnalysis

```bash
mlir/include/mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h
mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.h
```

1.AliasResult: 两个location之间是否有关系
- Kind
  - NoAlias
  - MayAlias
  - PartialAlias : 两个loc互相alias，但是部分重叠
  - MustAlias
- isNO / isMay / isPartial / isMust -> bool

2.AliasResult alias(Value lhs, Value rhs);

```cpp
// 确定一个op是否对一个value有读/写行为
bool isOpReadOrWriteInplace(Operation *op, Value val) {
  auto memInterface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface)
    return false;
  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);
  bool readOnVal = false;
  bool writeOnVal = false;
  LocalAliasAnalysis analysis;
  for (MemoryEffects::EffectInstance effect : effects) {
    if (llvm::isa<MemoryEffects::Read>(effect.getEffect()) &&
        !analysis.alias(val, effect.getValue()).isNo()) {
        readOnVal = true;
    }
    if (llvm::isa<MemoryEffects::Read>(effect.getEffect() &&
        !analysis.alias(val, effetc.getValue()).isNo()) {
      writeOnVal = true;
    }
  }
  return readOnVal || writeOnVal;
}
```

3.collectUnderlyingAddressValues

重载了多种形式，常用的有以下输入：

- (Value, SmallVectorImpl<Value> &output)

- (OpResult result, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)
  - result.getOwner() -> ViewLikeOpInterface -> 继续调用 viewOp.getViewSource()
  - result.getOwner() -> RegionBranchOpInterface

- (BlockArguement arg, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)

## SliceAnalysis

用来遍历 use-def 链的 analysis
一般可以将 use-def 理解为
- def : 写
- use : 读

```bash
 ____
 \  /  defs (in some topological order)
  \/
  op
  /\
 /  \  uses (in some topological order)
/____\
```

### getSlice

1.getForwardSlice : 获得root op的use链 (向ir的结尾找)

```bash
从 0 开始追，可以获得 {9, 7, 8, 5, 1, 2, 6, 3, 4}

              0
   ___________|___________
   1       2      3      4
   |_______|      |______|
   |   |             |
   |   5             6
   |___|_____________|
     |               |
     7               8
     |_______________|
             |
             9
```

输入， root可以是op，也可以是value
```cpp
void getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});

void getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});
```

2.getBackWardSlice : 获得root op的def链 (向ir的开头找)

```bash
从 node 8 开始， 可以获得 {1, 2, 5, 3, 4, 6}

   1       2      3      4
   |_______|      |______|
   |   |             |
   |   5             6
   |___|_____________|
     |               |
     7               8
     |_______________|
             |
             9
```

输入， root可以是op，也可以是value

```cpp
void getBackwardSlice(Operation *op, SetVector<Operation *> *bac
                      const BackwardSliceOptions &options = {});

void getBackwardSlice(Value root, SetVector<Operation *> *backwa
                      const BackwardSliceOptions &options = {});
```

### SliceOptions
- TransitiveFilter filter : 设置遍历条件，当遍历到的节点不符合 filter 时停止(注意第一个遍历对象就是 rootOp)

- bool inclusive : 返回的 sliceSetVec中 是否包含 rootOp

**ForwardSliceOptions** : using ForwardSliceOptions = SliceOptions;

**BackwardSliceOptions** : 相比 SliceOptions 多一个参数 ` bool omitBlockArguments`，这个参数控制是否避免遍历 blockArguement

```cpp
BackwardSliceOptions sliceOptions;
// 不遍历 blockArg(可以理解为到blockArg这就结束)
sliceOptions.omitBlockArguments = true;

// 所有加入backwardSlice的op都需要满足以下条件
// 第一下会遍历本身
sliceOptions.filter = [rootOp](Operation *slice) -> bool {
  return !llvm::isa<arith::ConstantOp, tensor::EmptyOp, memref::AllocOp，
                    scf::ForallOp, scf::ForOp, scf::IfOp>(slice)
          && rootOp->isProperAncestor(slice);
};

SmallVector<Operation *> backwardSlice;
getBackwardSlice(targetOp, &backwardSlice, sliceOptions);
```