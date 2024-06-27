# [MLIR]Analysis

## AliasAnalysis

## LocalAliasAnalysis

```bash
mlir/include/mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h
mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.h
```

- AliasResult: 两个location之间是否有关系
  - Kind
    - NoAlias
    - MayAlias
    - PartialAlias : 两个loc互相alias，但是部分重叠
    - MustAlias
  - isNO / isMay / isPartial / isMust -> bool

- AliasResult alias(Value lhs, Value rhs);

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

- collectUnderlyingAddressValues
  - (Value, SmallVectorImpl<Value> &output)
  - (OpResult result, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)
    - result.getOwner() -> ViewLikeOpInterface -> 继续调用 viewOp.getViewSource()
    - result.getOwner() -> RegionBranchOpInterface
  - (BlockArguement arg, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)

## SliceAnalysis

用来遍历 use-def 链的 analysis

```bash
///    ____
///    \  /  defs (in some topological order)
///     \/
///     op
///     /\
///    /  \  uses (in some topological order)
///   /____\
```

- getForwardSlice : 获得root op的use链 (向后寻找)

```bash
/// 从 0 开始追，可以获得 {9, 7, 8, 5, 1, 2, 6, 3, 4}
///               0
///    ___________|___________
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
```

输入， root可以是op，也可以是value
```cpp
void getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});

void getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});
```

- getBackWardSlice : 获得root op的def链 (往前回溯)

```bash
/// 从 node 8 开始， 可以获得 {1, 2, 5, 3, 4, 6}
/// ============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
```

```cpp
void getBackwardSlice(Operation *op, SetVector<Operation *> *bac
                      const BackwardSliceOptions &options = {});

void getBackwardSlice(Value root, SetVector<Operation *> *backwa
                      const BackwardSliceOptions &options = {});
```

- SliceOptions
  - TransitiveFilter filter : 设置遍历条件，当遍历到的节点不符合 filter 时停止(注意第一个遍历对象就是 rootOp)
  - bool inclusive : 返回的 sliceSetVec中 是否包含 rootOp

**ForwardSliceOptions** : using ForwardSliceOptions = SliceOptions;

**BackwardSliceOptions** : 相比 SliceOptions 多一个参数 ` bool omitBlockArguments`，这个参数控制是否避免遍历 blockArguement

```cpp
BackwardSliceOptions sliceOptions;
sliceOptions.omitBlockArguments = true; // 不遍历 blockArg(可以理解为到这就结束)
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