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

- collectUnderlyingAddressValues
  - (Value, SmallVectorImpl<Value> &output)
  - (OpResult result, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)
    - result.getOwner() -> ViewLikeOpInterface -> 继续调用 viewOp.getViewSource()
    - result.getOwner() -> RegionBranchOpInterface
  - (BlockArguement arg, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)

## SliceAnalysis

用来遍历 use-def 链的 analysis

- getForwardSlice : 获得root op的use链

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

- getBackWardSlice : 获得root op的def链

```cpp
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

**BackwardSliceOptions**

```cpp
BackwardSliceOptions sliceOptions;
sliceOptions.omitBlockArguments = true;
// 所有加入backwardSlice的op都需要满足以下条件
// 第一下会遍历本身
sliceOptions.filter = [rootOp](Operation *slice) -> bool {
  return !llvm::isa<memref::AllocOp>(slice) && rootOp->isProperAncestor(slice);
};

SmallVector<Operation *> backwardSlice;
getBackwardSlice(targetOp, &backwardSlice, sliceOptions);
```