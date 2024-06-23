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
