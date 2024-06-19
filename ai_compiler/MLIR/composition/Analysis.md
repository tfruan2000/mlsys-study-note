# [MLIR]Analysis

## AliasAnalysis

## LocalAliasAnalysis

```bash
mlir/include/mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h
```

- AliasResult
  - NoAlias
  - MayAlias
  - PartialAlias
  - MustAlias

- AliasResult alias(Value lhs, Value rhs);
