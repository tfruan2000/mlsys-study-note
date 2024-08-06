# [MLIR] PDLL

用pdll来实现pattern的match和rewrite


```cpp
// PatternRuleImpl.cpp
// 定义这些rewrite和constrain实现
static Operation *buildOpImpl(PDLResultList &results, Value value) {
  // insert special rewrite logic here.
  Operation *resultOp = ...;
  return resultOp;
}

void registerRuleFunctions(RewritePatternSet &patterns) {
  // patterns.getPDLPatterns().registerRewriteFunction("BuildOp", buildOpImpl);
  // 或者采用下面的形式
  auto &patternModule = patterns.getPDLPatterns();

#define RegRewrite(name)                                                       \
  patternModule.registerRewriteFunction(#name, name##Impl)

#define RegConstraints(name)                                                   \
  patternModule.registerConstraintFunction(#name, name##Impl)

  RegRewrite(tileOp);
  RegConstraints(tilingLoopSizeLimit);
}
```

## 辅助函数

```cpp
// patternUtils.h
inline Value getAnchorHandle(PatternRewriter &rewriter, Value &funcHandle,
                             Operation *op) {
  auto *context = rewriter.getContext();
  return rewriter.create<transform::MatchOp>(
      rewriter.getUnknownLoc(),
      /*results=*/pdl::OperationType::get(context),
      /*target=*/funcHandle,
      /*ops=*/
      ArrayAttr::get(context,
                     {StringAttr::get(context, op->getName().getStringRef())}),
      /*interface=*/nullptr,
      /*op_attrs=*/
      DictionaryAttr::get(context,
                          rewriter.getNamedAttr(attrName,
                                                op->getAttr(attrName))),
      /*filter_result_type=*/nullptr);
}

// 找到对funcOp的matchOp
inline transform::MatchOp matchFuncOp(Operation *targetOp) {
  // Get the parent module op of targetOp.
  transform_ext::MatchOp retOp{};
  if (auto module = targetOp->getParentOfType<ModuleOp>()) {
    auto seqOp = getTheLastTransformSequence(module);
    if (seqOp) {
      auto &block = seqOp.getRegion().front();
      for (auto iter = block.rbegin(); iter != block.rend(); ++iter) {
        if (auto matchOp = dyn_cast_if_present<transform::MatchOp>(*iter)) {
          retOp = matchOp;
          break;
        }
      }
    }
  }
  return retOp;
}

inline Value getFuncHandleAndSetRewriter(PatternRewriter &rewriter,
                                              Operation *targetOp) {
  Value ret(nullptr);
  auto matchOp = matchFuncOp(targetOp);
  if (matchOp) {
    ret = matchOp.getResult();
    rewriter.setInsertionPointAfter(matchOp);
  }
  assert(ret && "invalid rewrite func handle.");
  return ret;
}
```

## rewrite pattern impl

```cpp
//===----------------------------------------------------------------------===//
// rewrite methods
//===----------------------------------------------------------------------===//

static void tileOpImpl(PatternRewriter &rewriter, Operation *op,
                       ArrayAttr tileSize) {
  DBGS() << "Enter rewrite rule [tileOp], with target op: ";
  LLVM_DEBUG(op->print(DBGS()));
  Value funcHandle = getFuncHandleAndSetRewriter(rewriter, op); // 先找到对func matchop
  MLIRContext *context = op->getContext();
  auto pdlOpType = pdl::OperationType::get(context);
  Value anchor = getAnchorHandle(rewriter, funcHandle, op); // 生成对target的match op
  tileSize = tileSize.empty() ? nullptr : tileSize;
  rewriter.create<transform::TileToForallOp>(
      rewriter.getUnknownLoc(),
      /*resultTypes=*/TypeRange({pdlOpType, pdlOpType}),
      /*target=*/anchor,
      /*tile_sizes=*/tileSize,);
}

```

## constrain pattern impl

```cpp
static LogicalResult tilingLoopSizeLimitImpl(PatternRewriter &rewriter,
                                             Operation *root,
                                             Attribute loopPosAttr,
                                             Attribute lowNumAttr,
                                             Attribute highNumAttr) {
  DBGS() << "Enter constraint check: [tilingLoopSizeLimitImpl]\n";
	LLVM_DEBUG(op->print(DBGS()))
  auto loopPosIntAttr = loopPosAttr.dyn_cast_if_present<IntegerAttr>();
  auto lowNumIntAttr = lowNumAttr.dyn_cast_if_present<IntegerAttr>();
  auto highNumIntAttr = highNumAttr.dyn_cast_if_present<IntegerAttr>();
  if (!loopPosIntAttr || !lowNumIntAttr || !highNumIntAttr)
    return failure();
  auto loopPos = loopPosIntAttr.getInt();
  auto lowNum = lowNumIntAttr.getInt();
  auto highNum = highNumIntAttr.getInt();
  if (auto tilingInterface = llvm::dyn_cast_if_present<TilingInterface>(root)) {
    auto ranges = tilingInterface.getIterationDomain(rewriter);
    if (loopPos > ranges.size())
      return failure();
    if (ranges[loopPos].size.is<Attribute>()) {
      auto loopLimit =
          ranges[loopPos].size.get<Attribute>().cast<IntegerAttr>().getInt();
      if (loopLimit >= lowNum && loopLimit <= highNum)
        return success();
    }
  }
  return failure();
}
```

## pdll文件定义pattern

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
  rewrite root with {
    tileOp(root, attr<"[1, 6, 1, 4, 1, 1, 1]">);
  };
}
```

## pass中parse pdll并解析

```cpp
// 输入为.pdll文件的路径
// 参考：mlir/lib/Tools/mlir-pdll-lsp-server/PDLLServer.cpp
static inline OwningOpRef<ModuleOp>
parsePdllSourceFile(llvm::StringRef pdllSource, MLIRContext *context) {
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(pdllSource, &errorMessage);
  if (!memoryBuffer)
    return OwningOpRef<ModuleOp>();
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module *> module = parsePDLLAST(astContext, sourceMgr);
  if (failed(module))
    return OwningOpRef<ModuleOp>();
  return codegenPDLLToMLIR(context, astContext, sourceMgr, **module);
}
```