# [MLIR] PDLL

用pdll来实现pattern的match和rewrite

```cpp
// PatternRuleImpl.cpp
// 定义这些rewrite和constrain实现
void mlir::genesis::auto_genesis::registerRuleFunctions(
    RewritePatternSet &patterns) {
  auto &patternModule = patterns.getPDLPatterns();

#define RegRewrite(name)                                                       \
  patternModule.registerRewriteFunction(#name, name##Impl)

#define RegConstraints(name)                                                   \
  patternModule.registerConstraintFunction(#name, name##Impl)

  RegRewrite(tile);
  
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
                          rewriter.getNamedAttr(opIndexingIdName,
                                                op->getAttr(opIndexingIdName))),
      /*filter_result_type=*/nullptr);
}

// 找到对funcOp的matchO
inline transform::MatchOp sniffFuncMatchOp(Operation *targetOp) {
  // Get the parent module op of targetOp.
  transform_ext::MatchOp retOp{};
  if (auto module = targetOp->getParentOfType<ModuleOp>()) {
    auto seqOp = getTheLastTransformSequence(module);
    if (seqOp) {
      auto &block = seqOp.getRegion().front();
      for (auto iter = block.rbegin(); iter != block.rend(); ++iter) {
        if (auto matchOp = dyn_cast_or_null<transform::MatchOp>(*iter)) {
          retOp = matchOp;
          break;
        }
      }
    }
  }
  return retOp;
}

inline Value getFuncMatchHandleAndSetRewriter(PatternRewriter &rewriter,
                                              Operation *targetOp) {
  Value ret(nullptr);
  auto matchOp = sniffFuncMatchOp(targetOp);
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

static void tileImpl(PatternRewriter &rewriter, Operation *op,
                     ArrayAttr numThreads, ArrayAttr tileSize,
                     ArrayAttr threadDimMapping) {
  DBGS() << "Enter rewrite rule [tile], with target op: ";
  LLVM_DEBUG(op->print(DBGS()));
  Value funcHandle = getFuncMatchHandleAndSetRewriter(rewriter, op); // 先找到对func matchop
  MLIRContext *context = op->getContext();
  auto pdlOpType = pdl::OperationType::get(context);
  Value anchor = getAnchorHandle(rewriter, funcHandle, op); // 生成对target的match op
  numThreads = numThreads.empty() ? nullptr : numThreads;
  tileSize = tileSize.empty() ? nullptr : tileSize;
  rewriter.create<transform::TileToForallOp>(
      rewriter.getUnknownLoc(),
      /*resultTypes=*/TypeRange({pdlOpType, pdlOpType}),
      /*target=*/anchor,
      /*num_threads=*/numThreads,
      /*tile_sizes=*/tileSize,
      /*mapping=*/ArrayAttr::get(context, threadDimMapping));
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
  auto loopPosIntAttr = loopPosAttr.dyn_cast_or_null<IntegerAttr>();
  auto lowNumIntAttr = lowNumAttr.dyn_cast_or_null<IntegerAttr>();
  auto highNumIntAttr = highNumAttr.dyn_cast_or_null<IntegerAttr>();
  if (!loopPosIntAttr || !lowNumIntAttr || !highNumIntAttr)
    return failure();
  auto loopPos = loopPosIntAttr.getInt();
  auto lowNum = lowNumIntAttr.getInt();
  auto highNum = highNumIntAttr.getInt();
  if (auto tilingInterface = llvm::dyn_cast_or_null<TilingInterface>(root)) {
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
// ==== RewriteRules ======================
// @param[in] op 目标算子.
// @param[in] numOfThread 拆分的份数.
// @param[in] tileSize 拆分后的大小.
// @param[in] threadDimMapping 映射到哪些轴上.
Rewrite tile(op: Op, numOfThread: Attr, tileSize: Attr, threadDimMapping: Attr);

Rewrite tileTwiceAndRelay(op:Op, operandsToRelay:Attr,
                          numThreadsForFirst:Attr, tileSizeForFirst:Attr, threadDimMappingForFirst:Attr,
                          numThreadsForSecond:Attr, tileSizeForSecond:Attr, threadDimMappingForSecond:Attr);

// ==== Constraints ======================
// 判断某一个维度上的大小是否在一定的范围
// @param[in] op 目标算子.
// @param[in] loopPosAttr 维度的index.
// @param[in] lowNumAttr 范围下限
// @param[in] highNumAttr 范围上限
Constraint tilingLoopSizeLimit(op:Op, loopPosAttr:Attr, lowNumAttr:Attr, highNumAttr:Attr);

// ==== Patterns ======================
Pattern TileParallelofConvOpUseRange with benefit(9) {
  let root = op<linalg.conv_2d_nhwc_fhwc>;
  canTileParallel(root);
  tilingLoopSizeLimit(root, attr<"0">, attr<"1">, attr<"1">);
  tilingLoopSizeLimit(root, attr<"1">, attr<"513">, attr<"2000">);
  tilingLoopSizeLimit(root, attr<"2">, attr<"1">, attr<"1">);
  tilingLoopSizeLimit(root, attr<"3">, attr<"100">, attr<"512">);
  tilingLoopSizeLimit(root, attr<"4">, attr<"1">, attr<"1">);
  tilingLoopSizeLimit(root, attr<"5">, attr<"1">, attr<"1">);
  rewrite root with {
    // tile(root, attr<"[1, 6, 1, 4, 1, 1, 1]">, attr<"[]">, attr<"[8, 8, 8, 8, 8, 8, 8]">);
    tileTwiceAndRelay(root, attr<"[0]">, 
                      attr<"[1, 5, 1, 4, 1, 1, 1]">, attr<"[]">, attr<"[8, 8, 8, 8, 8, 8, 8]">,
                      attr<"[1, 1, 1, 4, 1, 1, 1]">, attr<"[]">, attr<"[1, 1, 1, 1, 1, 1, 1]">);
  };
}

```

## pass中parse pdll并解析

```cpp
static inline OwningOpRef<ModuleOp>
parsePdllSourceFile(llvm::StringRef patternSource, MLIRContext *context) {
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(patternSource, &errorMessage);
  if (!memoryBuffer)
    return OwningOpRef<ModuleOp>();
  // Collect the content from pdll source file.
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