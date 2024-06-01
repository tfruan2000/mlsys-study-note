# Dialect

新增一个dialect可以参考最近mlir中新增的[polynomial dialect](https://github.com/llvm/llvm-project/commit/55b6f17071d25b77fcdc910ca9b15f89305137e0) ，然后就是补充各种dialect2dialect的conversion了

## 组成

详见 [MLIR_Survey](../MLIR_Survey.md ':include') 的第二节 `dialect 和 operation` 相关的介绍


## DialectRegistry

The DialectRegistry maps a dialect namespace to a constructor for the matching dialect ：看起来像为dialect中的op外挂新的属性

```cpp
mlir/include/mlir/IR/DialectRegistry.h
```

例如为linalg的op挂上新的interface

```cpp
void mlir::xxx::utils::registerLinalgAggregatedOpInterfaceModel(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LinalgDialect *dialect) {
    linalg::MapOp::attachInterface<MapOpInterface>(*ctx);
    MatmulOp::attachInterface<
        MatmulOpInterface<MatmulOp, linalg::Conv2DNhwcFhwcOp>>(*ctx);
    BatchMatmulOp::attachInterface<
        MatmulOpInterface<BatchMatmulOp, linalg_ext::BatchConv2DNhwcFhwcOp>>(
        *ctx);
    ReduceOp::attachInterface<ReduceOpInterface>(*ctx);
  });
}

// 定义上例如，其中AggregatedOpInterface需要在LinalgExtInterface.td定义
template <typename SrcOpTy, typename DstOpTy>
struct MatmulOpInterface : public AggregatedOpInterface::ExternalModel<
                               MatmulOpInterface<SrcOpTy, DstOpTy>, SrcOpTy> {
  FailureOr<SmallVector<Operation *>>
  decomposeOperation(Operation *op, Operation *value,
                     PatternRewriter &rewriter) const {
	}
};
```

## 举例

### linalg

#### op

- linalg.generic
- linalg.fill
- linalg.map{ arith.op / math.op }

```cpp
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, adaptor.getOperands().front(), emptyTensor,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Type elementType = getElementTypeOrSelf(emptyTensor);
          Value operand = args.front();
          Value innerResult =
              elementType.isa<FloatType>()
                  ? rewriter.create<math::AbsFOp>(loc, elementType, operand)
                        .getResult()
                  : rewriter.create<math::AbsIOp>(loc, elementType, operand)
                        .getResult();
          b.create<linalg::YieldOp>(loc, innerResult);
        });
```

- linalg.matmul
- linalg.batch_matmul

#### function

- LinalgInterface
  - bool hasDynamicShape()
  - SmallVector<AffineMap> getIndexingMapsArray()
    ```cpp
    // 判断linalgOp是ElementwiseOp
    auto isElementwiseLinalg = [](linalg::LinalgOp linalgOp) -> bool {
      if (linalgOp.getNumDpsInints() != 1)
        return false;
      return llvm::all_of(linalgOp.getIndexingMapsArray(), [](AffineMap map) {
        return map.isIdentity();
      }) &&
          hasOnlyScalarElementwiseOp(linalgOp->getRegion(0));
    };
    ```

#### LinalgInterface

```bash
mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp
mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td
```

- getNumLoops() -> unsigned

即返回 getIteratorTypesArray().size()

- getNumParallelLoops

返回 loops 中 parallel轴的数量，这些轴一般可以并行(用`scf.forall`来tile)，而reduction轴都只能用`scf.for`来tile

- getIndexingMapsArray

返回region内的计算。generic op内部是由一堆的计算组成的，即可以看成一个`AffineMap`。

- payloadUsesValueFromOperand

输入是 `OpOperand`，返回这个 `OpOperand` 是否被使用，由此来获得准确 `Memory-Effect`。(inputOperand有user则有read，initOperand必被write，若有user则有read)

例如 https://github.com/llvm/llvm-project/pull/92079/files 中

```cpp
static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    LinalgOp linalgOp) {
  SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
  for (auto [index, operand] : llvm::enumerate(inputOperands)) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(&linalgOp->getOpOperand(index))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
  }
  unsigned inputOperandSize = inputOperands.size();

  for (auto [index, operand] : llvm::enumerate(linalgOp.getDpsInits())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(
            &linalgOp->getOpOperand(index + inputOperandSize))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

```

#### conversion

强烈推荐项目 [triton-linalg](https://github.com/Cambricon/triton-linalg)

### scf

```cpp
mlir/lib/Dialect/SCF/IR/SCF.cpp
```

#### op

- scf.for
- scf.forall / scf.parallel ： 循环body的程序是可以的并发执行，没有前后依赖的
  可以使用多线程的方式来执行，线程的id就是循环的迭代变量
  从scf到launch这种转换是可以通过代码自动完成的，需要的额外信息就是每一个循环的轴到launch的轴的映射关系

    ```llvm
    scf.forall (%thread_id_1, %thread_id_2) in (%num_threads_1, %num_thread_2) {
             // ...
          }
        }
    ```

- scf.if

	```cpp
	Block *IfOp::thenBlock() { return &getThenRegion().back(); }
	YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }

	auto cond = op.getCondition();
	auto thenYieldArgs = op.thenYield().getOperands();
	auto elseYieldArgs = op.elseYield().getOperands();
	```

	有一个 `scf.if` 的canonicalize pattern，叫 `ConvertTrivialIfToSelect`，可以尽量消除 else region

	经常在 `bufferize` 后的 `canonicalize` 起效，因为`bufferize` 后 `scf.yield` 的operand更关系更明确了


	```llvm
	// ./build/bin/mlir-opt test_if.mlir --split-input-file --one-shot-bufferize --canonicalize

	// 不能命中，因为thenRegion的yield value属于thenRegion
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// %2 = scf.if %1 -> (memref<2xi32>) {
	//   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
	//   linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc_0 : memref<2xi32>)
	//   scf.yield %alloc_0 : memref<2xi32>
	// } else {
	//   scf.yield %alloc : memref<2xi32>
	// }
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %4 = tensor.empty() : tensor<2xi32>
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%4: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %1 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}

	// -----
	// 可以命中，但不产生select，因为trueVal == falseVal
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// scf.if %1 {
	//    linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc : memref<2xi32>)
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%1: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %1 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}

	// -----
	// 产生select
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// %2 = arith.select %1, %alloc, %alloc_0 : memref<2xi32>
	// scf.if %1 {
	//  linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc : memref<2xi32>)
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %cst1 = arith.constant 1 :i32
	  %6 = tensor.empty() : tensor<2xi32>
	  %7 = linalg.fill ins(%cst1 : i32) outs(%6 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%1: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %7 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}
	```


### tensor

```cpp
mlir/Dialect/Tensor/IR/Tensor.h
```

#### op

- tensor.empty

```cpp
auto srcShape = srcType.getShape();
SmallVector<int64_t> newShapes(srcShape.begin(), srcShape.end())
Value input = rewriter.create<tensor::EmptyOp>(loc, newShapes, srcType.getElementType());
// RankedTenorType newType = RankedTensorType::get({srcDims[0], 1}), srcType.getElementType)
```

- tensor.extract_slice [$offsets] [$sizes] [$strides]
    - getSource()
    - getResult()
    - getType() → getResult().getType()
- tensor.collapse_shape

```cpp
SmallVector<int64_t> srcDims;
RankedTensorType collapseType = RankedTensorType::get(srcDims, srcType.getElementType());
rewriter.create<tensor::CollapseShapeOp>(loc, collapseType, collapseIn, collapseIndices); // type, value, ArrayRef<ReassociationIndices>
```

- tensor.expend_shape

```cpp
RankedTensorType inputType = RankedTensorType::get({1, srcDims[0], 1, srcDims[1]}, srcType.getElementType());
SmallVector<ReassociationIndices> inputIndices = {{0, 1}, {2, 3}};
Value opInput = rewriter.create<tensor::ExpandShapeOp>(loc, inputType, collapseOut, inputIndices);
```

应用上可以使用tensor.collapse_shape和tensor.expand_shape消除operands中dimSize=1的维（往往这些维度不会影响数据的layout），创建降维后的op时候需要为某些op set额外的属性，例如linalg.transpose的permutation、linalg.reduce和linalg.broadcast的dimensions

### memref

- memref.view
    - getMixedOffsets / getMixedSizes / getMixedStrides → SmallVector<OpFoldResult>

