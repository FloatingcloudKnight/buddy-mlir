//====- CBTransposeFusionVectorization.cpp
//----------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pooling vectorization.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include "mlir/Dialect/Math/IR/Math.h"
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// PoolingNhwcSum vectorization pattern
class TransposeTestPattern : public ConversionPattern {
public:
  explicit TransposeTestPattern(MLIRContext *context) : ConversionPattern(tosa::MatMulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::cout << "TransposeTestPattern::matchAndRewrite called" << std::endl;
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOpResult(0);

    auto reshapeBOp = B.getDefiningOp<tosa::ReshapeOp>();
    if (!reshapeBOp){
      std::cout << "ReshapeOp not found for B"<< "\n";
      return failure();
    }
    reshapeBOp->dump();
    auto transposeBOp =
        reshapeBOp.getOperand().getDefiningOp<tosa::TransposeOp>();
    if (!transposeBOp){
      std::cout << "TransposeOp not found for B"<< "\n";
      return failure();
    }
    transposeBOp->dump();

    auto reshapeCUserIt = C.getUsers().begin();
    if (reshapeCUserIt == C.getUsers().end()){
      std::cout << "ReshapeOp user not found for C"<< "\n";
      return failure();
    }
    Operation *reshapeCOp = *reshapeCUserIt;
    if (!isa<tosa::ReshapeOp>(reshapeCOp)){
      std::cout << "ReshapeOp not found for C"<< "\n";
      return failure();
    }
    reshapeCOp->dump();

    auto transposeCUserIt = reshapeCOp->getOpResult(0).getUsers().begin();
    if (transposeCUserIt == reshapeCOp->getOpResult(0).getUsers().end()){
      std::cout << "ReshapeOp user not found for C"<< "\n";
      return failure();
    }
    Operation *transposeCOp = *transposeCUserIt;
    if (!isa<tosa::TransposeOp>(transposeCOp)){
      std::cout << "TransposeOp not found for C"<< "\n";
      return failure();
    }
    transposeCOp->dump();
    
    return failure();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TransposeTestPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class TransposeTestPass
    : public PassWrapper<TransposeTestPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeTestPass)
  StringRef getArgument() const final {
    return "transpose-test";
  }
  StringRef getDescription() const final {
    return "Transpose Test.";
  }
  TransposeTestPass() = default;
  TransposeTestPass(const TransposeTestPass &) {}
  explicit TransposeTestPass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vector-size",
                          llvm::cl::desc("Affine Vector size."),
                          llvm::cl::init(32)};
};
} // end anonymous namespace.

void TransposeTestPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, memref::MemRefDialect,
                         linalg::LinalgDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<TransposeTestPattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerTransposeTestPass() {
  PassRegistration<TransposeTestPass>();
}
} // namespace buddy
} // namespace mlir
