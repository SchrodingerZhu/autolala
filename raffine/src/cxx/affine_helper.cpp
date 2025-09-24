#include <memory>
#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/IntegerSet.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <rust/cxx.h>
#include <stdexcept>

#include "affine_helper.hpp"

using namespace mlir;
namespace raffine {

MlirAffineMap forOpGetLowerBoundMap(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op))
    return wrap(forOp.getLowerBoundMap());
  throw std::invalid_argument(
      "Expected an AffineForOp, but got a different operation.");
}

MlirAffineMap forOpGetUpperBoundMap(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op))
    return wrap(forOp.getUpperBoundMap());
  throw std::invalid_argument(
      "Expected an AffineForOp, but got a different operation.");
}

ssize_t forOpGetStep(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op))
    return forOp.getStep().getSExtValue();
  throw std::invalid_argument(
      "Expected an AffineForOp, but got a different operation.");
}

size_t loadStoreOpGetAccessId(MlirOperation target) {
  Operation *op = unwrap(target);
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
    return llvm::bit_cast<size_t>(loadOp.getMemref().getAsOpaquePointer());
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    return llvm::bit_cast<size_t>(storeOp.getMemref().getAsOpaquePointer());
  throw std::invalid_argument(
      "Expected an AffineLoadOp/AffineStoreOp, but got a different operation.");
}

MlirAffineMap loadStoreOpGetAccessMap(MlirOperation target) {
  Operation *op = unwrap(target);
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
    return wrap(loadOp.getAffineMap());
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    return wrap(storeOp.getAffineMap());
  throw std::invalid_argument(
      "Expected an AffineLoadOp/AffineStoreOp, but got a different operation.");
}

MlirIntegerSet ifOpGetCondition(MlirOperation ifOp) {
  Operation *op = unwrap(ifOp);
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op))
    return wrap(ifOp.getCondition());
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}

MlirBlock ifOpGetThenBlock(MlirOperation ifOp) {
  Operation *op = unwrap(ifOp);
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op))
    return wrap(ifOp.getThenBlock());
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}
MlirBlock ifOpGetElseBlock(MlirOperation ifOp) {
  Operation *op = unwrap(ifOp);
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
    if (ifOp.hasElse())
      return wrap(ifOp.getElseBlock());
    else
      return wrap(static_cast<Block *>(nullptr));
  }
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}
rust::Vec<MlirValue> forOpGetLowerBoundOperands(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    rust::Vec<MlirValue> result;
    for (auto val : forOp.getLowerBoundOperands())
      result.push_back(wrap(val));
    return result;
  }
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}
rust::Vec<MlirValue> forOpGetUpperBoundOperands(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    rust::Vec<MlirValue> result;
    for (auto val : forOp.getUpperBoundOperands())
      result.push_back(wrap(val));
    return result;
  }
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}
MlirValue forOpGetInductionVar(MlirOperation forOp) {
  Operation *op = unwrap(forOp);
  if (auto forOp = dyn_cast<affine::AffineForOp>(op))
    return wrap(forOp.getInductionVar());
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}
bool mlirValueProperlyDominatesOperation(MlirValue value,
                                         MlirOperation operation,
                                         const DominanceInfo &dom) {
  Operation *op = unwrap(operation);
  Value val = unwrap(value);
  return dom.properlyDominates(val, op);
}
std::unique_ptr<DominanceInfo> createDominanceInfo(MlirModule op) {
  ModuleOp operation = unwrap(op);
  return std::make_unique<DominanceInfo>(operation);
}
rust::Vec<MlirValue> loadStoreOpGetAffineOperands(MlirOperation target) {
  Operation *op = unwrap(target);
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    rust::Vec<MlirValue> result;
    for (auto val : loadOp.getIndices())
      result.push_back(wrap(val));
    return result;
  }
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    rust::Vec<MlirValue> result;
    for (auto val : storeOp.getIndices())
      result.push_back(wrap(val));
    return result;
  }
  throw std::invalid_argument(
      "Expected an AffineLoadOp/AffineStoreOp, but got a different operation.");
}
rust::Vec<MlirValue> ifOpGetConditionOperands(MlirOperation ifOp) {
  Operation *op = unwrap(ifOp);
  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
    rust::Vec<MlirValue> result;
    for (auto val : ifOp->getOperands())
      result.push_back(wrap(val));
    return result;
  }
  throw std::invalid_argument(
      "Expected an AffineIfOp, but got a different operation.");
}

bool definedInAnyLoop(MlirValue value) {
  Value val = unwrap(value);
  
  Operation *op = nullptr;
  
  // If the value is a block argument, get its block parent
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Block *block = blockArg.getOwner();
    // Check if the block itself is from an affine for op
    if (block->getParentOp() && isa<affine::AffineForOp>(block->getParentOp()))
      return true;
    op = block->getParentOp();
  } else
    op = val.getDefiningOp();
  
  
  // If no op found in either case, return false
  if (!op) 
    return false;
  
  
  // Return true iff the op has an affine for op as its parent
  return op->getParentOfType<affine::AffineForOp>() != nullptr;
}

} // namespace raffine
