#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
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

} // namespace raffine
