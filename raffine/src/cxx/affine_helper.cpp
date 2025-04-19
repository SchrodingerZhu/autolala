#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <stdexcept>

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

} // namespace raffine
