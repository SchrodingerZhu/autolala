#pragma once
#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir/IR/Dominance.h>
#include <rust/cxx.h>
#include <sys/types.h>

namespace raffine {
MlirAffineMap forOpGetLowerBoundMap(MlirOperation forOp);
MlirAffineMap forOpGetUpperBoundMap(MlirOperation forOp);
ssize_t forOpGetStep(MlirOperation forOp);
size_t loadStoreOpGetAccessId(MlirOperation op);
MlirAffineMap loadStoreOpGetAccessMap(MlirOperation op);
MlirIntegerSet ifOpGetCondition(MlirOperation ifOp);
MlirBlock ifOpGetThenBlock(MlirOperation ifOp);
MlirBlock ifOpGetElseBlock(MlirOperation ifOp);
std::unique_ptr<mlir::DominanceInfo> createDominanceInfo(MlirModule op);
MlirValue forOpGetInductionVar(MlirOperation forOp);
rust::Vec<MlirValue> forOpGetLowerBoundOperands(MlirOperation forOp);
rust::Vec<MlirValue> forOpGetUpperBoundOperands(MlirOperation forOp);
bool mlirValueProperlyDominatesOperation(MlirValue value,
                                         MlirOperation operation,
                                         const mlir::DominanceInfo &dom);
rust::Vec<MlirValue> loadStoreOpGetAffineOperands(MlirOperation target);
rust::Vec<MlirValue> ifOpGetConditionOperands(MlirOperation ifOp);
bool definedInAnyLoop(MlirValue value);
} // namespace raffine
