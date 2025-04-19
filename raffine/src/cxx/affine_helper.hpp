#pragma once
#include <mlir-c/AffineMap.h>
#include <mlir-c/IntegerSet.h>
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
} // namespace raffine
