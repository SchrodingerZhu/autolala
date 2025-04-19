#pragma once
#include <mlir-c/AffineMap.h>
#include <sys/types.h>

namespace raffine {

MlirAffineMap forOpGetLowerBoundMap(MlirOperation forOp);
MlirAffineMap forOpGetUpperBoundMap(MlirOperation forOp);
ssize_t forOpGetStep(MlirOperation forOp);

} // namespace raffine
