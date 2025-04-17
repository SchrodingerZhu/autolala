find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include(${LLVM_DIR}/AddLLVM.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)