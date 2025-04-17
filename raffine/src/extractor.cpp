#include "extractor.hpp"
#include "mlir-c/Support.h"
#include <cstdint>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Dialect.h>
#include <isl/ctx.h>

#include <memory>
#include <numeric>
#include <string>


using namespace mlir;

namespace autolala {
struct Context::Impl {
    MLIRContext mlirContext;
    isl_ctx *islContext;
    Impl() : mlirContext(), islContext(nullptr) {
        // Initialize the MLIR context
        DialectRegistry registry;
        registerAllDialects(registry);
        registerAllExtensions(registry);
        mlirContext.appendDialectRegistry(registry);
        mlirContext.loadAllAvailableDialects();
        // Initialize the ISL context
        islContext = isl_ctx_alloc();
    }
    ~Impl() {
        // Free the ISL context
        if (islContext) {
            isl_ctx_free(islContext);
        }
    }
};
}
