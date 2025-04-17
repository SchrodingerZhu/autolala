
#include <mlir/IR/AffineExpr.h>
#include <memory>
namespace autolala {
class Context {
    struct Impl;
    std::unique_ptr<Impl> pImpl;
public:
    Context();
    ~Context();
};
}