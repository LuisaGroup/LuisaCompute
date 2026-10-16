#pragma once

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {
class Function;
class Argument;
class CallInst;
class XIRBuilder;
namespace detail {

// Pointer-free finite alias domain for static, nonrecursive call graphs.
// Runtime state stores only selectors and captured access-chain indices.
class CoroCallAliasLowering {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    explicit CoroCallAliasLowering(Function *root);
    ~CoroCallAliasLowering() noexcept;
    void add_parameter(Argument *argument, XIRBuilder &prologue);
    void bind_call(CallInst *call, XIRBuilder &builder, XIRBuilder &prologue);
    void lower_uses(XIRBuilder &prologue);
};
}// namespace detail
}// namespace luisa::compute::xir
