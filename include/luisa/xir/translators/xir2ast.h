#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}// namespace detail

namespace xir {

class LUISA_XIR_API XIR2AST {
public:
    [[nodiscard]] static luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> build(const KernelFunction *kernel) noexcept;
    [[nodiscard]] static luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> build(const CallableFunction *callable) noexcept;
};

}// namespace xir
}// namespace luisa::compute
