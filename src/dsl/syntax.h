//
// Created by Mike Smith on 2021/2/27.
//

#pragma once

#include <dsl/var.h>
#include <dsl/expr.h>
#include <dsl/func.h>

namespace luisa::compute::dsl::detail {

struct KernelFuncBuilder {
    template<typename F>
    [[nodiscard]] auto operator<<(F &&def) const noexcept { return KernelFunc{std::forward<F>(def)}; }
};

struct CallableFuncBuilder {
    template<typename F>
    [[nodiscard]] auto operator<<(F &&def) const noexcept { return CallableFunc{std::forward<F>(def)}; }
};

}// namespace luisa::compute::dsl::detail

#define LUISA_KERNEL ::luisa::compute::dsl::detail::KernelFuncBuilder{} << [&]
#define LUISA_CALLABLE ::luisa::compute::dsl::detail::CallableFuncBuilder{} << []
