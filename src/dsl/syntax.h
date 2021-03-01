//
// Created by Mike Smith on 2021/2/27.
//

#pragma once

#include <dsl/var.h>
#include <dsl/expr.h>
#include <dsl/func.h>

namespace luisa::compute::dsl::detail {

struct KernelBuilder {
    template<typename F>
    [[nodiscard]] auto operator<<(F &&def) const noexcept { return Kernel{std::forward<F>(def)}; }
};

struct CallableBuilder {
    template<typename F>
    [[nodiscard]] auto operator<<(F &&def) const noexcept { return Callable{std::forward<F>(def)}; }
};

}// namespace luisa::compute::dsl::detail

#define LUISA_KERNEL ::luisa::compute::dsl::detail::KernelBuilder{} << [&]
#define LUISA_CALLABLE ::luisa::compute::dsl::detail::CallableBuilder{} << []
