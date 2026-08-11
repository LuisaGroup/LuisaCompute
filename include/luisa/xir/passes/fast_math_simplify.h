#pragma once

#include <cstddef>

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct FastMathSimplifyOptions {
    bool enable_fast_math{false};
};

struct FastMathSimplifyInfo {
    size_t identity_count{0u};
    size_t radix_pow_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return identity_count != 0u || radix_pow_count != 0u;
    }
};

// Canonicalizes only identities whose complete f32 domain, including NaN,
// infinity, signed zero, and subnormal behavior, is covered by the fast-math
// contract. Precise mode is an explicit no-op.
[[nodiscard]] LUISA_XIR_API FastMathSimplifyInfo
fast_math_simplify_pass_run_on_function(
    Function *function,
    FastMathSimplifyOptions options = {}) noexcept;

[[nodiscard]] LUISA_XIR_API FastMathSimplifyInfo
fast_math_simplify_pass_run_on_module(
    Module *module, FastMathSimplifyOptions options = {},
    PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
