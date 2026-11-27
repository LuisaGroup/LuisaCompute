#pragma once

#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/coro_func.h>

namespace luisa::compute::xir {
class Module;
class FunctionDefinition;
}// namespace luisa::compute::xir

namespace luisa::compute::detail {

struct CoroCompileCacheLookup {
    bool hit{false};
    bool store_allowed{false};
    uint64_t key_hi{0u};
    uint64_t key_lo{0u};
};

[[nodiscard]] CoroCompileCacheLookup coro_compile_cache_try_load(
    const FunctionBuilder &source_builder,
    CoroutineCompileResult &out_result) noexcept;

void coro_compile_cache_store(
    uint64_t key_hi,
    uint64_t key_lo,
    const CoroutineCompileResult &result,
    const xir::Module *xir_module,
    luisa::span<const xir::FunctionDefinition *const> continuation_definitions) noexcept;

}// namespace luisa::compute::detail
