#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Memoize shared dependencies across the roots of one immutable XIR-to-AST batch.
class AstXirCacheFingerprint {
private:
    luisa::unordered_map<const Value *, luisa::optional<uint64_t>> _globals;
    [[nodiscard]] luisa::optional<uint64_t> _global(const Value *value) noexcept;
    [[nodiscard]] luisa::optional<uint64_t> _function(const Function *function) noexcept;

public:
    [[nodiscard]] auto operator()(const FunctionDefinition &root) noexcept { return _global(&root); }
};

}// namespace luisa::compute::xir
