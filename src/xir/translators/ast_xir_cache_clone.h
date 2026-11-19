#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class FunctionDefinition;
class Module;

// Import only the requested function and its transitive function operands.
// The target owns everything imported. Keep the immutable source modules alive
// throughout the cloner's lifetime, since its memoization keys use their addresses.
class AstXirCacheCloner final : private InstructionCloneValueResolver {

private:
    Module &_target;
    luisa::unordered_map<const Value *, Value *> _values;

    [[nodiscard]] Function *_import_function(const Function *source) noexcept;
    void _clone_body(const FunctionDefinition *source, FunctionDefinition *target) noexcept;
    [[nodiscard]] Value *_try_resolve(const Value *value) noexcept;
    [[nodiscard]] Value *resolve(const Value *value) noexcept override;

public:
    explicit AstXirCacheCloner(Module &target) noexcept : _target{target} {}
    AstXirCacheCloner(const AstXirCacheCloner &) = delete;
    AstXirCacheCloner &operator=(const AstXirCacheCloner &) = delete;

    [[nodiscard]] FunctionDefinition *import(const FunctionDefinition *source) noexcept;
};

}// namespace luisa::compute::xir
