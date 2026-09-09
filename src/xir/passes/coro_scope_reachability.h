#pragma once

#include <cstddef>
#include <cstdint>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {
class BasicBlock;
class FunctionDefinition;
namespace detail {

// A May reachability certificate for one latest-continuation owner. Absence
// proves infeasibility only when valid is true. A chosen successor is a raw
// conditional branch arm proven unique for every arrival in this scope.
struct CoroScopeReachability {
    struct Scope {
        luisa::unordered_set<BasicBlock *> blocks;
        luisa::unordered_map<BasicBlock *, BasicBlock *> selected_successors;
    };
    bool valid{false};
    bool widened{false};
    size_t state_count{0u};
    luisa::unordered_map<uint32_t, Scope> scopes;
};

[[nodiscard]] CoroScopeReachability analyze_coro_scope_reachability(
    FunctionDefinition *definition) noexcept;

}// namespace detail
}// namespace luisa::compute::xir
