#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Function;
class FunctionDefinition;
class CallInst;
class Module;

class LUISA_XIR_API CallGraph {

private:
    luisa::vector<Function *> _root_functions;
    luisa::unordered_map<FunctionDefinition *, luisa::vector<CallInst *>> _call_edges;

public:
    // only for internal use
    void _add_function(Function *f) noexcept;

public:
    [[nodiscard]] luisa::span<Function *const> root_functions() const noexcept;
    [[nodiscard]] luisa::span<CallInst *const> call_edges(FunctionDefinition *f) const noexcept;
};

[[nodiscard]] LUISA_XIR_API CallGraph compute_call_graph(Module *module) noexcept;

}// namespace luisa::compute::xir
