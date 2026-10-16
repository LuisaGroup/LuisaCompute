#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Argument;
class Module;

using UniqueResourceOriginMap = luisa::unordered_map<const Argument *, const Argument *>;

// Proves that a resource argument is the unchanged descriptor of one kernel
// resource argument. Kernel roots map to themselves. Callable formals are
// included only when every CallInst and RayQueryPipelineInst callback edge
// resolves to that same root, with the same type. Pipeline capture i maps to
// callback formal i + 1; formal zero is the query reference.
//
// All owned blocks are considered, including unreachable blocks. Unsupported
// function uses, unknown/computed actual values, missing callers, conflicting
// roots and cyclic resource dependencies have no entry. This is not a proof
// that the resource contents are immutable or that two arguments do not alias.
//
// A null module yields an empty map. The analysis does not modify the module;
// returned pointers remain valid only while their arguments remain alive.
[[nodiscard]] LUISA_XIR_API UniqueResourceOriginMap analyze_unique_resource_origins(const Module *module) noexcept;

}// namespace luisa::compute::xir
