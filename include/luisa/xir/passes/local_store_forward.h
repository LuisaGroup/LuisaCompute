#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class PassReport;

class Value;
class LoadInst;
class StoreInst;
class BasicBlock;
class Function;
class Module;

// This pass is used to forward stores to loads for thread-local variables within
// straight-line basic blocks. It is a simple peephole-style optimization but can
// effectively reduce the number of memory operations produced by the C++ DSL.
// Note: we do not remove the stores after forwarding them to loads, as the DCE
// pass should be able to remove them if they are dead.

struct LocalStoreForwardInfo {
    size_t removed_load_count{0u};
};

[[nodiscard]] LUISA_XIR_API LocalStoreForwardInfo local_store_forward_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LocalStoreForwardInfo local_store_forward_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
