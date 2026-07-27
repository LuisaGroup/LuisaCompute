#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Function;
class Module;
class LoadInst;
class StoreInst;
class ArithmeticInst;

// This pass converts load/store instructions on aggregate GEPs to extract/insert.
// Specifically, it follows the template below:
// - Load(GEP(agg, indices...)) => Extract(Load(agg), indices...)
// - Store(GEP(agg, indices...), elem) => Store(agg, Insert(Load(agg), elem, indices...))
// This pass is designed to help the mem2reg pass handle aggregates.
// Load/store metadata is cloned onto the corresponding replacement operation.
// Annotated GEPs are retained because one address may feed multiple operations.
// Null inputs are no-ops.

struct TransposeGEPInfo {
    // transpose_gep first canonicalizes nested and no-op GEPs. These are
    // externally visible mutations even when no load/store is transposed.
    size_t traced_gep_count{0u};
    size_t removed_noop_gep_count{0u};
    size_t transposed_load_count{0u};
    size_t transposed_store_count{0u};
    size_t removed_gep_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return traced_gep_count != 0u ||
               removed_noop_gep_count != 0u ||
               transposed_load_count != 0u ||
               transposed_store_count != 0u ||
               removed_gep_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API TransposeGEPInfo transpose_gep_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API TransposeGEPInfo transpose_gep_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
