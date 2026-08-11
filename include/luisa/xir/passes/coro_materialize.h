#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class Type;
}

namespace luisa::compute::xir {

class Module;
class Value;
struct CoroCfgDistillResult;
struct CoroSplitInfo;

struct CoroMaterializeInfo {
    struct FrameField {
        luisa::string name;
        const Type *type{nullptr};
        size_t index{0u};
    };

    size_t register_count{0u};
    size_t frame_field_count{0u};
    size_t callable_count{0u};
    size_t load_inserted_count{0u};
    size_t store_inserted_count{0u};
    size_t suspend_lowered_count{0u};
    size_t resume_lowered_count{0u};
    size_t terminal_lowered_count{0u};
    // Coro materialization is an unstructured-CFG-only transform. A non-zero
    // value means the complete request was rejected before any IR was mutated.
    size_t structured_cfg_error_count{0u};
    size_t invalid_input_error_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return callable_count != 0u ||
               load_inserted_count != 0u ||
               store_inserted_count != 0u ||
               suspend_lowered_count != 0u ||
               resume_lowered_count != 0u ||
               terminal_lowered_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return structured_cfg_error_count == 0u && invalid_input_error_count == 0u;
    }

    struct TransitionEdge {
        size_t from_scope{0u};
        size_t to_scope{0u};
        luisa::vector<size_t> load_fields;
        luisa::vector<size_t> store_fields;
    };
    luisa::vector<TransitionEdge> edges;
    // frame_fields describes physical storage exactly once per field.
    // name_to_field/name_to_type describe physical field aliases and may
    // contain several names for one interference-colored field. A logical
    // Boolean packed into a uint bit lane consequently has uint here: its
    // Boolean extraction/insertion is already explicit in split XIR.
    luisa::vector<FrameField> frame_fields;
    luisa::unordered_map<luisa::string, size_t> name_to_field;
    luisa::unordered_map<luisa::string, const Type *> name_to_type;
};

// These entry points require destructure_cfg first. Structured SwitchInst is
// converted to raw IndexedBranchInst. Rejection is atomic:
// no matching callable in the module is materialized. The split-aware overload
// also rejects missing/duplicate/out-of-range scopes, duplicate/null/foreign
// callables, and a frame argument that is not the callable's own reference
// argument.
[[nodiscard]] LUISA_XIR_API CoroMaterializeInfo coro_materialize_pass_run_on_module(Module *m) noexcept;
[[nodiscard]] LUISA_XIR_API CoroMaterializeInfo coro_materialize_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg) noexcept;
[[nodiscard]] LUISA_XIR_API CoroMaterializeInfo coro_materialize_pass_run_on_module_with_cfg(
    Module *m, const CoroCfgDistillResult &cfg, const CoroSplitInfo &split) noexcept;

}// namespace luisa::compute::xir
