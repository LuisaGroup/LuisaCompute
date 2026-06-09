#pragma once

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

struct CoroMaterializeInfo {
    size_t register_count{0u};
    size_t frame_field_count{0u};
    size_t callable_count{0u};
    size_t load_inserted_count{0u};
    size_t store_inserted_count{0u};
    size_t suspend_lowered_count{0u};
    size_t resume_lowered_count{0u};
    size_t terminal_lowered_count{0u};

    struct TransitionEdge {
        size_t from_scope{0u};
        size_t to_scope{0u};
        luisa::vector<size_t> load_fields;
        luisa::vector<size_t> store_fields;
    };
    luisa::vector<TransitionEdge> edges;
    luisa::unordered_map<luisa::string, size_t> name_to_field;
    luisa::unordered_map<luisa::string, const Type *> name_to_type;
};

[[nodiscard]] LUISA_XIR_API CoroMaterializeInfo coro_materialize_pass_run_on_module(Module *m) noexcept;

}// namespace luisa::compute::xir
