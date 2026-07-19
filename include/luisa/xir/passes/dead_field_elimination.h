#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute {
struct CoroFrameDesc;
}

namespace luisa::compute::xir {

struct CoroMaterializeInfo;

struct DeadFieldEliminationInfo {
    size_t original_field_count{0u};
    size_t eliminated_field_count{0u};
    size_t remaining_field_count{0u};
    size_t original_frame_size{0u};
    size_t new_frame_size{0u};
    size_t invalid_input_error_count{0u};
    luisa::unordered_set<size_t> eliminated_field_indices;

    [[nodiscard]] bool succeeded() const noexcept { return invalid_input_error_count == 0u; }
};

[[nodiscard]] LUISA_XIR_API DeadFieldEliminationInfo dead_field_elimination_pass_run(
    CoroMaterializeInfo &info,
    CoroFrameDesc &desc) noexcept;

}// namespace luisa::compute::xir
