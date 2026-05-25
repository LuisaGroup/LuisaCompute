#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coro/coro_frame.h>

namespace luisa::compute::xir {

class CallableFunction;

struct MaterializedCoroScope {
    uint32_t token{0u};
    CallableFunction *function{nullptr};
    luisa::vector<uint32_t> input_fields;
    luisa::vector<uint32_t> output_fields;
    luisa::vector<uint32_t> target_tokens;
};

struct MaterializeCoroResult {
    CallableFunction *entry{nullptr};
    luisa::vector<uint32_t> entry_input_fields;
    luisa::vector<uint32_t> entry_output_fields;
    luisa::vector<MaterializedCoroScope> scopes;
    const Type *frame_interface_type{nullptr};
    luisa::vector<CoroFrameFieldInfo> frame_fields;
    luisa::vector<CoroDesignatedFieldInfo> designated_fields;
    luisa::unordered_map<luisa::string, uint32_t> named_tokens;
};

[[nodiscard]] LUISA_XIR_API MaterializeCoroResult
materialize_coro_pass_run_on_function(CallableFunction *function) noexcept;

}// namespace luisa::compute::xir
