#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coro/coro_graph.h>

namespace luisa::compute {
class Type;
}

namespace luisa::compute::xir {

struct CoroFrameFieldInfo {
    uint32_t frame_index{0u};
    const Type *type{nullptr};
    Value *root{nullptr};
    luisa::vector<uint32_t> chain;
};

struct CoroDesignatedFieldInfo {
    uint32_t frame_index{0u};
    luisa::string name;
    const Type *type{nullptr};
    Value *value{nullptr};
};

struct CoroFrame {
    const Type *interface_type{nullptr};
    luisa::vector<CoroFrameFieldInfo> fields;
    luisa::vector<CoroDesignatedFieldInfo> designated_fields;
};

[[nodiscard]] LUISA_XIR_API CoroFrame compute_coro_frame(const CoroGraph &graph,
                                                         const CoroTransitionGraph &transition_graph) noexcept;

}// namespace luisa::compute::xir
