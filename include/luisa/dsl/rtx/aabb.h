#pragma once

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/rtx/aabb.h>

LUISA_STRUCT(luisa::compute::AABB, packed_min, packed_max) {
    [[nodiscard]] auto min() const noexcept {
        return make_float3(packed_min[0], packed_min[1], packed_min[2]);
    }
    [[nodiscard]] auto max() const noexcept {
        return make_float3(packed_max[0], packed_max[1], packed_max[2]);
    }
};
