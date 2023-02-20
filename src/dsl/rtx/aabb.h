//
// Created by Mike Smith on 2023/2/13.
//

#pragma once

#include <dsl/builtin.h>
#include <dsl/struct.h>
#include <runtime/rtx/aabb.h>

// clang-format off
LUISA_STRUCT(luisa::compute::AABB, packed_min, packed_max) {
    [[nodiscard]] auto min() const noexcept {
        return make_float3(packed_min[0], packed_min[1], packed_min[2]);
    }
    [[nodiscard]] auto max() const noexcept {
        return make_float3(packed_max[0], packed_max[1], packed_max[2]);
    }
};
// clang-format on
