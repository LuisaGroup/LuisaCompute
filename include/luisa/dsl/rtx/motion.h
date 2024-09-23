//
// Created by Mike on 2024/9/23.
//

#pragma once

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/rtx/motion_transform.h>

LUISA_STRUCT(luisa::compute::MotionInstanceTransformSRT,
             pivot, quaternion, scale, shear, translation) {
    [[nodiscard]] auto pivot_vector() const noexcept {
        return luisa::compute::dsl::make_float3(pivot[0], pivot[1], pivot[2]);
    }
    [[nodiscard]] auto quat_vector() const noexcept {
        return luisa::compute::dsl::make_float4(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    }
    [[nodiscard]] auto scale_vector() const noexcept {
        return luisa::compute::dsl::make_float3(scale[0], scale[1], scale[2]);
    }
    [[nodiscard]] auto shear_vector() const noexcept {
        return luisa::compute::dsl::make_float3(shear[0], shear[1], shear[2]);
    }
    [[nodiscard]] auto translation_vector() const noexcept {
        return luisa::compute::dsl::make_float3(translation[0], translation[1], translation[2]);
    }
};
