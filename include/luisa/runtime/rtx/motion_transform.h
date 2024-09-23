//
// Created by mike on 9/22/24.
//

#pragma once

#include <luisa/core/basic_types.h>

namespace luisa::compute {

using MotionInstanceTransformMatrix = float4x4;

struct alignas(16) MotionInstanceTransformSRT {
    float pivot[3] = {0.f, 0.f, 0.f};
    float quaternion[4] = {0.f, 0.f, 0.f, 1.f};
    float scale[3] = {1.f, 1.f, 1.f};
    float shear[3] = {0.f, 0.f, 0.f};
    float translation[3] = {0.f, 0.f, 0.f};
};

struct alignas(16) MotionInstanceTransform {

    float data[16] = {};

    MotionInstanceTransform() noexcept = default;
    explicit MotionInstanceTransform(const MotionInstanceTransformMatrix &m) noexcept;
    explicit MotionInstanceTransform(const MotionInstanceTransformSRT &m) noexcept;

    [[nodiscard]] auto &as_matrix() noexcept { return *reinterpret_cast<MotionInstanceTransformMatrix *>(data); }
    [[nodiscard]] auto &as_matrix() const noexcept { return *reinterpret_cast<const MotionInstanceTransformMatrix *>(data); }
    [[nodiscard]] auto &as_srt() noexcept { return *reinterpret_cast<MotionInstanceTransformSRT *>(data); }
    [[nodiscard]] auto &as_srt() const noexcept { return *reinterpret_cast<const MotionInstanceTransformSRT *>(data); }
};

inline MotionInstanceTransform::MotionInstanceTransform(const MotionInstanceTransformMatrix &m) noexcept {
    as_matrix() = m;
}

inline MotionInstanceTransform::MotionInstanceTransform(const MotionInstanceTransformSRT &m) noexcept {
    as_srt() = m;
}

static_assert(sizeof(MotionInstanceTransformMatrix) == 64u && alignof(MotionInstanceTransformMatrix) == 16u);
static_assert(sizeof(MotionInstanceTransformSRT) == 64u && alignof(MotionInstanceTransformSRT) == 16u);
static_assert(sizeof(MotionInstanceTransform) == 64u && alignof(MotionInstanceTransform) == 16u);

}
