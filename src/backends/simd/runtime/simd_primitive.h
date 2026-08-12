#pragma once

#include "simd_embree.h"

namespace luisa::compute::simd {

class SIMDPrimitive {

public:
    enum class Kind : uint8_t {
        mesh,
        curve,
        procedural,
        motion_instance,
    };

private:
    Kind _kind;

protected:
    explicit SIMDPrimitive(Kind kind) noexcept : _kind{kind} {}

public:
    virtual ~SIMDPrimitive() noexcept = default;
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] virtual RTCScene handle() const noexcept = 0;
};

}// namespace luisa::compute::simd
