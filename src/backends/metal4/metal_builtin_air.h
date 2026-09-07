#pragma once

#include <luisa/core/stl/vector.h>

#include "metal_air_pipeline.h"

namespace luisa::compute::metal {

[[nodiscard]] luisa::vector<std::byte> metal_codegen_builtin_air(
    const MetalAIRTarget &target) noexcept;

}// namespace luisa::compute::metal
