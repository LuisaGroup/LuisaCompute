#pragma once

#include <cstddef>
#include <cstdint>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::cuda {

// Encodes already optimized LLVM 7 bitcode in the experimental level-2
// container. The result is binary; callers must preserve its explicit length.
[[nodiscard]] luisa::string luisa_compute_cuda_llvm_encode_optix_ir(
    luisa::span<const std::byte> bitcode, uint32_t cuda_arch) noexcept;

}// namespace luisa::compute::cuda
