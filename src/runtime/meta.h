//
// Created by Mike Smith on 2021/10/28.
//

#pragma once

#include <algorithm>
#include <string_view>

namespace luisa::compute::meta {

#define LUISA_COMPUTE_BUILTIN_META_INFO(name) \
    static constexpr auto name = std::string_view{"__builtin__" #name};
LUISA_COMPUTE_BUILTIN_META_INFO(is_cpu_device)
LUISA_COMPUTE_BUILTIN_META_INFO(is_gpu_device)
LUISA_COMPUTE_BUILTIN_META_INFO(supports_hardware_texture)
LUISA_COMPUTE_BUILTIN_META_INFO(supports_custom_block_size)
LUISA_COMPUTE_BUILTIN_META_INFO(supports_block_shared_memory)
LUISA_COMPUTE_BUILTIN_META_INFO(supports_ray_tracing)
#undef LUISA_COMPUTE_BUILTIN_META_INFO

}// namespace luisa::compute::meta
