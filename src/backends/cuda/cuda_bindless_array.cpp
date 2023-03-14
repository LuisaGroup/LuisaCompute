//
// Created by Mike on 7/30/2021.
//

#include <runtime/bindless_array.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_bindless_array.h>

namespace luisa::compute::cuda {

CUDABindlessArray::CUDABindlessArray(size_t capacity) noexcept
    : _texture_tracker{capacity} {
    LUISA_CHECK_CUDA(cuMemAlloc(&_handle, capacity * sizeof(Slot)));
    _tex2d_slots.resize(capacity, 0ull);
    _tex3d_slots.resize(capacity, 0ull);
}

CUDABindlessArray::~CUDABindlessArray() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_handle));
    _texture_tracker.traverse([](auto tex) noexcept {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
    });
}

[[nodiscard]] static auto cuda_texture_descriptor(Sampler sampler) noexcept {
    CUDA_TEXTURE_DESC texture_desc{};
    switch (sampler.address()) {
        case Sampler::Address::EDGE:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            break;
        case Sampler::Address::REPEAT:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
            break;
        case Sampler::Address::MIRROR:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_MIRROR;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_MIRROR;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_MIRROR;
            break;
        case Sampler::Address::ZERO:
            texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
            texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
            texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
            break;
    }
    switch (sampler.filter()) {
        case Sampler::Filter::POINT:
            texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            break;
        case Sampler::Filter::LINEAR_POINT:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
            break;
        case Sampler::Filter::LINEAR_LINEAR:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxMipmapLevelClamp = 999.0f;
            break;
        case Sampler::Filter::ANISOTROPIC:
            texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
            texture_desc.maxAnisotropy = 16;
            texture_desc.maxMipmapLevelClamp = 999.0f;
            break;
    }
    texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    return texture_desc;
}

[[nodiscard]] inline auto create_cuda_texture_object(uint64_t handle, Sampler sampler) noexcept {
    CUDA_RESOURCE_DESC res_desc{};
    if (auto array = reinterpret_cast<const CUDAMipmapArray *>(handle);
        array->levels() == 1u) {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array->handle());
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array->handle());
    }
    auto tex_desc = cuda_texture_descriptor(sampler);
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    return texture;
}

void CUDABindlessArray::update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept {
    using Mod = BindlessArrayUpdateCommand::Modification;
    auto mods = cmd->steal_modifications();
    for (auto &m : mods) {
        // process buffer
        if (m.buffer.op == Mod::Operation::EMPLACE) {
            auto buffer = reinterpret_cast<const CUDABuffer *>(m.buffer.handle);
            LUISA_ASSERT(m.buffer.offset_bytes < buffer->size(),
                         "Offset {} exceeds buffer size {}.",
                         m.buffer.offset_bytes, buffer->size());
            auto address = buffer->handle() + m.buffer.offset_bytes;
            auto size = buffer->size() - m.buffer.offset_bytes;
            m.buffer.handle = address;
            m.buffer.offset_bytes = size;// FIXME: reusing this field is a bit hacky
        }
        // process tex2d
        if (m.tex2d.op == Mod::Operation::EMPLACE) {
            if (auto t = _tex2d_slots[m.slot]) { _texture_tracker.release(t); }
            auto t = create_cuda_texture_object(m.tex2d.handle, m.tex2d.sampler);
            m.tex2d.handle = t;
            _tex2d_slots[m.slot] = t;
            _texture_tracker.retain(t);
        } else if (m.tex2d.op == Mod::Operation::REMOVE) {
            if (auto t = _tex2d_slots[m.slot]) { _texture_tracker.release(t); }
            _tex2d_slots[m.slot] = 0ull;
        }
        // process tex3d
        if (m.tex3d.op == Mod::Operation::EMPLACE) {
            if (auto t = _tex3d_slots[m.slot]) { _texture_tracker.release(t); }
            auto t = create_cuda_texture_object(m.tex3d.handle, m.tex3d.sampler);
            m.tex3d.handle = t;
            _tex3d_slots[m.slot] = t;
            _texture_tracker.retain(t);
        } else if (m.tex3d.op == Mod::Operation::REMOVE) {
            if (auto t = _tex3d_slots[m.slot]) { _texture_tracker.release(t); }
            _tex3d_slots[m.slot] = 0ull;
        }
    }
    _texture_tracker.commit([](auto tex) noexcept {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
    });
    auto cuda_stream = encoder.stream()->handle();
    auto update_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(
        &update_buffer, mods.size() * sizeof(Mod), cuda_stream));
    auto size_bytes = mods.size() * sizeof(Mod);
    encoder.with_upload_buffer(size_bytes, [&](auto host_update_buffer) noexcept {
        std::memcpy(host_update_buffer->address(), mods.data(), size_bytes);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            update_buffer, host_update_buffer->address(),
            size_bytes, cuda_stream));
    });
    auto update_kernel = encoder.stream()->device()->bindless_array_update_function();
    auto n = static_cast<uint32_t>(mods.size());
    std::array<void *, 3u> args{&_handle, &update_buffer, &n};
    LUISA_CHECK_CUDA(cuLaunchKernel(
        update_kernel,
        (n + 255u) / 256u, 1u, 1u, 256u, 1u, 1u,
        0u, cuda_stream, args.data(), nullptr));
    LUISA_CHECK_CUDA(cuMemFreeAsync(update_buffer, cuda_stream));
}

}// namespace luisa::compute::cuda
