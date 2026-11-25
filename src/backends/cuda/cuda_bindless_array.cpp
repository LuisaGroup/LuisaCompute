#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/runtime/bindless_array.h>
#include "cuda_buffer.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_command_encoder.h"
#include "cuda_bindless_array.h"

namespace luisa::compute::cuda {

namespace {

// The precompiled CUDA update kernel has a stable 64-byte input ABI. Runtime
// commands carry additional host-side view data, so marshal explicitly at this
// boundary instead of relying on identical C++ struct layouts.
struct alignas(16) DeviceBindlessSlotModification {
    using Operation = BindlessArrayUpdateCommand::Operation;
    struct Buffer {
        uint64_t handle;
        size_t size_bytes;
        Operation op;
        uint32_t _padding;
    };
    struct Texture {
        uint64_t handle;
        uint64_t sampler_table;
        uint32_t payload_index;
        Operation op;
    };
    size_t slot;
    Buffer buffer;
    Texture tex2d;
    Texture tex3d;
};
static_assert(sizeof(DeviceBindlessSlotModification::Buffer) == 24u);
static_assert(sizeof(DeviceBindlessSlotModification::Texture) == 24u);
static_assert(sizeof(DeviceBindlessSlotModification) == 80u);

}// namespace

CUDABindlessArray::CUDABindlessArray(CUDADevice *device, size_t capacity) noexcept
    : _device{device}, _texture_tracker{capacity} {
    const auto total_bytes =
        capacity * (sizeof(Slot) + 2u * sizeof(uint64_t));
    if (total_bytes > 0u) {
        LUISA_CHECK_CUDA(cuMemAlloc(&_handle, total_bytes));
        LUISA_CHECK_CUDA(cuMemsetD8(_handle, 0u, total_bytes));
    }
    _tex2d_slots.resize(capacity);
    _tex3d_slots.resize(capacity);
}

CUDABindlessArray::~CUDABindlessArray() noexcept {
    if (_handle) { LUISA_CHECK_CUDA(cuMemFree(_handle)); }
    _texture_tracker.traverse([](auto tex) noexcept {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
    });
    if (_device != nullptr) {
        for (auto &table : _tex2d_slots) {
            _device->deallocate_sampler_table(table.device_ptr);
        }
        for (auto &table : _tex3d_slots) {
            _device->deallocate_sampler_table(table.device_ptr);
        }
    }
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
    auto array = reinterpret_cast<const CUDATexture *>(handle);
    if (array->levels() == 1u) {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array->handle());
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array->handle());
    }
    auto tex_desc = cuda_texture_descriptor(sampler);
    if (is_srgb(array->format())) { tex_desc.flags |= CU_TRSF_SRGB; }
    CUtexObject texture;
    LUISA_CHECK_CUDA(cuTexObjectCreate(&texture, &res_desc, &tex_desc, nullptr));
    return texture;
}

void CUDABindlessArray::update(CUDACommandEncoder &encoder,
                               BindlessArrayUpdateCommand *cmd) noexcept {

    std::scoped_lock lock{_mutex};

    if (cmd->modifications().empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Empty bindless array update command detected.");
        return;
    }

    if (!_name.empty()) { nvtxRangePushA(luisa::format("{}::update", _name).c_str()); }

    using Mod = BindlessArrayUpdateCommand::Modification;
    auto mods = cmd->steal_modifications();
    luisa::vector<DeviceBindlessSlotModification> device_mods;
    device_mods.reserve(mods.size());
    luisa::vector<uint64_t> sampler_payloads;
    luisa::vector<CUdeviceptr> retired_sampler_tables;
    auto update_texture_slot = [&](const Mod::Texture &src,
                                   SamplerTable &dst_slot,
                                   DeviceBindlessSlotModification::Texture &dst_mod) noexcept {
        dst_mod.op = src.op;
        if (src.op == Mod::Operation::NONE) {
            return;
        }
        if (dst_slot.device_ptr != 0ull) {
            for (auto t : dst_slot.objects) {
                if (t) { _texture_tracker.release(t); }
            }
            retired_sampler_tables.emplace_back(dst_slot.device_ptr);
            dst_slot = {};
        }
        if (src.op == Mod::Operation::EMPLACE) {
            SamplerTable table{};
            table.device_ptr = _device->allocate_sampler_table();
            const auto payload_index =
                static_cast<uint32_t>(sampler_payloads.size() / sampler_variant_count);
            for (auto f = 0u; f < 4u; ++f) {
                for (auto a = 0u; a < 4u; ++a) {
                    const Sampler s{
                        static_cast<Sampler::Filter>(f),
                        static_cast<Sampler::Address>(a)};
                    const auto t = create_cuda_texture_object(src.handle, s);
                    const auto code = (f << 2u) | a;
                    table.objects[code] = t;
                    _texture_tracker.retain(t);
                    sampler_payloads.emplace_back(t);
                }
            }
            dst_mod.handle = table.objects[src.sampler.code()];
            dst_mod.sampler_table = table.device_ptr;
            dst_mod.payload_index = payload_index;
            dst_slot = table;
        } else {
            dst_mod.handle = 0ull;
            dst_mod.sampler_table = 0ull;
            dst_mod.payload_index = 0u;
        }
    };
    for (auto &m : mods) {
        DeviceBindlessSlotModification device_mod{};
        device_mod.slot = m.slot;
        device_mod.buffer.handle = m.buffer.handle;
        device_mod.buffer.size_bytes = m.buffer.size_bytes;
        device_mod.buffer.op = m.buffer.op;
        // process buffer
        if (m.buffer.op == Mod::Operation::EMPLACE) {
            auto buffer = reinterpret_cast<const CUDABuffer *>(m.buffer.handle);
            LUISA_ASSERT(m.buffer.offset_bytes < buffer->size_bytes(),
                         "Offset {} exceeds buffer size {}.",
                         m.buffer.offset_bytes, buffer->size_bytes());
            auto address = buffer->device_address() + m.buffer.offset_bytes;
            auto remaining_size = buffer->size_bytes() - m.buffer.offset_bytes;
            auto size =
                m.buffer.size_bytes ==
                        BindlessArrayUpdateCommand::ModifiedBuffer::whole_buffer_size ?
                    remaining_size :
                    m.buffer.size_bytes;
            LUISA_ASSERT(size > 0u && size <= remaining_size,
                         "Bindless buffer view [{}, {}) exceeds buffer size {}.",
                         m.buffer.offset_bytes,
                         m.buffer.offset_bytes + size,
                         buffer->size_bytes());
            device_mod.buffer.handle = address;
            device_mod.buffer.size_bytes = size;
        }
        update_texture_slot(m.tex2d, _tex2d_slots[m.slot], device_mod.tex2d);
        update_texture_slot(m.tex3d, _tex3d_slots[m.slot], device_mod.tex3d);
        device_mods.emplace_back(device_mod);
    }
    _texture_tracker.commit([](auto tex) noexcept {
        LUISA_CHECK_CUDA(cuTexObjectDestroy(tex));
    });
    for (auto ptr : retired_sampler_tables) {
        _device->deallocate_sampler_table(ptr);
    }
    auto cuda_stream = encoder.stream()->handle();
    const auto mods_bytes = device_mods.size() * sizeof(DeviceBindlessSlotModification);
    const auto payloads_bytes = sampler_payloads.size() * sizeof(uint64_t);
    const auto total_upload_bytes = mods_bytes + payloads_bytes;
    auto upload_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAllocAsync(&upload_buffer, total_upload_bytes, cuda_stream));
    encoder.with_upload_buffer(total_upload_bytes, [&](auto host_upload_buffer) noexcept {
        auto *dst = static_cast<std::byte *>(host_upload_buffer->address());
        std::memcpy(dst, device_mods.data(), mods_bytes);
        if (payloads_bytes != 0u) {
            std::memcpy(dst + mods_bytes, sampler_payloads.data(), payloads_bytes);
        }
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            upload_buffer, host_upload_buffer->address(),
            total_upload_bytes, cuda_stream));
    });
    auto update_kernel = encoder.stream()->device()->bindless_array_update_function();
    auto capacity = static_cast<uint64_t>(_tex2d_slots.size());
    auto payloads_ptr = payloads_bytes != 0u ? (upload_buffer + mods_bytes) : 0ull;
    auto n = static_cast<uint32_t>(mods.size());
    std::array<void *, 5u> args{&_handle, &capacity, &upload_buffer, &payloads_ptr, &n};
    LUISA_CHECK_CUDA(cuLaunchKernel(
        update_kernel,
        (n + 255u) / 256u, 1u, 1u, 256u, 1u, 1u,
        0u, cuda_stream, args.data(), nullptr));
    LUISA_CHECK_CUDA(cuMemFreeAsync(upload_buffer, cuda_stream));

    if (!_name.empty()) { nvtxRangePop(); }
}

void CUDABindlessArray::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = std::move(name);
}

}// namespace luisa::compute::cuda
