#include "shader.h"
#include <luisa/core/fiber.h>
#include "memory_manager.h"
namespace lc::toy_c {
LCShader::LCShader(DynamicModule &dyn_module, luisa::span<const Type *const> arg_types, luisa::string_view kernel_name) {
    kernel = dyn_module.function<void(uint3 thd_id, uint3 blk_id, uint3 dsp_id, uint3 dsp_size, uint3 ker_id, void *args)>(kernel_name);
    vstd::string md5_name{kernel_name};
    md5_name += "_args_md5_c4434d750cf64f0eae3f73cca8650b16";

    vstd::string block_name{kernel_name};
    block_name += "_block_size_c4434d750cf64f0eae3f73cca8650b16";

    vstd::string usage_name{kernel_name};
    usage_name += "_arg_usage_c4434d750cf64f0eae3f73cca8650b16";
    auto md5_value = dyn_module.invoke<ulong2()>(md5_name);
    luisa::vector<char> arg_decs;
    arg_decs.reserve(1024);
    for (auto &i : arg_types) {
        auto &&desc = i->description();
        vstd::push_back_all(arg_decs, desc.data(), desc.size());
        arg_decs.emplace_back(' ');
    }
    vstd::MD5 md5{luisa::span{reinterpret_cast<uint8_t const *>(arg_decs.data()), arg_decs.size_bytes()}};
    if (std::memcmp(&md5, &md5_value, sizeof(vstd::MD5)) != 0) {
        LUISA_ERROR("LCShader type validation mismatch.");
    }

    block_size = dyn_module.invoke<uint3()>(block_name);
    get_usage = dyn_module.function<uint32_t(uint32_t)>(usage_name);
}
LCShader::~LCShader() {}
void LCShader::_emplace_arg(luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer) {
    arg_buffer.clear();
    for (auto &i : arguments) {
        switch (i.tag) {
            case Argument::Tag::BUFFER: {
                auto idx = arg_buffer.size();
                arg_buffer.push_back_uninitialized(16);
                auto ptr = reinterpret_cast<ulong *>(arg_buffer.data() + idx);
                ptr[0] = i.buffer.handle + i.buffer.offset;
                ptr[1] = i.buffer.size;
            } break;
            case Argument::Tag::UNIFORM: {
                auto size = i.uniform.size;
                auto aligned_size = (size + 15ull) & (~15ull);
                auto idx = arg_buffer.size();
                arg_buffer.push_back_uninitialized(aligned_size);
                std::memcpy(arg_buffer.data() + idx, uniform_data + i.uniform.offset, size);
            } break;
            case Argument::Tag::TEXTURE:
                LUISA_ERROR("Texture argument not supported.");
                break;
            case Argument::Tag::BINDLESS_ARRAY:
                LUISA_ERROR("BindlessArray argument not supported.");
                break;
            case Argument::Tag::ACCEL:
                LUISA_ERROR("Accel argument not supported.");
                break;
        }
    }
}
void LCShader::dispatch(LCDevice* device, LCStream *stream, MemoryManager &manager, uint3 size, luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer) {
    _emplace_arg(arguments, uniform_data, arg_buffer);
    auto disp_count = (size + block_size - 1u) / block_size;

    luisa::fiber::parallel(
        disp_count.x * disp_count.y * disp_count.z,
        [&](uint idx) {
            uint3 block_idx;
            uint xy = block_size.x * block_size.y;
            block_idx.z = idx / xy;
            idx -= block_idx.z * xy;
            block_idx.y = idx / block_size.x;
            idx -= block_idx.y * block_size.x;
            block_idx.x = idx;

            uint3 start_idx = block_idx * block_size;
            uint3 end_idx = min(size, (block_idx + 1u) * block_size);
            uint3 disp_count = end_idx - start_idx;
            manager.alloc_tlocal_ctx();
            auto ctx = MemoryManager::get_tlocal_ctx();
            ctx->device = device;
            ctx->stream = stream;
            for (uint z = 0; z < disp_count.z; ++z)
                for (uint y = 0; y < disp_count.y; ++y)
                    for (uint x = 0; x < disp_count.x; ++x) {
                        kernel(
                            uint3(x, y, z),
                            block_idx,
                            start_idx + uint3(x, y, z),
                            size,
                            uint3(0),
                            arg_buffer.data());
                    }
            manager.dealloc_tlocal_ctx();
        },
        1);
}
namespace detail {
template<class F>
    requires(std::is_invocable_v<F, uint32_t>)
[[nodiscard]] void async_parallel(luisa::fiber::counter &evt, uint32_t job_count, F &&lambda, uint32_t internal_jobs = 1) noexcept {
    using namespace luisa::fiber;
    auto thread_count = std::clamp<uint32_t>(job_count / internal_jobs, 1u, worker_thread_count());
    evt.add(thread_count);
    luisa::SharedFunction<void()> func{[counter = luisa::fiber::detail::NonMovableAtomic<uint32_t>(0), job_count, internal_jobs, evt, lambda = std::forward<F>(lambda)]() mutable noexcept {
        uint32_t i = 0u;
        while ((i = counter.value.fetch_add(internal_jobs)) < job_count) {
            auto end = std::min<uint32_t>(i + internal_jobs, job_count);
            for (uint32_t v = i; v < end; ++v) {
                lambda(v);
            }
        }
        evt.done();
    }};
    for (uint32_t i = 0; i < thread_count; ++i) {
        marl::schedule(func);
    }
}
}// namespace detail
void LCShader::dispatch(LCDevice *device, LCStream *stream, MemoryManager &manager, luisa::span<uint3 const> sizes, luisa::span<const Argument> arguments, std::byte const *uniform_data, luisa::vector<std::byte> &arg_buffer) {
    if (sizes.empty()) return;
    if (sizes.size() == 1) {
        dispatch(device, stream, manager, sizes[0], arguments, uniform_data, arg_buffer);
    }
    _emplace_arg(arguments, uniform_data, arg_buffer);
    luisa::fiber::counter evt;
    for (auto size : sizes) {
        auto disp_count = (size + block_size - 1u) / block_size;
        detail::async_parallel(
            evt,
            disp_count.x * disp_count.y * disp_count.z,
            [&](uint idx) {
                uint3 block_idx;
                uint xy = block_size.x * block_size.y;
                block_idx.z = idx / xy;
                idx -= block_idx.z * xy;
                block_idx.y = idx / block_size.x;
                idx -= block_idx.y * block_size.x;
                block_idx.x = idx;

                uint3 start_idx = block_idx * block_size;
                uint3 end_idx = min(size, (block_idx + 1u) * block_size);
                uint3 disp_count = end_idx - start_idx;
                manager.alloc_tlocal_ctx();
                auto ctx = MemoryManager::get_tlocal_ctx();
                ctx->device = device;
                ctx->stream = stream;
                for (uint z = 0; z < disp_count.z; ++z)
                    for (uint y = 0; y < disp_count.y; ++y)
                        for (uint x = 0; x < disp_count.x; ++x) {
                            kernel(
                                uint3(x, y, z),
                                block_idx,
                                start_idx + uint3(x, y, z),
                                size,
                                uint3(0),
                                arg_buffer.data());
                        }
                manager.dealloc_tlocal_ctx();
            },
            1);
    }
    evt.wait();
}
}// namespace lc::toy_c