#include <fstream>

#include <luisa/runtime/rhi/command.h>
#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_buffer.h"
#include "cuda_accel.h"
#include "cuda_texture.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "cuda_shader_native.h"

namespace luisa::compute::cuda {

CUDAShaderNative::CUDAShaderNative(CUDADevice *device,
                                   const char *ptx, size_t ptx_size,
                                   const char *entry, uint3 block_size,
                                   luisa::vector<Usage> argument_usages,
                                   luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : CUDAShader{std::move(argument_usages)},
      _entry{entry},
      _block_size{block_size.x, block_size.y, block_size.z},
      _bound_arguments{std::move(bound_arguments)} {

    auto load_ptx = [&](const char *ptx, size_t ptx_size) noexcept {
        CUlinkState link_state{};
        size_t cubin_size;
        void *cubin = nullptr;
        LUISA_CHECK_CUDA(cuLinkCreate(0u, nullptr, nullptr, &link_state));
        auto devrt = device->cudadevrt_library();
        if (!devrt.empty()) {
            LUISA_CHECK_CUDA(cuLinkAddData(link_state, CU_JIT_INPUT_LIBRARY,
                                           const_cast<char *>(device->cudadevrt_library().data()),
                                           device->cudadevrt_library().size(),
                                           "cudadevrt", 0u, nullptr, nullptr));
        }
        auto ret = cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                                 const_cast<char *>(ptx), ptx_size,
                                 "my_kernel.ptx", 0u, nullptr, nullptr);
        if (ret != CUDA_SUCCESS) {
            LUISA_CHECK_CUDA(cuLinkDestroy(link_state));
            return ret;
        }
        LUISA_CHECK_CUDA(cuLinkComplete(link_state, &cubin, &cubin_size));
        LUISA_CHECK_CUDA(cuModuleLoadData(&_module, cubin));
        LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
        if (!devrt.empty()) {
            if (cuModuleGetFunction(&_indirect_function, _module, "kernel_launcher") != CUDA_SUCCESS) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to find kernel_launcher() in the PTX module. "
                    "Indirect dispatch will not be available for this kernel.");
                _indirect_function = nullptr;
            }
        }
        LUISA_CHECK_CUDA(cuLinkDestroy(link_state));
        return CUDA_SUCCESS;
    };

    auto ret = load_ptx(ptx, ptx_size);
    if (ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {

        LUISA_WARNING_WITH_LOCATION(
            "The PTX version is not supported by the installed CUDA driver. "
            "Trying to patch the PTX to make it compatible with the driver. "
            "This might cause unexpected behavior. "
            "Please consider upgrading your CUDA driver.");

        // For users with newer CUDA and older driver,
        // the generated PTX might be reported invalid.
        // We have to patch the ".version 7.x" instruction.
        using namespace std::string_view_literals;
        luisa::string s{ptx, ptx_size};
        auto pattern = ".version 7."sv;
        if (auto p = s.find(pattern); p != luisa::string_view::npos) {
            auto begin = p + pattern.size();
            auto end = begin;
            for (; isdigit(s[end]); end++) {}
            s.replace(begin, end - begin, "0");
        }
        ret = load_ptx(s.c_str(), s.size());
    }
    LUISA_CHECK_CUDA(ret);
}

CUDAShaderNative::~CUDAShaderNative() noexcept {
    LUISA_CHECK_CUDA(cuModuleUnload(_module));
}

void CUDAShaderNative::_launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    static thread_local std::array<std::byte, 65536u> argument_buffer;// should be enough

    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        LUISA_ASSERT(offset + bytes <= argument_buffer.size(),
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return argument_buffer.data() + offset;
    };

    auto encode_argument = [&allocate_argument, command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        switch (arg.tag) {
            case Tag::BUFFER: {
                if (reinterpret_cast<const CUDABufferBase *>(arg.buffer.handle)->is_indirect()) {
                    auto buffer = reinterpret_cast<const CUDAIndirectDispatchBuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                } else {
                    auto buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                }
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<const CUDATexture *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                auto ptr = allocate_argument(uniform.size_bytes());
                std::memcpy(ptr, uniform.data(), uniform.size_bytes());
                break;
            }
            case Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<const CUDABindlessArray *>(arg.bindless_array.handle);
                auto binding = array->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::ACCEL: {
                auto accel = reinterpret_cast<const CUDAAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
        }
    };
    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }
    // launch
    auto cuda_stream = encoder.stream()->handle();
    if (command->is_indirect()) {
        LUISA_ASSERT(_indirect_function != nullptr,
                     "Indirect dispatch is not supported by this shader.");
        auto indirect = command->indirect_dispatch();
        auto indirect_buffer = reinterpret_cast<const CUDAIndirectDispatchBuffer *>(indirect.handle);
        auto indirect_binding = indirect_buffer->binding(indirect.offset, indirect.max_dispatch_size);
        void *arguments[] = {argument_buffer.data(), &indirect_binding};
        static constexpr auto block_size = 64u;
        auto block_count = (indirect_binding.capacity - indirect_binding.offset + block_size - 1u) / block_size;
        LUISA_CHECK_CUDA(cuLaunchKernel(
            _indirect_function,
            static_cast<uint>(block_count), 1u, 1u,
            block_size, 1u, 1u,
            0u, cuda_stream, arguments, nullptr));
    } else {
        // the last argument is the launch size
        auto launch_size_and_kernel_id = make_uint4(command->dispatch_size(), 0u);
        auto ptr = allocate_argument(sizeof(launch_size_and_kernel_id));
        std::memcpy(ptr, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));
        // launch configuration
        auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
        auto blocks = (command->dispatch_size() + block_size - 1u) / block_size;
        auto arguments = static_cast<void *>(argument_buffer.data());
        LUISA_CHECK_CUDA(cuLaunchKernel(
            _function,
            blocks.x, blocks.y, blocks.z,
            block_size.x, block_size.y, block_size.z,
            0u, cuda_stream,
            &arguments, nullptr));
    }
}

}// namespace luisa::compute::cuda

