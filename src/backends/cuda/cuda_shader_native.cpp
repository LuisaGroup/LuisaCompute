//
// Created by Mike on 3/18/2023.
//

#include <runtime/rhi/command.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_shader_native.h>

namespace luisa::compute::cuda {

CUDAShaderNative::CUDAShaderNative(const char *ptx, size_t ptx_size,
                                   const char *entry, uint3 block_size,
                                   luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : _entry{entry},
      _block_size{block_size.x, block_size.y, block_size.z},
      _bound_arguments{std::move(bound_arguments)} {

    auto ret = cuModuleLoadData(&_module, ptx);
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
        ret = cuModuleLoadData(&_module, s.c_str());
    }
    LUISA_CHECK_CUDA(ret);
    LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
}

CUDAShaderNative::~CUDAShaderNative() noexcept {
    LUISA_CHECK_CUDA(cuModuleUnload(_module));
}

void CUDAShaderNative::launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    // TODO: support indirect dispatch
    LUISA_ASSERT(!command->is_indirect(), "Indirect dispatch is not supported on CUDA backend.");

    static thread_local std::array<std::byte, 65536u> argument_buffer;// should be enough
    static thread_local std::array<void *, 256u> arguments;           // should be enough, too

    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto argument_count = 0u;
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        LUISA_ASSERT(offset + bytes <= argument_buffer.size() &&
                         argument_count < arguments.size(),
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return arguments[argument_count++] = argument_buffer.data() + offset;
    };

    auto encode_argument = [&allocate_argument, command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        switch (arg.tag) {
            case Tag::BUFFER: {
                auto buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<const CUDAMipmapArray *>(arg.texture.handle);
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
    // the last argument is the launch size
    auto launch_size = command->dispatch_size();
    auto ptr = allocate_argument(sizeof(launch_size));
    std::memcpy(ptr, &launch_size, sizeof(launch_size));
    // launch configuration
    auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    auto blocks = (launch_size + block_size - 1u) / block_size;
    // launch
    auto cuda_stream = encoder.stream()->handle();
    LUISA_CHECK_CUDA(cuLaunchKernel(
        _function,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, cuda_stream,
        arguments.data(), nullptr));
}

}// namespace luisa::compute::cuda
