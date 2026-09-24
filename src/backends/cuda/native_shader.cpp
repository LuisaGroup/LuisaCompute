#include <cstring>

#include <luisa/core/logging.h>

#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_stream.h"
#include "cuda_command_encoder.h"
#include "native_shader.h"

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] inline luisa::string cuda_error_description(CUresult result) noexcept {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    return luisa::format("{} ({})",
                         message == nullptr ? "unknown CUDA error" : message,
                         name == nullptr ? "unknown" : name);
}

}// namespace

CUDANativeShader::CUDANativeShader(CUmodule module, CUfunction function,
                                   luisa::string entry, uint3 block_size,
                                   luisa::vector<Parameter> parameters) noexcept
    : _module{module}, _function{function}, _entry{std::move(entry)},
      _block_size{block_size}, _parameters{std::move(parameters)} {
    for (auto i = 0u; i < _parameters.size(); i++) {
        if (_parameters[i].is_buffer) {
            _buffer_parameters.emplace_back(i);
        } else {
            _scalar_parameters.emplace_back(i);
        }
    }
}

CUDANativeShader::~CUDANativeShader() noexcept {
    if (_module != nullptr) {
        LUISA_CHECK_CUDA(cuModuleUnload(_module));
        _module = nullptr;
        _function = nullptr;
    }
}

CUDANativeShader *CUDANativeShader::create(
    luisa::span<const std::byte> ptx, luisa::string_view entry,
    uint3 block_size, luisa::vector<Parameter> parameters,
    luisa::string &error) noexcept {
    if (ptx.empty()) {
        error = "Empty PTX image.";
        return nullptr;
    }
    if (entry.empty()) {
        error = "The native CUDA shader has no entry point name.";
        return nullptr;
    }
    CUmodule module = nullptr;
    if (auto ret = cuModuleLoadData(&module, ptx.data()); ret != CUDA_SUCCESS) {
        error = luisa::format("cuModuleLoadData failed for kernel '{}': {}.",
                              entry, cuda_error_description(ret));
        return nullptr;
    }
    auto name = luisa::string{entry};
    CUfunction function = nullptr;
    if (auto ret = cuModuleGetFunction(&function, module, name.c_str());
        ret != CUDA_SUCCESS) {
        LUISA_CHECK_CUDA(cuModuleUnload(module));
        error = luisa::format("The PTX module declares no kernel named '{}': {}.",
                              entry, cuda_error_description(ret));
        return nullptr;
    }
    return new_with_allocator<CUDANativeShader>(
        module, function, std::move(name), block_size, std::move(parameters));
}

uint32_t CUDANativeShader::uniform_bytes() const noexcept {
    auto bytes = 0u;
    for (auto index : _scalar_parameters) {
        bytes += _parameters[index].size;
    }
    return bytes;
}

void CUDANativeShader::launch(
    CUDACommandEncoder &encoder,
    NativeShaderDispatchCommand const *command) const noexcept {

    // Scratch for the kernel parameters. The driver reads one value per
    // parameter through the pointer array, so the values are laid out
    // contiguously (respecting each parameter's compiled alignment) and the
    // pointers point into that storage.
    static thread_local luisa::vector<std::byte> parameter_values;
    static thread_local luisa::vector<void *> parameter_pointers;
    static thread_local luisa::vector<size_t> parameter_offsets;

    parameter_offsets.resize(_parameters.size());
    auto cursor = static_cast<size_t>(0u);
    for (auto i = 0u; i < _parameters.size(); i++) {
        auto alignment = static_cast<size_t>(_parameters[i].alignment);
        parameter_offsets[i] = luisa::align(cursor, std::max(alignment, size_t{1u}));
        cursor = parameter_offsets[i] + _parameters[i].size;
    }
    luisa::vector_resize(parameter_values, cursor);
    parameter_pointers.resize(_parameters.size());
    for (auto i = 0u; i < _parameters.size(); i++) {
        parameter_pointers[i] = parameter_values.data() + parameter_offsets[i];
    }

    // Walk the command's arguments: they arrive as the launcher resolved them,
    // i.e. every buffer in canonical (parameter) order, followed by the uniform
    // values in declaration order.
    auto buffer_index = 0u;
    auto scalar_index = 0u;
    auto fill_buffer = [&](Argument::Buffer const &buffer) noexcept {
        if (buffer_index >= _buffer_parameters.size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Native CUDA shader '{}' was given more than {} buffer "
                "argument(s).",
                _entry, _buffer_parameters.size());
        }
        auto parameter = _buffer_parameters[buffer_index++];
        auto *b = reinterpret_cast<CUDABuffer const *>(buffer.handle);
        auto address = b->binding(buffer.offset, buffer.size).handle;
        static_assert(sizeof(address) == sizeof(uint64_t));
        std::memcpy(parameter_pointers[parameter], &address, sizeof(address));
    };
    auto fill_scalar = [&](Argument::Uniform const &uniform) noexcept {
        if (scalar_index >= _scalar_parameters.size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Native CUDA shader '{}' was given more than {} uniform "
                "value(s).",
                _entry, _scalar_parameters.size());
        }
        auto parameter = _scalar_parameters[scalar_index++];
        auto data = command->uniform(uniform);
        if (data.size() != _parameters[parameter].size) {
            LUISA_ERROR_WITH_LOCATION(
                "Native CUDA shader '{}' expects {} byte(s) for uniform "
                "parameter {} but the launcher supplied {} byte(s) (kernel "
                "parameters are matched in declaration order; see the "
                "native-shader contract).",
                _entry, _parameters[parameter].size, scalar_index - 1u,
                data.size());
        }
        std::memcpy(parameter_pointers[parameter], data.data(), data.size());
    };
    for (auto &&argument : command->arguments()) {
        switch (argument.tag) {
            case Argument::Tag::BUFFER: fill_buffer(argument.buffer); break;
            case Argument::Tag::UNIFORM: fill_scalar(argument.uniform); break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "The CUDA native-shader route supports buffer arguments and "
                    "uniform values only, but shader '{}' was dispatched with a "
                    "{} argument.",
                    _entry, argument.tag == Argument::Tag::TEXTURE ? "texture" :
                              argument.tag == Argument::Tag::BINDLESS_ARRAY ?
                                  "bindless array" :
                                  "acceleration-structure");
        }
    }
    if (buffer_index != _buffer_parameters.size() ||
        scalar_index != _scalar_parameters.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Native CUDA shader '{}' declares {} buffer and {} uniform "
            "parameter(s) but the dispatch supplied {} buffer and {} uniform "
            "argument(s).",
            _entry, _buffer_parameters.size(), _scalar_parameters.size(),
            buffer_index, scalar_index);
    }

    // Launch configuration: `dispatch_size()` is the exact thread count.
    auto block_size = command->block_size();
    if (any(block_size != _block_size)) {
        LUISA_ERROR_WITH_LOCATION(
            "Native CUDA shader '{}' is dispatched with block size ({}, {}, {}) "
            "but was compiled for ({}, {}, {}).",
            _entry, block_size.x, block_size.y, block_size.z,
            _block_size.x, _block_size.y, _block_size.z);
    }
    auto dispatch_size = command->dispatch_size();
    auto blocks = (dispatch_size + block_size - 1u) / block_size;
    auto cuda_stream = encoder.stream()->handle();
    // `cuLaunchKernel`'s `kernelParams` is the *array of pointers to the
    // parameter values* itself (one entry per parameter of the kernel, in
    // declaration order), so the array is passed directly here. Passing the
    // address of a local that happens to hold the array base instead would make
    // the driver read the second and following parameters out of unrelated
    // memory, which is the classic way to corrupt a multi-parameter launch.
    LUISA_CHECK_CUDA(cuLaunchKernel(
        _function,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, cuda_stream, parameter_pointers.data(), nullptr));
}

}// namespace luisa::compute::cuda
