#pragma once

#include <cuda.h>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::cuda {

class CUDACommandEncoder;

// A native (non-DSL) CUDA kernel: a PTX module compiled by NVRTC from
// user-supplied CUDA C++ source, its `__global__` entry function, and the
// launch-time layout of the kernel's parameter list.
//
// The parameter layout is the whole ABI of this shader (see
// native_shader_reflection.h): the kernel's pointer parameters are the
// launcher's buffer arguments - each supplied as the buffer's 64-bit device
// address - and its non-pointer parameters are scalar kernel parameters whose
// values come from the launcher's `add_uniform` values, in declaration order.
class CUDANativeShader {

public:
    struct Parameter {
        uint32_t size{0u};      // bytes, as compiled
        uint32_t alignment{0u}; // bytes, as compiled
        bool is_buffer{false};  // a pointer parameter (a reflected buffer binding)
    };

private:
    CUmodule _module{nullptr};    CUfunction _function{nullptr};
    luisa::string _entry;
    uint3 _block_size{0u, 0u, 0u};
    luisa::vector<Parameter> _parameters;
    // Signature indices of the buffer / scalar parameters, in declaration order.
    luisa::vector<uint32_t> _buffer_parameters;
    luisa::vector<uint32_t> _scalar_parameters;

public:
    // `new_with_allocator` needs a public constructor.
    CUDANativeShader(CUmodule module, CUfunction function, luisa::string entry,
                     uint3 block_size,
                     luisa::vector<Parameter> parameters) noexcept;

public:
    CUDANativeShader(CUDANativeShader const &) = delete;
    CUDANativeShader(CUDANativeShader &&) = delete;
    CUDANativeShader &operator=(CUDANativeShader const &) = delete;
    CUDANativeShader &operator=(CUDANativeShader &&) = delete;
    // Unloads the module; the CUDA context must be bound (the extension does
    // this through `CUDADevice::with_handle`).
    ~CUDANativeShader() noexcept;

    // Loads the PTX module and looks up `entry`. Returns nullptr and fills
    // `error` when the module cannot be loaded or does not declare the kernel.
    [[nodiscard]] static CUDANativeShader *create(
        luisa::span<const std::byte> ptx, luisa::string_view entry,
        uint3 block_size, luisa::vector<Parameter> parameters,
        luisa::string &error) noexcept;

    [[nodiscard]] auto module() const noexcept { return _module; }
    [[nodiscard]] auto function() const noexcept { return _function; }
    [[nodiscard]] luisa::string_view entry() const noexcept { return _entry; }
    [[nodiscard]] uint3 block_size() const noexcept { return _block_size; }
    [[nodiscard]] luisa::span<const Parameter> parameters() const noexcept {
        return _parameters;
    }
    [[nodiscard]] luisa::span<const uint32_t> buffer_parameters() const noexcept {
        return _buffer_parameters;
    }
    [[nodiscard]] luisa::span<const uint32_t> scalar_parameters() const noexcept {
        return _scalar_parameters;
    }
    [[nodiscard]] size_t buffer_parameter_count() const noexcept {
        return _buffer_parameters.size();
    }
    // Total size in bytes of the scalar (uniform) kernel parameters.
    [[nodiscard]] uint32_t uniform_bytes() const noexcept;

    // Encodes the dispatch of `command` on `encoder`'s stream.
    void launch(CUDACommandEncoder &encoder,
                NativeShaderDispatchCommand const *command) const noexcept;
};

}// namespace luisa::compute::cuda
