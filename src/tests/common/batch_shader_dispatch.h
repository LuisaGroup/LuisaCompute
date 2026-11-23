// Dispatch a shader that was created outside a `Shader` object (plan P5.1).
//
// `Device::compile` wraps a created handle in a `Shader<N, Args...>`, whose
// constructor from a `ShaderCreationInfo` is private. A caller of
// `VulkanShaderCompileExt` receives raw `ShaderCreationInfo` handles, so the
// fixtures that verify them build exactly the dispatch a `Shader` would: the public
// `detail::ShaderInvoke<1>` front end, given the handle, the argument count and the
// uniform size the encoder computes. Nothing here is backend-specific - it is the
// same front end the DSL uses.
#pragma once
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::test {

// Number of encoded arguments of a kernel (one per kernel argument).
[[nodiscard]] inline size_t batch_shader_arg_count(compute::Function kernel) noexcept {
    return kernel.arguments().size();
}

// Byte size of the uniform block a dispatch of `kernel` writes; identical to the
// value a `Shader` object would store.
[[nodiscard]] inline size_t batch_shader_uniform_size(compute::Function kernel) noexcept {
    return compute::ShaderDispatchCmdEncoder::compute_uniform_size(
        kernel.unbound_arguments());
}

// Dispatch a one-buffer shader handle that has no `Shader` object and read `buffer`
// back into `host`. `host` must already be sized to the buffer's element count.
inline void dispatch_one_buffer_shader(
    compute::Stream &stream, uint64_t shader_handle, size_t arg_count,
    size_t uniform_size, compute::Buffer<float> &buffer,
    uint32_t dispatch_size, luisa::vector<float> &host) noexcept {
    compute::detail::ShaderInvoke<1> invoke{shader_handle, arg_count, uniform_size};
    invoke << buffer;
    stream << std::move(invoke).dispatch(dispatch_size)
           << buffer.copy_to(luisa::span{host})
           << compute::synchronize();
}

}// namespace luisa::test