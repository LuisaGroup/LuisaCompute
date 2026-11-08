#pragma once

#include <atomic>

#include <cuda.h>
#include <luisa/core/stl/string.h>
#include "cuda_shader.h"

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAShaderNative final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    CUfunction _indirect_function{};
    luisa::string _entry;
    uint _block_size[3];
    luisa::vector<ShaderDispatchCommand::Argument> _bound_arguments;
    // The image this shader's module was loaded from: the (possibly
    // version-patched) PTX text, or the cudadevrt-linked cubin when the PTX
    // contained kernel_launcher. Retained for cross-backend import (e.g. the
    // Vulkan VK_NV_cuda_kernel_launch interop).
    luisa::vector<std::byte> _module_image;
    // Recorded from the shader metadata at creation time; the launch path has no
    // access to the AST (a cache hit never runs codegen).
    bool _uses_grid_dependency_synchronize{false};
    // Per-shader latch for `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`:
    // once the driver has refused the attribute this shader keeps using the
    // classic entry point. Mutable because `_launch` is const.
    mutable std::atomic_bool _pdl_launch_enabled{true};

private:
    void _launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;

public:
    CUDAShaderNative(CUDADevice *device, luisa::vector<std::byte> ptx,
                     const char *entry, const CUDAShaderMetadata &metadata,
                     luisa::vector<ShaderDispatchCommand::Argument> bound_arguments = {}) noexcept;
    ~CUDAShaderNative() noexcept override;
          [[nodiscard]] bool is_graph_compatible() const noexcept override { return true; }
      [[nodiscard]] bool is_native() const noexcept override { return true; }
    [[nodiscard]] bool uses_grid_dependency_synchronize() const noexcept override { return _uses_grid_dependency_synchronize; }
    [[nodiscard]] void *handle() const noexcept override { return _function; }
    [[nodiscard]] luisa::span<const std::byte> module_image() const noexcept override { return _module_image; }
    [[nodiscard]] luisa::string_view entry() const noexcept override { return _entry; }
    [[nodiscard]] uint3 block_size() const noexcept override {
        return uint3{_block_size[0], _block_size[1], _block_size[2]};
    }
    [[nodiscard]] size_t bound_argument_count() const noexcept override { return _bound_arguments.size(); }
};

}// namespace luisa::compute::cuda
