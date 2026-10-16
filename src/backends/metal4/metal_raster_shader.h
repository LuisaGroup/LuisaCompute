#pragma once

#include <array>
#include <mutex>

#include <luisa/ast/function.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/rhi/command.h>

#include "metal_api.h"

namespace luisa::compute::metal {

class MetalCommandEncoder;
class MetalDevice;
class MetalDepthBuffer;
class MetalTexture;

}// namespace luisa::compute::metal

namespace luisa::compute {
class DrawRasterSceneCommand;
}// namespace luisa::compute

namespace luisa::compute::metal {

class MetalRasterShader {

public:
    using Argument = ShaderDispatchCommandBase::Argument;

    struct RootArgument {
        Argument binding{};
        Usage usage{Usage::NONE};
        MTL::RenderStages stages{MTL::RenderStageVertex};
        bool is_bound{false};
    };

    struct Pipeline {
        NS::SharedPtr<MTL::RenderPipelineState> render;
        NS::SharedPtr<MTL::DepthStencilState> depth;
    };

private:
    struct PipelineKey {
        std::array<MTL::PixelFormat, 8u> color_formats{};
        std::array<NS::UInteger, 4u> vertex_strides{};
        MTL::PixelFormat depth_format{MTL::PixelFormatInvalid};
        RasterState state{};
        uint8_t color_count{};
        uint8_t vertex_stream_count{};

        [[nodiscard]] bool operator==(const PipelineKey &rhs) const noexcept;
        [[nodiscard]] uint64_t hash() const noexcept;
    };

    struct PipelineCacheEntry {
        PipelineKey key;
        Pipeline pipeline;
    };

private:
    const MetalDevice *_device;
    NS::SharedPtr<MTL::Library> _library;
    NS::SharedPtr<MTL4::LibraryFunctionDescriptor> _vertex;
    NS::SharedPtr<MTL4::LibraryFunctionDescriptor> _fragment;
    MeshFormat _mesh_format;
    luisa::vector<RootArgument> _root_arguments;
    size_t _root_argument_size;
    uint32_t _fragment_output_count;
    mutable std::mutex _pipeline_mutex;
    mutable luisa::unordered_map<uint64_t, luisa::vector<PipelineCacheEntry>> _pipelines;
    mutable std::mutex _name_mutex;
    NS::String *_name{nullptr};

private:
    [[nodiscard]] Pipeline _create_pipeline(const PipelineKey &key) const noexcept;

public:
    MetalRasterShader(
        const MetalDevice *device,
        luisa::span<const std::byte> metallib,
        MeshFormat mesh_format,
        luisa::vector<RootArgument> root_arguments,
        size_t root_argument_size,
        uint32_t fragment_output_count,
        luisa::string_view name) noexcept;
    ~MetalRasterShader() noexcept;
    [[nodiscard]] bool valid() const noexcept {
        return _library && _vertex && _fragment;
    }
    [[nodiscard]] const MeshFormat &mesh_format() const noexcept { return _mesh_format; }
    [[nodiscard]] uint32_t fragment_output_count() const noexcept { return _fragment_output_count; }
    [[nodiscard]] bool matches_mesh_format(const MeshFormat &mesh_format) const noexcept;
    [[nodiscard]] Pipeline pipeline(
        luisa::span<MetalTexture *const> color_targets,
        const MetalDepthBuffer *depth_target,
        const RasterState &state,
        luisa::span<const size_t> vertex_strides) const noexcept;
    [[nodiscard]] MTL::GPUAddress encode_arguments(
        MetalCommandEncoder &encoder,
        const DrawRasterSceneCommand *command) const noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
