#include <array>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/backends/ext/raster_cmd.h>

#include "metal_accel.h"
#include "metal_bindless_array.h"
#include "metal_buffer.h"
#include "metal_command_encoder.h"
#include "metal_device.h"
#include "metal_depth_buffer.h"
#include "metal_raster_archive.h"
#include "metal_raster_shader.h"
#include "metal_texture.h"

namespace luisa::compute::metal {

namespace {

[[nodiscard]] MTL::VertexFormat vertex_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: return MTL::VertexFormatChar;
        case PixelFormat::R8UInt: return MTL::VertexFormatUChar;
        case PixelFormat::R8UNorm: return MTL::VertexFormatUCharNormalized;
        case PixelFormat::RG8SInt: return MTL::VertexFormatChar2;
        case PixelFormat::RG8UInt: return MTL::VertexFormatUChar2;
        case PixelFormat::RG8UNorm: return MTL::VertexFormatUChar2Normalized;
        case PixelFormat::RGBA8SInt: return MTL::VertexFormatChar4;
        case PixelFormat::RGBA8UInt: return MTL::VertexFormatUChar4;
        case PixelFormat::RGBA8UNorm: return MTL::VertexFormatUChar4Normalized;
        case PixelFormat::R16SInt: return MTL::VertexFormatShort;
        case PixelFormat::R16UInt: return MTL::VertexFormatUShort;
        case PixelFormat::R16UNorm: return MTL::VertexFormatUShortNormalized;
        case PixelFormat::RG16SInt: return MTL::VertexFormatShort2;
        case PixelFormat::RG16UInt: return MTL::VertexFormatUShort2;
        case PixelFormat::RG16UNorm: return MTL::VertexFormatUShort2Normalized;
        case PixelFormat::RGBA16SInt: return MTL::VertexFormatShort4;
        case PixelFormat::RGBA16UInt: return MTL::VertexFormatUShort4;
        case PixelFormat::RGBA16UNorm: return MTL::VertexFormatUShort4Normalized;
        case PixelFormat::R32SInt: return MTL::VertexFormatInt;
        case PixelFormat::R32UInt: return MTL::VertexFormatUInt;
        case PixelFormat::RG32SInt: return MTL::VertexFormatInt2;
        case PixelFormat::RG32UInt: return MTL::VertexFormatUInt2;
        case PixelFormat::RGBA32SInt: return MTL::VertexFormatInt4;
        case PixelFormat::RGBA32UInt: return MTL::VertexFormatUInt4;
        case PixelFormat::R16F: return MTL::VertexFormatHalf;
        case PixelFormat::RG16F: return MTL::VertexFormatHalf2;
        case PixelFormat::RGBA16F: return MTL::VertexFormatHalf4;
        case PixelFormat::R32F: return MTL::VertexFormatFloat;
        case PixelFormat::RG32F: return MTL::VertexFormatFloat2;
        case PixelFormat::RGBA32F: return MTL::VertexFormatFloat4;
        case PixelFormat::R10G10B10A2UNorm: return MTL::VertexFormatUInt1010102Normalized;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Pixel format 0x{:02x} is not a supported Metal vertex format.",
        luisa::to_underlying(format));
}

[[nodiscard]] MTL::PrimitiveTopologyClass primitive_topology(TopologyType topology) noexcept {
    switch (topology) {
        case TopologyType::Point: return MTL::PrimitiveTopologyClassPoint;
        case TopologyType::Line: return MTL::PrimitiveTopologyClassLine;
        case TopologyType::Triangle: return MTL::PrimitiveTopologyClassTriangle;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid raster topology.");
}

[[nodiscard]] MTL::CompareFunction compare_function(Comparison comparison) noexcept {
    switch (comparison) {
        case Comparison::Never: return MTL::CompareFunctionNever;
        case Comparison::Less: return MTL::CompareFunctionLess;
        case Comparison::Equal: return MTL::CompareFunctionEqual;
        case Comparison::LessEqual: return MTL::CompareFunctionLessEqual;
        case Comparison::Greater: return MTL::CompareFunctionGreater;
        case Comparison::NotEqual: return MTL::CompareFunctionNotEqual;
        case Comparison::GreaterEqual: return MTL::CompareFunctionGreaterEqual;
        case Comparison::Always: return MTL::CompareFunctionAlways;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid raster comparison function.");
}

[[nodiscard]] MTL::BlendOperation blend_operation(BlendOp operation) noexcept {
    switch (operation) {
        case BlendOp::Add: return MTL::BlendOperationAdd;
        case BlendOp::Subtract: return MTL::BlendOperationSubtract;
        case BlendOp::Min: return MTL::BlendOperationMin;
        case BlendOp::Max: return MTL::BlendOperationMax;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid raster blend operation.");
}

[[nodiscard]] MTL::BlendFactor blend_factor(BlendWeight factor) noexcept {
    switch (factor) {
        case BlendWeight::Zero: return MTL::BlendFactorZero;
        case BlendWeight::One: return MTL::BlendFactorOne;
        case BlendWeight::PrimColor: return MTL::BlendFactorSourceColor;
        case BlendWeight::ImgColor: return MTL::BlendFactorDestinationColor;
        case BlendWeight::PrimAlpha: return MTL::BlendFactorSourceAlpha;
        case BlendWeight::ImgAlpha: return MTL::BlendFactorDestinationAlpha;
        case BlendWeight::OneMinusPrimColor: return MTL::BlendFactorOneMinusSourceColor;
        case BlendWeight::OneMinusImgColor: return MTL::BlendFactorOneMinusDestinationColor;
        case BlendWeight::OneMinusPrimAlpha: return MTL::BlendFactorOneMinusSourceAlpha;
        case BlendWeight::OneMinusImgAlpha: return MTL::BlendFactorOneMinusDestinationAlpha;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid raster blend factor.");
}

[[nodiscard]] uint mtl_resource_usage(Usage usage) noexcept {
    auto result = 0u;
    if ((luisa::to_underlying(usage) & luisa::to_underlying(Usage::READ)) != 0u) {
        result |= MTL::ResourceUsageRead;
    }
    if ((luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) != 0u) {
        result |= MTL::ResourceUsageWrite;
    }
    return result;
}

[[nodiscard]] uint mtl_texture_resource_usage(Usage usage) noexcept {
    auto result = mtl_resource_usage(usage);
    if ((luisa::to_underlying(usage) & luisa::to_underlying(Usage::READ)) != 0u) {
        result |= MTL::ResourceUsageSample;
    }
    return result;
}

[[nodiscard]] bool mesh_format_equal(
    const MeshFormat &lhs, const MeshFormat &rhs) noexcept {
    if (lhs.vertex_stream_count() != rhs.vertex_stream_count()) { return false; }
    for (auto stream = 0u; stream < lhs.vertex_stream_count(); stream++) {
        auto lhs_attributes = lhs.attributes(stream);
        auto rhs_attributes = rhs.attributes(stream);
        if (lhs_attributes.size() != rhs_attributes.size()) { return false; }
        if (std::memcmp(lhs_attributes.data(), rhs_attributes.data(),
                        lhs_attributes.size_bytes()) != 0) {
            return false;
        }
    }
    return true;
}

}// namespace

bool MetalRasterShader::PipelineKey::operator==(const PipelineKey &rhs) const noexcept {
    return color_count == rhs.color_count &&
           vertex_stream_count == rhs.vertex_stream_count &&
           depth_format == rhs.depth_format &&
           color_formats == rhs.color_formats &&
           vertex_strides == rhs.vertex_strides &&
           std::memcmp(&state, &rhs.state, sizeof(RasterState)) == 0;
}

uint64_t MetalRasterShader::PipelineKey::hash() const noexcept {
    auto result = luisa::hash64(color_formats.data(),
                                color_formats.size() * sizeof(MTL::PixelFormat),
                                luisa::hash64_default_seed);
    result = luisa::hash64(vertex_strides.data(),
                           vertex_strides.size() * sizeof(NS::UInteger), result);
    result = luisa::hash64(&depth_format, sizeof(depth_format), result);
    result = luisa::hash64(&state, sizeof(state), result);
    result = luisa::hash64(&color_count, sizeof(color_count), result);
    return luisa::hash64(&vertex_stream_count, sizeof(vertex_stream_count), result);
}

MetalRasterShader::MetalRasterShader(
    const MetalDevice *device,
    luisa::span<const std::byte> metallib,
    MeshFormat mesh_format,
    luisa::vector<RootArgument> root_arguments,
    size_t root_argument_size,
    uint32_t fragment_output_count,
    luisa::string_view name) noexcept
    : _device{device},
      _mesh_format{std::move(mesh_format)},
      _root_arguments{std::move(root_arguments)},
      _root_argument_size{root_argument_size},
      _fragment_output_count{fragment_output_count} {
    luisa::string mesh_format_reason;
    LUISA_ASSERT(validate_metal_raster_mesh_format(
                     _mesh_format, &mesh_format_reason),
                 "Invalid Metal raster mesh format: {}", mesh_format_reason);
    LUISA_ASSERT(_root_argument_size >= 16u &&
                     _root_argument_size % 16u == 0u,
                 "Invalid Metal raster root-argument ABI size {}.",
                 _root_argument_size);
    LUISA_ASSERT(_fragment_output_count >= 1u &&
                     _fragment_output_count <= 8u,
                 "Invalid Metal raster fragment-output count {}.",
                 _fragment_output_count);
    auto library_data = dispatch_data_create(
        metallib.data(), metallib.size_bytes(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *error = nullptr;
    _library = NS::TransferPtr(device->handle()->newLibrary(library_data, &error));
    dispatch_release(library_data);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal raster AIR library '{}': {}.", name,
            error->localizedDescription()->utf8String());
    }
    if (!_library) { return; }
    auto make_function_descriptor = [&](NS::String *entry_name) noexcept {
        auto descriptor = NS::TransferPtr(
            MTL4::LibraryFunctionDescriptor::alloc()->init());
        descriptor->setLibrary(_library.get());
        descriptor->setName(entry_name);
        return descriptor;
    };
    _vertex = make_function_descriptor(MTLSTR("vertex_main"));
    _fragment = make_function_descriptor(MTLSTR("fragment_main"));
    set_name(name);
}

MetalRasterShader::~MetalRasterShader() noexcept {
    if (_name != nullptr) { _name->release(); }
}

bool MetalRasterShader::matches_mesh_format(const MeshFormat &mesh_format) const noexcept {
    return mesh_format_equal(_mesh_format, mesh_format);
}

MetalRasterShader::Pipeline
MetalRasterShader::_create_pipeline(const PipelineKey &key) const noexcept {
    LUISA_ASSERT(!key.state.stencil_state.enable_stencil,
                 "Metal raster AIR does not support stencil state yet.");
    LUISA_ASSERT(!key.state.conservative,
                 "Metal raster AIR does not support conservative rasterization.");
    LUISA_ASSERT(key.state.topology == TopologyType::Triangle ||
                     key.state.fill_mode == FillMode::Solid,
                 "Wireframe fill is only valid for triangle topology on Metal.");

    auto descriptor = NS::TransferPtr(MTL4::RenderPipelineDescriptor::alloc()->init());
    descriptor->setVertexFunctionDescriptor(_vertex.get());
    descriptor->setFragmentFunctionDescriptor(_fragment.get());
    descriptor->setRasterSampleCount(1u);
    descriptor->setInputPrimitiveTopology(primitive_topology(key.state.topology));

    auto color_attachments = descriptor->colorAttachments();
    for (auto i = 0u; i < key.color_count; i++) {
        auto attachment = color_attachments->object(i);
        attachment->setPixelFormat(key.color_formats[i]);
        attachment->setBlendingState(
            key.state.blend_state.enable_blend ?
                MTL4::BlendStateEnabled :
                MTL4::BlendStateDisabled);
        if (key.state.blend_state.enable_blend) {
            auto operation = blend_operation(key.state.blend_state.op);
            auto source = blend_factor(key.state.blend_state.prim_op);
            auto destination = blend_factor(key.state.blend_state.img_op);
            attachment->setRgbBlendOperation(operation);
            attachment->setAlphaBlendOperation(operation);
            attachment->setSourceRGBBlendFactor(source);
            attachment->setSourceAlphaBlendFactor(source);
            attachment->setDestinationRGBBlendFactor(destination);
            attachment->setDestinationAlphaBlendFactor(destination);
        }
    }

    auto vertex_descriptor = NS::TransferPtr(MTL::VertexDescriptor::alloc()->init());
    auto location = 0u;
    for (auto stream = 0u; stream < _mesh_format.vertex_stream_count(); stream++) {
        auto offset = 0u;
        for (auto attribute : _mesh_format.attributes(stream)) {
            auto descriptor_attribute = vertex_descriptor->attributes()->object(location++);
            descriptor_attribute->setFormat(vertex_format(attribute.format));
            descriptor_attribute->setOffset(offset);
            descriptor_attribute->setBufferIndex(stream + 2u);
            offset += pixel_format_size(attribute.format, make_uint3(1u));
        }
        auto layout = vertex_descriptor->layouts()->object(stream + 2u);
        LUISA_ASSERT(key.vertex_strides[stream] >= offset,
                     "Metal vertex stream {} has stride {}, but its attributes require {} bytes.",
                     stream, key.vertex_strides[stream], offset);
        layout->setStride(key.vertex_strides[stream]);
        layout->setStepFunction(MTL::VertexStepFunctionPerVertex);
        layout->setStepRate(1u);
    }
    descriptor->setVertexDescriptor(vertex_descriptor.get());

    {
        std::scoped_lock lock{_name_mutex};
        if (_name != nullptr) { descriptor->setLabel(_name); }
    }
    NS::Error *error = nullptr;
    auto render = NS::TransferPtr(
        _device->metal4_compiler()->newRenderPipelineState(
            descriptor.get(), nullptr, &error));
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create Metal raster pipeline: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(render, "Failed to create Metal raster pipeline.");

    auto depth_descriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    depth_descriptor->setDepthCompareFunction(
        key.state.depth_state.enable_depth ?
            compare_function(key.state.depth_state.comparison) :
            MTL::CompareFunctionAlways);
    depth_descriptor->setDepthWriteEnabled(
        key.state.depth_state.enable_depth && key.state.depth_state.write);
    auto depth = NS::TransferPtr(
        _device->handle()->newDepthStencilState(depth_descriptor.get()));
    LUISA_ASSERT(depth, "Failed to create Metal depth-stencil state.");
    return {.render = std::move(render), .depth = std::move(depth)};
}

MetalRasterShader::Pipeline MetalRasterShader::pipeline(
    luisa::span<MetalTexture *const> color_targets,
    const MetalDepthBuffer *depth_target,
    const RasterState &state,
    luisa::span<const size_t> vertex_strides) const noexcept {
    LUISA_ASSERT(!state.depth_state.enable_depth || depth_target != nullptr,
                 "Depth testing requires a depth attachment.");
    LUISA_ASSERT(vertex_strides.size() == _mesh_format.vertex_stream_count(),
                 "Metal raster pipeline received {} vertex stride(s), expected {}.",
                 vertex_strides.size(), _mesh_format.vertex_stream_count());
    LUISA_ASSERT(vertex_strides.size() <= PipelineKey{}.vertex_strides.size(),
                 "Metal raster pipeline received more than 4 vertex streams.");
    PipelineKey key{};
    key.color_count = static_cast<uint8_t>(color_targets.size());
    key.vertex_stream_count = static_cast<uint8_t>(vertex_strides.size());
    key.depth_format = depth_target == nullptr ?
                           MTL::PixelFormatInvalid :
                           depth_target->pixel_format();
    key.state = state;
    for (auto i = 0u; i < vertex_strides.size(); i++) {
        LUISA_ASSERT(vertex_strides[i] != 0u,
                     "Metal vertex stream {} has a zero stride.", i);
        key.vertex_strides[i] = vertex_strides[i];
    }
    for (auto i = 0u; i < color_targets.size(); i++) {
        LUISA_ASSERT(color_targets[i]->is_raster_target(),
                     "Color attachment {} was not created as a raster target.", i);
        key.color_formats[i] = color_targets[i]->pixel_format();
    }
    auto hash = key.hash();
    std::scoped_lock lock{_pipeline_mutex};
    auto &bucket = _pipelines[hash];
    for (auto &&entry : bucket) {
        if (entry.key == key) { return entry.pipeline; }
    }
    auto result = _create_pipeline(key);
    bucket.emplace_back(PipelineCacheEntry{key, result});
    return result;
}

MTL::GPUAddress MetalRasterShader::encode_arguments(
    MetalCommandEncoder &encoder,
    const DrawRasterSceneCommand *command) const noexcept {
    static constexpr auto argument_buffer_capacity = 65536u;
    static constexpr auto argument_alignment = 16u;
    static thread_local std::array<std::byte, argument_buffer_capacity> argument_buffer;
    LUISA_ASSERT(_root_argument_size <= argument_buffer.size(),
                 "Metal raster AIR root argument buffer exceeds the runtime capacity.");
    std::memset(argument_buffer.data(), 0, _root_argument_size);
    auto offset = static_cast<size_t>(0u);
    auto copy = [&](const void *data, size_t size) noexcept {
        offset = luisa::align(offset, argument_alignment);
        LUISA_ASSERT(offset + size <= argument_buffer.size(),
                     "Metal raster root argument buffer overflow.");
        std::memcpy(argument_buffer.data() + offset, data, size);
        offset += size;
    };

    auto dynamic_arguments = command->arguments();
    auto dynamic_index = static_cast<size_t>(0u);
    for (auto &&root : _root_arguments) {
        auto argument = root.binding;
        if (!root.is_bound) {
            LUISA_ASSERT(dynamic_index < dynamic_arguments.size(),
                         "Metal raster dispatch has too few dynamic arguments.");
            argument = dynamic_arguments[dynamic_index++];
        }
        switch (argument.tag) {
            case Argument::Tag::BUFFER: {
                auto buffer_base = reinterpret_cast<const MetalBufferBase *>(argument.buffer.handle);
                if (buffer_base->is_indirect()) {
                    auto buffer = static_cast<const MetalIndirectDispatchBuffer *>(buffer_base);
                    auto binding = buffer->binding(argument.buffer.offset, argument.buffer.size);
                    copy(&binding, sizeof(binding));
                    encoder.use_resource(buffer->dispatch_buffer());
                    encoder.use_resource(buffer->command_buffer());
                } else {
                    auto buffer = static_cast<const MetalBuffer *>(buffer_base);
                    auto binding = buffer->binding(argument.buffer.offset, argument.buffer.size);
                    copy(&binding, sizeof(binding));
                    encoder.use_resource(buffer->handle());
                }
                break;
            }
            case Argument::Tag::TEXTURE: {
                auto texture = reinterpret_cast<const MetalTextureBase *>(argument.texture.handle);
                LUISA_ASSERT(
                    texture->kind() != MetalTextureBase::Kind::DEPTH ||
                        (luisa::to_underlying(root.usage) &
                         luisa::to_underlying(Usage::WRITE)) == 0u,
                    "Metal depth textures cannot be written by raster shaders.");
                auto binding = texture->binding(argument.texture.level);
                copy(&binding, sizeof(binding));
                encoder.use_resource(texture->handle(argument.texture.level));
                break;
            }
            case Argument::Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<MetalBindlessArray *>(argument.bindless_array.handle);
                auto binding = array->binding();
                copy(&binding, sizeof(binding));
                array->mark_resource_usages(encoder);
                break;
            }
            case Argument::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(argument.accel.handle);
                auto binding = accel->binding();
                copy(&binding, sizeof(binding));
                accel->mark_resource_usages(encoder);
                break;
            }
            case Argument::Tag::UNIFORM: {
                auto uniform = command->uniform(argument.uniform);
                copy(uniform.data(), uniform.size_bytes());
                break;
            }
        }
    }
    LUISA_ASSERT(dynamic_index == dynamic_arguments.size(),
                 "Metal raster root argument count mismatch (encoded {}, supplied {}).",
                 dynamic_index, dynamic_arguments.size());
    auto encoded_size = luisa::align(offset, argument_alignment);
    LUISA_ASSERT(encoded_size <= _root_argument_size &&
                     (encoded_size == _root_argument_size || _root_arguments.empty()),
                 "Metal raster root ABI mismatch (runtime {}, AIR {}).",
                 encoded_size, _root_argument_size);
    return encoder.upload(argument_buffer.data(), _root_argument_size);
}

void MetalRasterShader::set_name(luisa::string_view name) noexcept {
    std::scoped_lock lock{_name_mutex};
    if (_name != nullptr) {
        _name->release();
        _name = nullptr;
    }
    if (!name.empty()) {
        _name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
    }
    if (_library) { _library->setLabel(_name); }
}

}// namespace luisa::compute::metal
