#include <luisa/core/pool.h>
#include <luisa/core/logging.h>
#include <thread>
#include <luisa/runtime/rhi/pixel.h>
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_accel.h"
#include "metal_mesh.h"
#include "metal_curve.h"
#include "metal_shader.h"
#include "metal_procedural_primitive.h"
#include "metal_motion_instance.h"
#include "metal_bindless_array.h"
#include "metal_depth_buffer.h"
#include "metal_raster_shader.h"
#include "metal_command_encoder.h"
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/backends/ext/registry.h>

namespace luisa::compute::metal {

class UserCallbackContext : public MetalCallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

private:
    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<UserCallbackContext, true> pool;
        return pool;
    }

public:
    explicit UserCallbackContext(CallbackContainer &&cbs) noexcept
        : _functions{std::move(cbs)} {}

    [[nodiscard]] static auto create(CallbackContainer &&cbs) noexcept {
        return _object_pool().create(std::move(cbs));
    }

    void recycle() noexcept override {
        for (auto &&f : _functions) { f(); }
        _object_pool().destroy(this);
    }
};

MetalCommandEncoder::MetalCommandEncoder(MetalStream *stream) noexcept
    : _stream{stream} {}

MetalCommandEncoder::~MetalCommandEncoder() noexcept {
    LUISA_ASSERT(_command_buffer == nullptr && _command_allocator == nullptr,
                 "Metal4 command encoder was destroyed before submission.");
}

void MetalCommandEncoder::_prepare_command_buffer() noexcept {
    if (_command_buffer == nullptr) {
        _command_allocator = device()->newCommandAllocator();
        LUISA_ASSERT(_command_allocator != nullptr,
                     "Failed to create Metal4 command allocator.");
        _command_buffer = device()->newCommandBuffer();
        LUISA_ASSERT(_command_buffer != nullptr,
                     "Failed to create Metal4 command buffer.");
        if (auto name = _stream->name()) {
            _command_buffer->setLabel(name);
        }
        auto options = NS::TransferPtr(
            MTL4::CommandBufferOptions::alloc()->init());
        options->setLogState(_stream->log_state());
        _command_buffer->beginCommandBuffer(
            _command_allocator, options.get());
    }
}

void MetalCommandEncoder::add_callback(MetalCallbackContext *cb) noexcept {
    _callbacks.emplace_back(cb);
}

MTL4::CommandBuffer *MetalCommandEncoder::command_buffer() noexcept {
    _prepare_command_buffer();
    return _command_buffer;
}

MTL4::ComputeCommandEncoder *MetalCommandEncoder::compute_encoder() noexcept {
    _prepare_command_buffer();
    auto encoder = _command_buffer->computeCommandEncoder();
    LUISA_ASSERT(encoder != nullptr,
                 "Failed to create Metal4 compute command encoder.");
    encoder->barrierAfterQueueStages(
        MTL::StageAll, MTL::StageAll,
        MTL4::VisibilityOptionDevice);
    return encoder;
}

MTL4::RenderCommandEncoder *MetalCommandEncoder::render_encoder(
    MTL4::RenderPassDescriptor *descriptor) noexcept {
    _prepare_command_buffer();
    auto encoder = _command_buffer->renderCommandEncoder(descriptor);
    LUISA_ASSERT(encoder != nullptr,
                 "Failed to create Metal4 render command encoder.");
    encoder->barrierAfterQueueStages(
        MTL::StageAll, MTL::StageAll,
        MTL4::VisibilityOptionDevice);
    return encoder;
}

MTL4::ArgumentTable *MetalCommandEncoder::argument_table(
    size_t buffer_count, size_t texture_count,
    size_t sampler_count) noexcept {
    auto descriptor = NS::TransferPtr(
        MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setInitializeBindings(true);
    descriptor->setMaxBufferBindCount(buffer_count);
    descriptor->setMaxTextureBindCount(texture_count);
    descriptor->setMaxSamplerStateBindCount(sampler_count);
    NS::Error *error = nullptr;
    auto table = device()->newArgumentTable(descriptor.get(), &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create Metal4 argument table: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(table != nullptr,
                 "Failed to create Metal4 argument table.");
    add_callback(FunctionCallbackContext::create(
        [table]() noexcept { table->release(); }));
    return table;
}

MTL::GPUAddress MetalCommandEncoder::upload(
    const void *data, size_t size) noexcept {
    if (size == 0u) { return 0u; }
    auto address = MTL::GPUAddress{0u};
    with_upload_buffer(size, [&](auto allocation) noexcept {
        std::memcpy(allocation->data(), data, size);
        use_resource(allocation->buffer());
        address = allocation->buffer()->gpuAddress() + allocation->offset();
    });
    return address;
}

void MetalCommandEncoder::use_resource(
    const MTL::Allocation *allocation) noexcept {
    if (allocation != nullptr) { _allocations.emplace(allocation); }
}

MetalStream::SubmissionHandle MetalCommandEncoder::submit(
    CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        add_callback(UserCallbackContext::create(
            std::move(user_callbacks)));
    }
    auto callbacks = std::exchange(_callbacks, {});
    if (!callbacks.empty()) { _prepare_command_buffer(); }
    auto command_buffer = _command_buffer;
    if (command_buffer != nullptr) {
        if (!_allocations.empty()) {
            auto descriptor = NS::TransferPtr(
                MTL::ResidencySetDescriptor::alloc()->init());
            descriptor->setInitialCapacity(_allocations.size());
            NS::Error *error = nullptr;
            auto residency = device()->newResidencySet(
                descriptor.get(), &error);
            if (error != nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create Metal4 residency set: {}.",
                    error->localizedDescription()->utf8String());
            }
            LUISA_ASSERT(residency != nullptr,
                         "Failed to create Metal4 residency set.");
            luisa::vector<const MTL::Allocation *> allocations;
            allocations.reserve(_allocations.size());
            for (auto allocation : _allocations) {
                allocations.emplace_back(allocation);
            }
            residency->addAllocations(
                allocations.data(), allocations.size());
            residency->commit();
            command_buffer->useResidencySet(residency);
            callbacks.emplace_back(FunctionCallbackContext::create(
                [residency]() noexcept { residency->release(); }));
        }
        command_buffer->endCommandBuffer();
        auto command_allocator = _command_allocator;
        _command_buffer = nullptr;
        _command_allocator = nullptr;
        _allocations.clear();
        return _stream->submit(
            command_buffer, command_allocator,
            std::move(callbacks));
    }
    auto submission = luisa::make_shared<MetalStream::Submission>();
    submission->completed.store(true, std::memory_order_relaxed);
    return submission;
}

void MetalCommandEncoder::submit_and_wait(
    CommandList::CallbackContainer &&user_callbacks) noexcept {
    auto submission = submit(std::move(user_callbacks));
    while (!submission->completed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void MetalCommandEncoder::visit(BufferUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_upload_buffer(size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, size);
        auto encoder = compute_encoder();
        use_resource(upload_buffer->buffer());
        use_resource(buffer);
        encoder->copyFromBuffer(upload_buffer->buffer(),
                                upload_buffer->offset(),
                                buffer, offset, size);
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_download_buffer(size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = compute_encoder();
        use_resource(buffer);
        use_resource(download_buffer->buffer());
        encoder->copyFromBuffer(buffer, offset,
                                download_buffer->buffer(),
                                download_buffer->offset(), size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, size] {
            std::memcpy(data, download_buffer->data(), size);
        }));
    });
}

void MetalCommandEncoder::visit(BufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_buffer = reinterpret_cast<const MetalBuffer *>(command->src_handle())->handle();
    auto dst_buffer = reinterpret_cast<const MetalBuffer *>(command->dst_handle())->handle();
    auto src_offset = command->src_offset();
    auto dst_offset = command->dst_offset();
    auto size = command->size();
    auto encoder = compute_encoder();
    use_resource(src_buffer);
    use_resource(dst_buffer);
    encoder->copyFromBuffer(src_buffer, src_offset, dst_buffer, dst_offset, size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto texture = reinterpret_cast<const MetalTextureBase *>(command->texture())->handle();
    auto texture_level = command->level();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = compute_encoder();
    use_resource(buffer);
    use_resource(texture);
    encoder->copyFromBuffer(buffer, buffer_offset, pitch_size, image_size,
                            MTL::Size{size.x, size.y, size.z},
                            texture, 0u, texture_level,
                            MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    _prepare_command_buffer();
    auto shader = reinterpret_cast<const MetalShader *>(command->handle());
    shader->launch(*this, command);
}

void MetalCommandEncoder::visit(TextureUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTextureBase *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_upload_buffer(total_size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, total_size);
        auto encoder = compute_encoder();
        use_resource(upload_buffer->buffer());
        use_resource(texture);
        encoder->copyFromBuffer(upload_buffer->buffer(), upload_buffer->offset(),
                                pitch_size, image_size, MTL::Size{size.x, size.y, size.z},
                                texture, 0u, level, MTL::Origin{0u, 0u, 0u});
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTextureBase *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_download_buffer(total_size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = compute_encoder();
        use_resource(texture);
        use_resource(download_buffer->buffer());
        encoder->copyFromTexture(texture, 0u, level,
                                 MTL::Origin{0u, 0u, 0u},
                                 MTL::Size{size.x, size.y, size.z},
                                 download_buffer->buffer(),
                                 download_buffer->offset(),
                                 pitch_size, image_size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, total_size] {
            std::memcpy(data, download_buffer->data(), total_size);
        }));
    });
}

void MetalCommandEncoder::visit(TextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_texture = reinterpret_cast<const MetalTextureBase *>(command->src_handle())->handle();
    auto dst_texture = reinterpret_cast<const MetalTextureBase *>(command->dst_handle())->handle();
    auto src_level = command->src_level();
    auto dst_level = command->dst_level();
    auto storage = command->storage();
    auto size = command->size();
    auto encoder = compute_encoder();
    use_resource(src_texture);
    use_resource(dst_texture);
    encoder->copyFromTexture(src_texture, 0u, src_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             dst_texture, 0u, dst_level,
                             MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTextureBase *>(command->texture())->handle();
    auto texture_level = command->level();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = compute_encoder();
    use_resource(texture);
    use_resource(buffer);
    encoder->copyFromTexture(texture, 0u, texture_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             buffer, buffer_offset,
                             pitch_size, image_size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(AccelBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto accel = reinterpret_cast<MetalAccel *>(command->handle());
    accel->build(*this, command);
}

void MetalCommandEncoder::visit(MeshBuildCommand *command) noexcept {
    LUISA_ASSERT(command->vertex_stride() % 16u == 0u, "Vertex stride must be aligned to 16 bytes.");
    _prepare_command_buffer();
    auto mesh = reinterpret_cast<MetalMesh *>(command->handle());
    mesh->build(*this, command);
}

void MetalCommandEncoder::visit(CurveBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto curve = reinterpret_cast<MetalCurve *>(command->handle());
    curve->build(*this, command);
}

void MetalCommandEncoder::visit(ProceduralPrimitiveBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto prim = reinterpret_cast<MetalProceduralPrimitive *>(command->handle());
    prim->build(*this, command);
}

void MetalCommandEncoder::visit(MotionInstanceBuildCommand *command) noexcept {
    auto instance =
        reinterpret_cast<MetalMotionInstance *>(command->handle());
    instance->build(command);
}

void MetalCommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    _prepare_command_buffer();
    auto bindless_array = reinterpret_cast<MetalBindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void MetalCommandEncoder::visit(CustomCommand *command) noexcept {
    _prepare_command_buffer();
    switch (command->custom_cmd_uuid()) {
        case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET): {
            auto clear = static_cast<ClearRenderTargetCommand *>(command);
            auto texture_base = reinterpret_cast<MetalTextureBase *>(clear->handle());
            LUISA_ASSERT(texture_base->kind() == MetalTextureBase::Kind::TEXTURE,
                         "Cannot clear a depth texture as a color render target.");
            auto texture = static_cast<MetalTexture *>(texture_base);
            LUISA_ASSERT(texture->is_raster_target(),
                         "Cannot clear a texture that was not created as a raster target.");
            auto descriptor = NS::TransferPtr(
                MTL4::RenderPassDescriptor::alloc()->init());
            auto attachment = descriptor->colorAttachments()->object(0u);
            attachment->setTexture(texture->handle(clear->level()));
            attachment->setLoadAction(MTL::LoadActionClear);
            attachment->setStoreAction(MTL::StoreActionStore);
            auto value = clear->value();
            attachment->setClearColor(MTL::ClearColor::Make(
                value.x, value.y, value.z, value.w));
            use_resource(texture->handle(clear->level()));
            auto encoder = render_encoder(descriptor.get());
            encoder->endEncoding();
            break;
        }
        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
            auto clear = static_cast<ClearDepthCommand *>(command);
            auto texture = reinterpret_cast<MetalTextureBase *>(clear->handle());
            LUISA_ASSERT(texture->kind() == MetalTextureBase::Kind::DEPTH,
                         "Cannot clear a color texture as a depth buffer.");
            auto depth = static_cast<MetalDepthBuffer *>(texture);
            auto descriptor = NS::TransferPtr(
                MTL4::RenderPassDescriptor::alloc()->init());
            auto attachment = descriptor->depthAttachment();
            attachment->setTexture(depth->handle());
            attachment->setLoadAction(MTL::LoadActionClear);
            attachment->setStoreAction(MTL::StoreActionStore);
            attachment->setClearDepth(clear->value());
            use_resource(depth->handle());
            auto encoder = render_encoder(descriptor.get());
            encoder->endEncoding();
            break;
        }
        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
            auto draw = static_cast<DrawRasterSceneCommand *>(command);
            auto shader = reinterpret_cast<MetalRasterShader *>(draw->handle());
            LUISA_ASSERT(shader->matches_mesh_format(draw->mesh_format()),
                         "Draw-time MeshFormat does not match the format used to compile the Metal raster shader.");
            auto render_targets = draw->rtv_texs();
            LUISA_ASSERT(render_targets.size() == shader->fragment_output_count(),
                         "Metal raster shader returns {} fragment output(s), but the draw binds {} color attachment(s).",
                         shader->fragment_output_count(), render_targets.size());
            LUISA_ASSERT(render_targets.size() <= 8u,
                         "Metal supports at most 8 color attachments.");
            luisa::fixed_vector<MetalTexture *, 8u> color_targets;
            auto descriptor = NS::TransferPtr(
                MTL4::RenderPassDescriptor::alloc()->init());
            size_t target_width = 0u;
            size_t target_height = 0u;
            auto set_target_size = [&](MTL::Texture *texture) noexcept {
                if (target_width == 0u) {
                    target_width = texture->width();
                    target_height = texture->height();
                } else {
                    LUISA_ASSERT(target_width == texture->width() &&
                                     target_height == texture->height(),
                                 "Metal render-pass attachments must have identical dimensions.");
                }
            };
            color_targets.reserve(render_targets.size());
            for (auto i = 0u; i < render_targets.size(); i++) {
                auto target_base = reinterpret_cast<MetalTextureBase *>(render_targets[i].handle);
                LUISA_ASSERT(target_base->kind() == MetalTextureBase::Kind::TEXTURE,
                             "Depth texture {} cannot be bound as a color attachment.", i);
                auto target = static_cast<MetalTexture *>(target_base);
                LUISA_ASSERT(target->is_raster_target(),
                             "Color attachment {} was not created as a raster target.", i);
                color_targets.emplace_back(target);
                auto texture = target->handle(render_targets[i].level);
                use_resource(texture);
                set_target_size(texture);
                auto attachment = descriptor->colorAttachments()->object(i);
                attachment->setTexture(texture);
                attachment->setLoadAction(MTL::LoadActionLoad);
                attachment->setStoreAction(MTL::StoreActionStore);
            }
            MetalDepthBuffer *depth_target = nullptr;
            if (draw->dsv_tex().handle != invalid_resource_handle) {
                auto target = reinterpret_cast<MetalTextureBase *>(draw->dsv_tex().handle);
                LUISA_ASSERT(target->kind() == MetalTextureBase::Kind::DEPTH,
                             "A color texture cannot be bound as a Metal depth attachment.");
                depth_target = static_cast<MetalDepthBuffer *>(target);
                use_resource(depth_target->handle());
                set_target_size(depth_target->handle());
                auto attachment = descriptor->depthAttachment();
                attachment->setTexture(depth_target->handle());
                attachment->setLoadAction(MTL::LoadActionLoad);
                attachment->setStoreAction(MTL::StoreActionStore);
            }
            LUISA_ASSERT(target_width != 0u && target_height != 0u,
                         "Metal raster draw has no render target.");
            auto state = draw->raster_state();
            auto encoder = render_encoder(descriptor.get());
            switch (state.cull_mode) {
                case CullMode::None: encoder->setCullMode(MTL::CullModeNone); break;
                case CullMode::Front: encoder->setCullMode(MTL::CullModeFront); break;
                case CullMode::Back: encoder->setCullMode(MTL::CullModeBack); break;
            }
            encoder->setFrontFacingWinding(
                state.front_counter_clockwise ?
                    MTL::WindingCounterClockwise :
                    MTL::WindingClockwise);
            encoder->setTriangleFillMode(
                state.fill_mode == FillMode::Solid ?
                    MTL::TriangleFillModeFill :
                    MTL::TriangleFillModeLines);
            encoder->setDepthClipMode(
                state.depth_clip ? MTL::DepthClipModeClip : MTL::DepthClipModeClamp);
            auto viewport = draw->viewport();
            LUISA_ASSERT(viewport.size.x != 0u && viewport.size.y != 0u,
                         "Metal raster viewport must be non-zero.");
            LUISA_ASSERT(static_cast<size_t>(viewport.start.x) + viewport.size.x <= target_width &&
                             static_cast<size_t>(viewport.start.y) + viewport.size.y <= target_height,
                         "Metal raster viewport exceeds the render-target dimensions.");
            encoder->setViewport(MTL::Viewport{
                static_cast<double>(viewport.start.x),
                static_cast<double>(viewport.start.y),
                static_cast<double>(viewport.size.x),
                static_cast<double>(viewport.size.y),
                0.0, 1.0});
            encoder->setScissorRect(MTL::ScissorRect{
                viewport.start.x, viewport.start.y,
                viewport.size.x, viewport.size.y});
            auto root_argument_address =
                shader->encode_arguments(*this, draw);
            auto primitive = [&]() noexcept {
                switch (state.topology) {
                    case TopologyType::Point: return MTL::PrimitiveTypePoint;
                    case TopologyType::Line: return MTL::PrimitiveTypeLine;
                    case TopologyType::Triangle: return MTL::PrimitiveTypeTriangle;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid Metal raster topology.");
            }();
            auto packed_vertex_stride = [&](size_t stream) noexcept {
                auto size = static_cast<size_t>(0u);
                for (auto attribute : shader->mesh_format().attributes(stream)) {
                    size += pixel_format_size(
                        attribute.format, make_uint3(1u));
                }
                return size;
            };
            for (auto &&mesh : draw->scene()) {
                auto vertex_buffers = mesh.vertex_buffers();
                LUISA_ASSERT(vertex_buffers.size() == shader->mesh_format().vertex_stream_count(),
                             "Metal raster mesh has {} vertex stream(s), but the shader expects {}.",
                             vertex_buffers.size(), shader->mesh_format().vertex_stream_count());
                LUISA_ASSERT(vertex_buffers.size() <= 4u,
                             "Metal raster meshes support at most 4 vertex streams.");
                std::array<size_t, 4u> vertex_strides{};
                for (auto i = 0u; i < vertex_buffers.size(); i++) {
                    auto &&view = vertex_buffers[i];
                    auto required_stride = packed_vertex_stride(i);
                    LUISA_ASSERT(required_stride != 0u &&
                                     view.stride() >= required_stride &&
                                     view.size() >= required_stride,
                                 "Metal vertex stream {} has stride {} and size {}, "
                                 "but its compiled attributes require {} bytes.",
                                 i, view.stride(), view.size(), required_stride);
                    vertex_strides[i] = view.stride();
                    auto buffer = reinterpret_cast<MetalBuffer *>(view.handle());
                    auto buffer_size = buffer->handle()->length();
                    LUISA_ASSERT(view.offset() <= buffer_size &&
                                     view.size() <= buffer_size - view.offset(),
                                 "Metal vertex buffer view for stream {} is out of bounds.", i);
                    use_resource(buffer->handle());
                }
                auto pipeline = shader->pipeline(
                    color_targets, depth_target, state,
                    luisa::span<const size_t>{vertex_strides.data(), vertex_buffers.size()});
                encoder->setRenderPipelineState(pipeline.render.get());
                encoder->setDepthStencilState(pipeline.depth.get());
                use_resource(pipeline.render.get());
                auto object_id = mesh.object_id();
                auto table = argument_table(2u + vertex_buffers.size());
                table->setAddress(root_argument_address, 0u);
                table->setAddress(upload(&object_id, sizeof(object_id)), 1u);
                for (auto i = 0u; i < vertex_buffers.size(); i++) {
                    auto &&view = vertex_buffers[i];
                    auto buffer = reinterpret_cast<MetalBuffer *>(view.handle());
                    table->setAddress(
                        buffer->handle()->gpuAddress() + view.offset(), i + 2u);
                }
                encoder->setArgumentTable(
                    table, MTL::RenderStageVertex | MTL::RenderStageFragment);
                luisa::visit(
                    [&]<typename T>(const T &index) noexcept {
                        if constexpr (std::is_same_v<T, uint>) {
                            LUISA_ASSERT(mesh.vertex_offset() >= 0,
                                         "Non-indexed Metal draws do not support a negative vertex offset.");
                            if (index != 0u) {
                                auto first_vertex = static_cast<size_t>(mesh.vertex_offset());
                                for (auto i = 0u; i < vertex_buffers.size(); i++) {
                                    auto &&view = vertex_buffers[i];
                                    auto required_stride = packed_vertex_stride(i);
                                    auto available_vertices =
                                        (view.size() - required_stride) /
                                            view.stride() +
                                        1u;
                                    LUISA_ASSERT(first_vertex < available_vertices &&
                                                     index <= available_vertices - first_vertex,
                                                 "Non-indexed Metal draw exceeds vertex stream {}.",
                                                 i);
                                }
                            }
                            encoder->drawPrimitives(
                                primitive,
                                static_cast<NS::UInteger>(mesh.vertex_offset()),
                                index, mesh.instance_count());
                        } else {
                            auto buffer = reinterpret_cast<MetalBuffer *>(index.handle());
                            auto buffer_size = buffer->handle()->length();
                            LUISA_ASSERT(index.offset_bytes() <= buffer_size &&
                                             index.size_bytes() <= buffer_size - index.offset_bytes(),
                                         "Metal index buffer view is out of bounds.");
                            LUISA_ASSERT(index.offset_bytes() % alignof(uint) == 0u,
                                         "Metal uint index-buffer offsets must be 4-byte aligned.");
                            use_resource(buffer->handle());
                            encoder->drawIndexedPrimitives(
                                primitive,
                                index.size_bytes() / sizeof(uint),
                                MTL::IndexTypeUInt32,
                                buffer->handle()->gpuAddress() + index.offset_bytes(),
                                index.size_bytes(),
                                mesh.instance_count(), mesh.vertex_offset(), 0u);
                        }
                    },
                    mesh.index());
            }
            encoder->endEncoding();
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Custom command (uuid = 0x{:04x}) is not supported in Metal backend.",
                command->custom_cmd_uuid());
    }
}

}// namespace luisa::compute::metal
