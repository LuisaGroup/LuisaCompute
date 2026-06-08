#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCDevice.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <Shader/ComputeShader.h>
#include <Resource/RenderTexture.h>
#include <Resource/TopAccel.h>
#include <DXApi/LCSwapChain.h>
#include "../../common/command_reorder_visitor.h"
#include <Shader/RasterShader.h>
#include <luisa/core/stl/variant.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/aabb.h>
#include <Resource/DepthBuffer.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/runtime/swapchain.h>
#include <Resource/SparseTexture.h>
#include <luisa/backends/ext/dx_custom_cmd.h>
#include "../../common/shader_print_formatter.h"
#ifdef LCDX_ENABLE_WINPIX
#include <WinPixEventRuntime/pix3.h>
#endif

namespace lc::dx {
auto get_resource_view(DXCustomCmd::ResourceHandle const &res) {
    return luisa::visit(
        [&]<typename T>(T const &t) -> EnhancedBarrierTracker::ResourceView {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                return EnhancedBarrierTracker::ResourceView{
                    BufferView{
                        static_cast<Buffer const *>(reinterpret_cast<Resource const *>(t.handle)),
                        t.offset,
                        t.size}};
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                return EnhancedBarrierTracker::ResourceView{
                    EnhancedBarrierTracker::TexView{
                        static_cast<TextureBase const *>(reinterpret_cast<Resource const *>(t.handle)),
                        t.level}};
            } else {
                auto buffer = static_cast<BindlessArray const *>(reinterpret_cast<Resource const *>(t.handle))->BindlessBuffer();
                return EnhancedBarrierTracker::ResourceView{
                    BufferView{
                        buffer,
                        0,
                        buffer->GetByteSize()}};
            }
        },
        res);
};
CmdQueueBase::CmdQueueBase(Device *device, CmdQueueTag tag)
    : Resource{device}, _tag{tag},
      log_callback([](luisa::string_view str) {
          LUISA_INFO("[DEVICE] {}", str);
      }) {}
using Argument = luisa::compute::Argument;
static bool is_device_buffer(Resource const *res) {
    auto tag = res->get_tag();
    return (tag != Resource::Tag::UploadBuffer) && (tag != Resource::Tag::ReadbackBuffer);
}
template<typename Visitor>
void decode_cmd(vstd::span<const Argument> args, Visitor &&visitor) {
    using Tag = Argument::Tag;
    for (auto &&i : args) {
        switch (i.tag) {
            case Tag::BUFFER: {
                visitor(i.buffer);
            } break;
            case Tag::TEXTURE: {
                visitor(i.texture);
            } break;
            case Tag::UNIFORM: {
                visitor(i.uniform);
            } break;
            case Tag::BINDLESS_ARRAY: {
                visitor(i.bindless_array);
            } break;
            case Tag::ACCEL: {
                visitor(i.accel);
            } break;
            default: {
                LUISA_ASSUME(false);
            } break;
        }
    }
}
class LCPreProcessVisitor : public CommandVisitor {
public:
    CommandBufferBuilder *bd{};
    EnhancedBarrierTracker *state_tracker{};
    vstd::vector<std::pair<size_t, size_t>> *arg_vecs{};
    vstd::vector<uint8_t> *arg_buffer{};
    vstd::vector<BottomAccelData> *bottom_accel_datas{};
    vstd::fixed_vector<std::pair<size_t, size_t>, 4> *accel_offset{};
    size_t build_accel_size = 0;
    void add_build_accel(size_t size) {
        size = CalcAlign(size, 256);
        accel_offset->emplace_back(build_accel_size, size);
        build_accel_size += size;
    }
    void uniform_align(size_t align) const {
        luisa::vector_resize(*arg_buffer, CalcAlign(arg_buffer->size(), align));
    }
    template<typename T>
    void emplace_data(T const &data, size_t alignment) {
        size_t sz = arg_buffer->size();
        alignment -= 1;
        auto aligned_size = (sz + alignment) & (~alignment);
        luisa::enlarge_by(*arg_buffer, sizeof(T) + aligned_size - sz);
        using PlaceHolder = luisa::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(arg_buffer->data() + aligned_size) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void emplace_data(T const *data, size_t size, size_t alignment) {
        alignment -= 1;
        size_t sz = arg_buffer->size();
        auto aligned_size = (sz + alignment) & (~alignment);
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(*arg_buffer, byteSize + aligned_size - sz);
        std::memcpy(arg_buffer->data() + aligned_size, data, byteSize);
    }
    struct Visitor {
        LCPreProcessVisitor *self;
        SavedArgument const *arg;
        ShaderDispatchCommandBase const &cmd;
        EnhancedBarrierTracker::Usage uav_usage;
        EnhancedBarrierTracker::Usage read_usage;
        EnhancedBarrierTracker::Usage accel_read_usage;
        uint validation_count = 0;
        Visitor(
            LCPreProcessVisitor *self,
            SavedArgument const *arg,
            ShaderDispatchCommandBase const &cmd,
            bool is_raster,
            uint validation_count = 0) : self(self), arg(arg), cmd(cmd), validation_count(validation_count) {
            if (is_raster) {
                uav_usage = EnhancedBarrierTracker::Usage::RasterUAV;
                read_usage = EnhancedBarrierTracker::Usage::RasterRead;
                accel_read_usage = EnhancedBarrierTracker::Usage::RasterAccelRead;
            } else {
                uav_usage = EnhancedBarrierTracker::Usage::ComputeUAV;
                read_usage = EnhancedBarrierTracker::Usage::ComputeRead;
                accel_read_usage = EnhancedBarrierTracker::Usage::ComputeAccelRead;
            }
        }
        void operator()(Argument::Buffer const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);
            if (((uint)arg->var_usage & (uint)Usage::WRITE) != 0) {
                LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
                self->state_tracker->Record(
                    BufferView{res, bf.offset, bf.size},
                    uav_usage);
                // self->state_tracker->RecordState(
                //     res,
                //     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                //     true);
            } else {
                if (is_device_buffer(res))
                    self->state_tracker->Record(
                        BufferView{res, bf.offset, bf.size},
                        read_usage);
                else {
                    LUISA_ASSERT(res->get_tag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as shader's resource.");
                }
            }
            // Emplace buffer validation size (element count) when debug info is enabled
            if (bf.handle != 0 && validation_count > 0) {
                auto element_size = arg->struct_size;
                if (element_size > 0) {
                    self->emplace_data(static_cast<uint>(bf.size / element_size), /*alignment=*/4);
                } else {
                    self->emplace_data(static_cast<uint>(bf.size), /*alignment=*/4);
                }
            }
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->var_usage & (uint)Usage::WRITE) != 0) {
                if (!rt->AllowUAV()) [[unlikely]] {
                    LUISA_ERROR("Texture not allowed for Unordered-Access.");
                }
                self->state_tracker->Record(
                    EnhancedBarrierTracker::TexView{rt, bf.level},
                    uav_usage);
            }
            // SRV
            else {
                self->state_tracker->Record(
                    EnhancedBarrierTracker::TexView{rt, bf.level},
                    read_usage);
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            vstd::fixed_vector<vstd::HashMap<Resource const *, size_t>::Index, 16> writeMap;
            auto &write_state_map = self->state_tracker->WriteStateMap();
            arr->Lock();
            for (auto iter = write_state_map.begin(); iter != write_state_map.end(); ++iter) {
                auto &i = *iter;
                if (arr->IsPtrInBindless(reinterpret_cast<size_t>(i.first))) {
                    writeMap.emplace_back(write_state_map.get_index(iter));
                }
            }
            arr->Unlock();

            for (auto &&iter : writeMap) {
                self->state_tracker->Record(
                    iter.key(),
                    EnhancedBarrierTracker::Range(0, iter.value()),
                    read_usage);
                write_state_map.remove(iter);
            }
            self->state_tracker->Record(
                BufferView(arr->BindlessBuffer()),
                read_usage);
            ++arg;
            // Emplace bindless array validation size when debug info is enabled
            if (bf.handle != 0 && validation_count > 0) {
                auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
                self->emplace_data(static_cast<uint>(arr->size()), /*alignment=*/4);
            }
        }
        void operator()(Argument::Uniform const &a) {
            auto bf = cmd.uniform(a);
            self->emplace_data(bf.data(), bf.size_bytes(), a.alignment);
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if (accel->GetInstBuffer()) [[likely]] {
                if (((uint)arg->var_usage & (uint)Usage::WRITE) != 0) {
                    self->state_tracker->Record(
                        BufferView{accel->GetInstBuffer(), 0, accel->GetInstBuffer()->GetByteSize()},
                        uav_usage);
                } else {
                    self->state_tracker->Record(
                        BufferView{accel->GetInstBuffer(), 0, accel->GetInstBuffer()->GetByteSize()},
                        read_usage);
                    auto accelBuffer = accel->GetAccelBuffer();
                    self->state_tracker->Record(
                        BufferView{accelBuffer, 0, accelBuffer->GetByteSize()},
                        accel_read_usage);
                }
            } else {
                LUISA_ERROR("Accel not initialized.");
            }
            ++arg;
        }
    };
    void visit(const DXCustomCmd *cmd) noexcept {
        for (auto i : const_cast<DXCustomCmd *>(cmd)->get_resource_usages()) {
            auto res_view = get_resource_view(i.resource);
            state_tracker->Record(res_view, i.required_state);
        }
        for (auto i : const_cast<DXCustomCmd *>(cmd)->get_enhanced_resource_usages()) {
            auto res_view = get_resource_view(i.resource);
            state_tracker->Record(
                res_view,
                i.sync,
                i.access,
                i.texture_layout);
        }
    }
    void visit(const BufferUploadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res)) {
            state_tracker->Record(
                BufferView(res, cmd->offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopyDest);
            // state_tracker->RecordState(res, D3D12_RESOURCE_STATE_COPY_DEST);
        } else {
            LUISA_ERROR("Host-buffer should not be used to upload.");
        }
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res)) {
            state_tracker->Record(
                BufferView(res, cmd->offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopySource);
        } else {
            LUISA_ERROR("Host-buffer should not be used to download.");
        }
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        if (is_device_buffer(srcBf)) {
            state_tracker->Record(
                BufferView(srcBf, cmd->src_offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopySource);
        } else {
            LUISA_ASSERT(srcBf->get_tag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
        if (is_device_buffer(dstBf)) {
            state_tracker->Record(
                BufferView(dstBf, cmd->dst_offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopyDest);
        } else {
            LUISA_ASSERT(dstBf->get_tag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopyDest);
        if (is_device_buffer(bf)) {
            state_tracker->Record(
                BufferView(bf, cmd->buffer_offset(), pixel_storage_size(cmd->storage(), cmd->size())),
                EnhancedBarrierTracker::Usage::CopySource);
        } else {
            LUISA_ASSERT(bf->get_tag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
    }

    void visit(const TextureUploadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopyDest);
    }
    void visit(const ClearDepthCommand *cmd) noexcept {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, 0),
            EnhancedBarrierTracker::Usage::DepthWrite);
    }
    void visit(const ClearRenderTargetCommand *cmd) noexcept {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::RenderTarget);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopySource);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(src, cmd->src_level()),
            EnhancedBarrierTracker::Usage::CopySource);
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(dst, cmd->dst_level()),
            EnhancedBarrierTracker::Usage::CopyDest);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        state_tracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopySource);
        if (is_device_buffer(bf)) {
            state_tracker->Record(
                BufferView(bf, cmd->buffer_offset(), pixel_storage_size(cmd->storage(), cmd->size())),
                EnhancedBarrierTracker::Usage::CopyDest);
        } else {
            LUISA_ASSERT(bf->get_tag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        auto cs = reinterpret_cast<ComputeShader *>(cmd->handle());
        uniform_align(32);
        size_t beforeSize = arg_buffer->size();
        Visitor visitor{this, cs->args().data(), *cmd, false, cs->validation_count()};
        decode_cmd(cs->arg_bindings(), visitor);
        decode_cmd(cmd->arguments(), visitor);
        size_t afterSize = arg_buffer->size();
        arg_vecs->emplace_back(beforeSize, afterSize - beforeSize);
        if (cmd->is_indirect()) {
            auto buffer = reinterpret_cast<Buffer *>(cmd->indirect_dispatch().handle);
            state_tracker->Record(
                BufferView(buffer, cmd->indirect_dispatch().offset / ComputeShader::kDispatchIndirectStride, cmd->indirect_dispatch().max_dispatch_size / ComputeShader::kDispatchIndirectStride), EnhancedBarrierTracker::Usage::IndirectArgs);
        }
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        if (!cmd->update_instance_buffer_only()) {
            add_build_accel(
                accel->PreProcess(
                    *state_tracker,
                    *bd,
                    cmd->instance_count(),
                    cmd->modifications(),
                    cmd->request() == AccelBuildRequest::PREFER_UPDATE));
        } else {
            accel->PreProcessInst(
                *state_tracker,
                *bd,
                cmd->instance_count(),
                cmd->modifications());
        }
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        BottomAccel::MeshOptions meshOptions{
            .vHandle = reinterpret_cast<Buffer const *>(cmd->vertex_buffer()),
            .vOffset = cmd->vertex_buffer_offset(),
            .vStride = cmd->vertex_stride(),
            .vSize = cmd->vertex_buffer_size(),
            .iHandle = reinterpret_cast<Buffer const *>(cmd->triangle_buffer()),
            .iOffset = cmd->triangle_buffer_offset(),
            .iSize = cmd->triangle_buffer_size()};
        add_build_accel(
            accel->PreProcessStates(
                *bd,
                *state_tracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                meshOptions,
                bottom_accel_datas->emplace_back()));
    }
    void visit(const MotionInstanceBuildCommand *) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void visit(const ProceduralPrimitiveBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        BottomAccel::AABBOptions aabbOptions{
            .aabbBuffer = reinterpret_cast<Buffer const *>(cmd->aabb_buffer()),
            .offset = cmd->aabb_buffer_offset(),
            .size = cmd->aabb_buffer_size()};
        add_build_accel(
            accel->PreProcessStates(
                *bd,
                *state_tracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                aabbOptions,
                bottom_accel_datas->emplace_back()));
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        // reinterpret_cast<BindlessArray *>(cmd->handle())->Bind(cmd->modifications());
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        if (!cmd->empty())
            arr->PreProcessStates(
                *bd,
                *state_tracker);
    };

    void visit(const CustomCommand *cmd) noexcept override {
        switch (cmd->custom_cmd_uuid()) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET):
                visit(static_cast<ClearRenderTargetCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<DXCustomCmd const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }

    void visit(const DrawRasterSceneCommand *cmd) noexcept {
        auto cs = reinterpret_cast<RasterShader *>(cmd->handle());
        uniform_align(32);
        size_t beforeSize = arg_buffer->size();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        decode_cmd(cmd->arguments(), Visitor{this, cs->args().data(), *cmd, true, cs->validation_count()});
        size_t afterSize = arg_buffer->size();
        arg_vecs->emplace_back(beforeSize, afterSize - beforeSize);

        for (auto &&mesh : cmd->scene()) {
            for (auto &&v : mesh.vertex_buffers()) {
                state_tracker->Record(
                    BufferView(reinterpret_cast<Buffer *>(v.handle()), v.offset(), v.size()),
                    EnhancedBarrierTracker::Usage::VertexRead);
            }
            auto &&i = mesh.index();
            if (i.index() == 0) {
                auto &&idx = luisa::get<0>(i);
                state_tracker->Record(
                    BufferView(reinterpret_cast<Buffer *>(idx.handle()), idx.offset_bytes(), idx.size_bytes()),
                    EnhancedBarrierTracker::Usage::IndexRead);
            }
        }
        for (auto &&i : rtvs) {
            state_tracker->Record(
                EnhancedBarrierTracker::TexView(
                    reinterpret_cast<TextureBase *>(i.handle),
                    i.level),
                EnhancedBarrierTracker::Usage::RenderTarget);
        }
        if (dsv.handle != ~0ull) {
            state_tracker->Record(
                EnhancedBarrierTracker::TexView(
                    reinterpret_cast<TextureBase *>(dsv.handle),
                    dsv.level),
                EnhancedBarrierTracker::Usage::DepthWrite);
        }
    }
};
#ifdef LCDX_ENABLE_WINPIX
inline DWORD get_pix_color() {
    return ~0;
}
#endif
class LCCmdVisitor : public CommandVisitor {
public:
    Device *device{};
    luisa::function<void(luisa::string_view)> *logger{};
    CommandBufferBuilder *bd{};
    EnhancedBarrierTracker *state_tracker{};
    BufferView arg_buffer{};
    Buffer const *accel_scratch_buffer{};
    std::pair<size_t, size_t> *accel_scratch_offsets{};
    std::pair<size_t, size_t> *buffer_vec{};
    vstd::vector<BindProperty> *bind_props{};
    vstd::vector<ButtomCompactCmd> *update_accel{};
    vstd::vector<D3D12_VERTEX_BUFFER_VIEW> *vbv{};
    BottomAccelData *bottom_accel_data{};
    vstd::func_ptr_t<void(Device *, CommandBufferBuilder *)>
        after_custom_cmd{};

    void visit(const BufferUploadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Buffer upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->upload(bf, cmd->data());
    }

    void visit(const BufferDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Buffer download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->readback(
            bf,
            cmd->data());
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Buffer copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        bd->copy_buffer(
            srcBf,
            dstBf,
            cmd->src_offset(),
            cmd->dst_offset(),
            cmd->size());
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Buffer copy to texture");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->copy_buffer_texture(
            BufferView{bf, cmd->buffer_offset()},
            rt,
            cmd->texture_offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture,
            true);
    }

    void visit(const MotionInstanceBuildCommand *) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }

    struct Visitor {
        LCCmdVisitor *self;
        SavedArgument const *arg;

        void operator()(Argument::Buffer const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);

            self->bind_props->emplace_back(
                BufferView(res, bf.offset));
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->var_usage & (uint)Usage::WRITE) != 0) {
                self->bind_props->emplace_back(
                    DescriptorHeapView(
                        self->device->global_heap.get(),
                        rt->GetGlobalUAVIndex(bf.level)));
            }
            // SRV
            else {
                self->bind_props->emplace_back(
                    DescriptorHeapView(
                        self->device->global_heap.get(),
                        rt->GetGlobalSRVIndex(bf.level)));
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            auto res = arr->BindlessBuffer();
            self->bind_props->emplace_back(
                BufferView(res, 0));
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if ((static_cast<uint>(arg->var_usage) & static_cast<uint>(Usage::WRITE)) == 0) {
                self->bind_props->emplace_back(
                    accel);
            }
            self->bind_props->emplace_back(
                BufferView(accel->GetInstBuffer()));
            ++arg;
        }
        void operator()(Argument::Uniform const &) {
            ++arg;
        }
    };
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        GraphicsCmdlistBarrierCallback barrier_callback(*bd);
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Shader dispatch");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        bind_props->clear();
        auto shader = reinterpret_cast<ComputeShader const *>(cmd->handle());
        auto &&tempBuffer = *buffer_vec;
        buffer_vec++;
        auto cs = static_cast<ComputeShader const *>(shader);
        BufferView readback_count_buffer;
        BufferView readback_buffer;
        BufferView count_buffer;
        BufferView data_buffer;
        CommandAllocator *alloc{nullptr};
        auto BeforeDispatch = [&]() {
            bind_props->emplace_back(DescriptorHeapView(device->sampler_heap.get()));
            if (tempBuffer.second > 0) {
                bind_props->emplace_back(BufferView(arg_buffer.buffer, arg_buffer.offset + tempBuffer.first, tempBuffer.second));
            }
            DescriptorHeapView global_heapView(device->global_heap.get());
            vstd::push_back_func(*bind_props, shader->bindless_count(), [&] { return global_heapView; });
            Visitor visitor{this, cs->args().data()};
            decode_cmd(shader->arg_bindings(), visitor);
            decode_cmd(cmd->arguments(), visitor);
            auto printers = shader->printers();
            if (!printers.empty()) [[unlikely]] {
                alloc = bd->get_cb()->get_alloc();
                static const uint zero = 0;
                auto upload_buffer = alloc->get_temp_upload_buffer(sizeof(uint), 16);
                count_buffer = alloc->get_temp_default_buffer(sizeof(uint), 16);
                readback_count_buffer = alloc->get_temp_readback_buffer(sizeof(uint), 16);
                data_buffer = alloc->get_temp_default_buffer(1024ull * 1024ull, 16);
                readback_buffer = alloc->get_temp_readback_buffer(1024ull * 1024ull, 16);
                static_cast<UploadBuffer const *>(upload_buffer.buffer)->CopyData(upload_buffer.offset, {reinterpret_cast<uint8_t const *>(&zero), sizeof(uint)});
                state_tracker->Record(
                    BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::CopyDest);
                state_tracker->UpdateState(&barrier_callback);
                bd->copy_buffer(upload_buffer.buffer, count_buffer.buffer, upload_buffer.offset, count_buffer.offset, sizeof(uint));
                state_tracker->Record(
                    BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                state_tracker->Record(
                    BufferView(data_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                state_tracker->UpdateState(&barrier_callback);
                bind_props->emplace_back(count_buffer);
                bind_props->emplace_back(data_buffer);
            }
        };
        if (cmd->is_indirect()) {
            auto &&t = cmd->indirect_dispatch();
            auto buffer = reinterpret_cast<Buffer *>(t.handle);
            bind_props->emplace_back();
            BeforeDispatch();
            bd->dispatch_compute_indirect(cs, *buffer, t.offset, t.max_dispatch_size, *bind_props);
        } else if (cmd->is_multiple_dispatch()) {
            size_t bindCount = bind_props->size();
            bind_props->emplace_back();
            BeforeDispatch();
            auto sizes = cmd->dispatch_sizes();
            bd->dispatch_compute(
                cs,
                sizes,
                bindCount,
                *bind_props);
        } else {
            auto &&t = cmd->dispatch_size();
            bind_props->emplace_back(4, make_uint4(t, 0));
            BeforeDispatch();
            bd->dispatch_compute(
                cs,
                t,
                *bind_props);
        }
        if (logger && data_buffer.buffer != nullptr) [[unlikely]] {
            state_tracker->Record(
                BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                EnhancedBarrierTracker::Usage::CopySource);
            state_tracker->Record(
                BufferView(data_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                EnhancedBarrierTracker::Usage::CopySource);
            state_tracker->UpdateState(&barrier_callback);
            bd->copy_buffer(count_buffer.buffer, readback_count_buffer.buffer, count_buffer.offset, readback_count_buffer.offset, sizeof(uint));
            bd->copy_buffer(data_buffer.buffer, readback_buffer.buffer, data_buffer.offset, readback_buffer.offset, data_buffer.byteSize);
            alloc->execute_after_complete([logger = this->logger, shader, readback_count_buffer, readback_buffer]() {
                uint size{0};
                static_cast<ReadbackBuffer const *>(readback_count_buffer.buffer)
                    ->CopyData(
                        readback_count_buffer.offset,
                        {reinterpret_cast<uint8_t *>(&size), sizeof(uint)});
                if (size == 0) return;
                vstd::vector<std::byte> data;
                luisa::enlarge_by(data, std::min<size_t>(readback_buffer.byteSize, size));
                static_cast<ReadbackBuffer const *>(readback_buffer.buffer)
                    ->CopyData(
                        readback_buffer.offset,
                        {reinterpret_cast<uint8_t *>(data.data()), data.size()});
                auto printers = shader->printers();
                size_t offset = 0;
                const auto ptr = data.data();
                const auto end = data.size();
                while (offset < end) {
                    uint flagTypeIdx = *reinterpret_cast<uint32_t *>(ptr + offset);
                    auto &type = printers[flagTypeIdx];
                    ShaderPrintFormatter formatter{type.first, type.second, false};
                    luisa::string result;
                    auto align = std::max<size_t>(4, type.second->alignment());
                    formatter(result, {ptr + offset + align, type.second->size()});
                    size_t ele_size = align + type.second->size();
                    ele_size = ((ele_size + 15ull) & (~15ull));
                    offset += ele_size;
                    (*logger)(result);
                }
            });
        }
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Texture upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::get_copy_texture_buffer_size(
            rt,
            cmd->size());
        auto bfView = bd->get_cb()->get_alloc()->get_temp_upload_buffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        auto uploadBuffer = static_cast<UploadBuffer const *>(bfView.buffer);
        if (copyInfo.bufferSize == copyInfo.alignedBufferSize) {
            uploadBuffer->CopyData(
                bfView.offset,
                {reinterpret_cast<uint8_t const *>(cmd->data()),
                 bfView.byteSize});
        } else {
            size_t bufferOffset = bfView.offset;
            size_t leftedSize = copyInfo.bufferSize;
            auto dataPtr = reinterpret_cast<uint8_t const *>(cmd->data());
            while (leftedSize > 0) {
                uploadBuffer->CopyData(
                    bufferOffset,
                    {dataPtr, copyInfo.copySize});
                dataPtr += copyInfo.copySize;
                leftedSize -= copyInfo.copySize;
                bufferOffset += copyInfo.stepSize;
            }
        }
        bd->copy_buffer_texture(
            bfView,
            rt,
            cmd->offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture,
            false);
    }
    void visit(const ClearDepthCommand *cmd) noexcept {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Clear depth");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto cmdList = bd->get_cb()->cmd_list();
        auto alloc = bd->get_cb()->get_alloc();
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
        auto chunk = alloc->dsv_allocator.allocate(1);
        auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
        dsvHandle = descHeap->hCPU(chunk.offset);
        D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc{
            .Format = static_cast<DXGI_FORMAT>(rt->Format()),
            .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
            .Flags = D3D12_DSV_FLAG_NONE};
        viewDesc.Texture2D.MipSlice = 0;
        device->device->CreateDepthStencilView(rt->GetResource(), &viewDesc, dsvHandle);
        D3D12_CLEAR_FLAGS clearFlags = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
        cmdList->ClearDepthStencilView(dsvHandle, clearFlags, cmd->value(), 0, 0, nullptr);
    }
    void visit(const ClearRenderTargetCommand *cmd) {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto cmdList = bd->get_cb()->cmd_list();
        FLOAT values[4];
        float4 c_values = cmd->value();
        std::memcpy(values, &c_values, sizeof(float4));

        auto alloc = bd->get_cb()->get_alloc();
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
        auto chunk = alloc->rtv_allocator.allocate(1);
        auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
        rtvHandle = descHeap->hCPU(chunk.offset);
        auto viewDesc = rt->GetRenderTargetDesc(cmd->level());
        device->device->CreateRenderTargetView(rt->GetResource(), &viewDesc, rtvHandle);
        cmdList->ClearRenderTargetView(rtvHandle, values, 0, nullptr);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Texture download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::get_copy_texture_buffer_size(
            rt,
            cmd->size());
        auto alloc = bd->get_cb()->get_alloc();
        auto bfView = alloc->get_temp_readback_buffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);

        if (copyInfo.alignedBufferSize == copyInfo.bufferSize) {
            alloc->execute_after_complete(
                [bfView,
                 ptr = cmd->data()] {
                    auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
                    size_t bufferOffset = bfView.offset;
                    rbBuffer->CopyData(
                        bufferOffset,
                        {reinterpret_cast<uint8_t *>(ptr), bfView.byteSize});
                });
        } else {
            auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
            size_t bufferOffset = bfView.offset;
            alloc->execute_after_complete(
                [rbBuffer,
                 bufferOffset,
                 dataPtr = reinterpret_cast<uint8_t *>(cmd->data()),
                 copyInfo]() mutable {
                    while (copyInfo.bufferSize > 0) {

                        rbBuffer->CopyData(
                            bufferOffset,
                            {dataPtr, copyInfo.copySize});
                        dataPtr += copyInfo.copySize;
                        copyInfo.bufferSize -= copyInfo.copySize;
                        bufferOffset += copyInfo.stepSize;
                    }
                });
        }
        bd->copy_buffer_texture(
            bfView,
            rt,
            cmd->offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer,
            false);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Texture copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        bd->copy_texture(
            src,
            0,
            cmd->src_level(),
            dst,
            0,
            cmd->dst_level());
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Texture copy to buffer");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->copy_buffer_texture(
            BufferView{bf, cmd->buffer_offset()},
            rt,
            cmd->texture_offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer,
            true);
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Accel build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        vstd::optional<BufferView> scratch;
        if (!cmd->update_instance_buffer_only()) {
            scratch.create(BufferView(accel_scratch_buffer, accel_scratch_offsets->first, accel_scratch_offsets->second));
            if (accel->RequireCompact()) {
                update_accel->emplace_back(ButtomCompactCmd{
                    .accel = accel,
                    .offset = accel_scratch_offsets->first,
                    .size = accel_scratch_offsets->second});
            }
            accel_scratch_offsets++;
        }
        accel->Build(
            *state_tracker,
            *bd,
            scratch.has_value() ? scratch.ptr() : nullptr);
    }
    void bottom_build(uint64 handle) {
        auto accel = reinterpret_cast<BottomAccel *>(handle);
        accel->UpdateStates(
            *state_tracker,
            *bd,
            BufferView(accel_scratch_buffer, accel_scratch_offsets->first, accel_scratch_offsets->second),
            *bottom_accel_data);
        if (accel->RequireCompact()) {
            update_accel->emplace_back(ButtomCompactCmd{
                .accel = accel,
                .offset = accel_scratch_offsets->first,
                .size = accel_scratch_offsets->second});
        }
        accel_scratch_offsets++;
        bottom_accel_data++;
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
        LUISA_NOT_IMPLEMENTED();
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Mesh build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        bottom_build(cmd->handle());
    }
    void visit(const ProceduralPrimitiveBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Procedural build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        bottom_build(cmd->handle());
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Bindless-array update");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        cmd->visit_modifications([&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, luisa::vector<BindlessArrayUpdateCommand::Texture3DModification>>) {
                arr->UpdateStates(
                    *bd,
                    *state_tracker,
                    luisa::span{
                        reinterpret_cast<BindlessArrayUpdateCommand::Texture2DModification const *>(t.data()),
                        t.size()});
            } else {
                arr->UpdateStates(
                    *bd,
                    *state_tracker,
                    luisa::span{t});
            }
        });
    }
    void visit(const DXCustomCmd *cmd) noexcept {
        cmd->execute(
            device->adapter.Get(),
            device->dxgi_factory.Get(),
            device->device.Get(),
            bd->get_cb()->cmd_list());
        after_custom_cmd(device, bd);
    }
    void visit(const CustomCommand *cmd) noexcept override {
        switch (cmd->custom_cmd_uuid()) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET):
                visit(static_cast<ClearRenderTargetCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<DXCustomCmd const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }
    void visit(const DrawRasterSceneCommand *cmd) noexcept {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->get_cb()->cmd_list(), get_pix_color(), "Draw raster command");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->get_cb()->cmd_list());
        });
#endif
        bind_props->clear();
        auto cmdList = bd->get_cb()->cmd_list();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        DepthFormat dsvFormat{DepthFormat::None};
        // TODO:Set render target
        // Set viewport
        auto alloc = bd->get_cb()->get_alloc();
        {
            D3D12_VIEWPORT view;
            auto &&viewport = cmd->viewport();
            view.MinDepth = 0;
            view.MaxDepth = 1;
            view.TopLeftX = static_cast<FLOAT>(viewport.start.x);
            view.TopLeftY = static_cast<FLOAT>(viewport.start.y);
            view.Width = static_cast<FLOAT>(viewport.size.x);
            view.Height = static_cast<FLOAT>(viewport.size.y);
            cmdList->RSSetViewports(1, &view);
            RECT rect{
                .left = static_cast<int>(view.TopLeftX + 0.4999f),
                .top = static_cast<int>(view.TopLeftY + 0.4999f),
                .right = static_cast<int>(view.TopLeftX + view.Width + 0.4999f),
                .bottom = static_cast<int>(view.TopLeftY + view.Height + 0.4999f)};
            cmdList->RSSetScissorRects(1, &rect);
        }
        GFXFormat rtvFormats[8];
        {

            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE *dsvHandlePtr = nullptr;
            if (!rtvs.empty()) {
                auto chunk = alloc->rtv_allocator.allocate(rtvs.size());
                auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
                rtvHandle = descHeap->hCPU(chunk.offset);
                for (auto i : vstd::range(static_cast<int64>(rtvs.size()))) {
                    auto &&rtv = rtvs[i];
                    auto tex = reinterpret_cast<TextureBase *>(rtv.handle);
                    rtvFormats[i] = tex->Format();
                    D3D12_RENDER_TARGET_VIEW_DESC viewDesc{
                        .Format = static_cast<DXGI_FORMAT>(rtvFormats[i]),
                        .ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D};
                    viewDesc.Texture2D = {
                        .MipSlice = rtv.level,
                        .PlaneSlice = 0};
                    descHeap->CreateRTV(tex->GetResource(), viewDesc, chunk.offset + i);
                }
            }
            if (dsv.handle != ~0ull) {
                dsvHandlePtr = &dsvHandle;
                auto chunk = alloc->dsv_allocator.allocate(1);
                auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
                dsvHandle = descHeap->hCPU(chunk.offset);
                auto tex = reinterpret_cast<TextureBase *>(dsv.handle);
                D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc{
                    .Format = static_cast<DXGI_FORMAT>(tex->Format()),
                    .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
                    .Flags = D3D12_DSV_FLAG_NONE};
                viewDesc.Texture2D.MipSlice = 0;
                device->device->CreateDepthStencilView(tex->GetResource(), &viewDesc, dsvHandle);
                dsvFormat = DepthBuffer::GFXFormatToDepth(tex->Format());
            }
            cmdList->OMSetRenderTargets(rtvs.size(), &rtvHandle, true, dsvHandlePtr);
        }
        auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
        auto rasterState = cmd->raster_state();
        auto pso = shader->get_pso({rtvFormats, rtvs.size()}, cmd->mesh_format(), dsvFormat, rasterState);
        auto &&tempBuffer = *buffer_vec;
        buffer_vec++;
        bind_props->emplace_back(DescriptorHeapView(device->sampler_heap.get()));
        if (tempBuffer.second > 0) {
            bind_props->emplace_back(BufferView(arg_buffer.buffer, arg_buffer.offset + tempBuffer.first, tempBuffer.second));
        }
        auto global_heapView = DescriptorHeapView(device->global_heap.get());
        vstd::push_back_func(*bind_props, shader->bindless_count(), [&] { return global_heapView; });
        decode_cmd(cmd->arguments(), Visitor{this, shader->args().data()});
        bd->set_raster_shader(shader, pso, *bind_props);
        cmdList->IASetPrimitiveTopology([&] {
            switch (rasterState.topology) {
                case TopologyType::Line:
                    return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
                case TopologyType::Point:
                    return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
                default:
                    return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            }
        }());
        auto &&meshes = cmd->scene();
        auto propCount = shader->properties().size();
        for (auto idx : vstd::range(static_cast<int64>(meshes.size()))) {
            auto &&mesh = meshes[idx];
            cmdList->SetGraphicsRoot32BitConstant(propCount, mesh.object_id(), 0);
            vbv->clear();
            auto src = mesh.vertex_buffers();
            vstd::push_back_func(
                *vbv,
                src.size(),
                [&](size_t i) {
                    auto &e = src[i];
                    auto bf = reinterpret_cast<Buffer *>(e.handle());
                    return D3D12_VERTEX_BUFFER_VIEW{
                        .BufferLocation = bf->GetAddress() + e.offset(),
                        .SizeInBytes = static_cast<uint>(e.size()),
                        .StrideInBytes = static_cast<uint>(e.stride())};
                });
            cmdList->IASetVertexBuffers(0, vbv->size(), vbv->data());
            auto const &i = mesh.index();

            luisa::visit(
                [&]<typename T>(T const &i) {
                    if constexpr (std::is_same_v<T, uint>) {
                        cmdList->DrawInstanced(i, mesh.instance_count(), mesh.vertex_offset(), 0);
                    } else {
                        auto bf = reinterpret_cast<Buffer *>(i.handle());
                        D3D12_INDEX_BUFFER_VIEW idx{
                            .BufferLocation = bf->GetAddress() + i.offset_bytes(),
                            .SizeInBytes = static_cast<uint>(i.size_bytes()),
                            .Format = DXGI_FORMAT_R32_UINT};
                        cmdList->IASetIndexBuffer(&idx);
                        cmdList->DrawIndexedInstanced(i.size_bytes() / sizeof(uint), mesh.instance_count(), 0, mesh.vertex_offset(), 0);
                    }
                },
                i);
        }
    }
};

LCCmdBuffer::LCCmdBuffer(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : CmdQueueBase{device, CmdQueueTag::MainCmd},
      reorder({}),
      queue(
          device,
          resourceAllocator,
          type) {
}
void LCCmdBuffer::Execute(
    vstd::span<const luisa::unique_ptr<Command>> commands,
    luisa::vector<luisa::move_only_function<void()>> &&funcs,
    vstd::span<const SwapchainPresent> presents,
    size_t max_alloc) {
    auto allocator = queue.create_allocator(max_alloc);
    auto alloc_type = allocator->type();
    bool cmd_list_is_empty = commands.empty();
    luisa::fixed_vector<std::pair<IDXGISwapChain *, bool>, 4> present_swapchains;
    present_swapchains.reserve(presents.size());
    {
        std::unique_lock lck{mtx};
        LCPreProcessVisitor pp_visitor;
        pp_visitor.arg_vecs = &argVecs;
        pp_visitor.arg_buffer = &argBuffer;
        pp_visitor.bottom_accel_datas = &bottomAccelDatas;
        pp_visitor.accel_offset = &accelOffset;
        argVecs.clear();
        argBuffer.clear();
        bottomAccelDatas.clear();
        accelOffset.clear();

        LCCmdVisitor visitor;
        if (log_callback) {
            visitor.logger = &log_callback;
        } else {
            visitor.logger = nullptr;
        }
        visitor.bind_props = &bindProps;
        visitor.update_accel = &updateAccel;
        visitor.vbv = &vbv;
        visitor.device = device;
        visitor.after_custom_cmd = [](Device *device, CommandBufferBuilder *bd) {
            ID3D12DescriptorHeap *h[2] = {
                device->global_heap->GetHeap(),
                device->sampler_heap->GetHeap()};
            auto cb = bd->get_cb();
            if (cb->get_alloc()->type() != D3D12_COMMAND_LIST_TYPE_COPY) {
                cb->cmd_list()->SetDescriptorHeaps(vstd::array_count(h), h);
            }
        };
        auto cmd_buffer = allocator->get_buffer();
        auto cmd_builder = cmd_buffer->build();
        GraphicsCmdlistBarrierCallback barrier_callback(cmd_builder);
        if (!tracker) {
            if (device->feature_check.enhanced_barriers_supported()) {
                tracker = luisa::make_unique<EnhancedBarrierTrackerImpl>();
            } else {
                tracker = luisa::make_unique<EnhancedBarrierTrackerBackup>();
            }
        }
        tracker->listType = allocator->type();
        visitor.state_tracker = tracker.get();
        pp_visitor.state_tracker = tracker.get();
        visitor.bd = &cmd_builder;
        pp_visitor.bd = &cmd_builder;
        reorder.clear();
        size_t uniform_size = 0;
        auto add_size = [&](auto const &c, Argument const &a) {
            if (a.tag != Argument::Tag::UNIFORM) [[likely]]
                return;

            // uniform_buffer_size +=
            uniform_size = CalcAlign(uniform_size, a.uniform.alignment);
            auto bf = c.uniform(a.uniform);
            uniform_size += std::max<size_t>(4, bf.size_bytes());
        };
        for (auto &&command : commands) {
            // if (command->tag() == Command::Tag::EBindlessArrayUpdateCommand) {
            //     auto cmd = static_cast<BindlessArrayUpdateCommand const *>(command.get());
            //     reinterpret_cast<BindlessArray *>(cmd->handle())->Bind(cmd->modifications());
            // }
            command->accept(reorder);
            switch (command->tag()) {
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(command.get());
                    auto cs = reinterpret_cast<ComputeShader *>(c->handle());
                    uniform_size = CalcAlign(uniform_size, 32);
                    for (auto &&i : cs->arg_bindings()) {
                        add_size(*c, i);
                    }
                    for (auto &&i : c->arguments()) {
                        add_size(*c, i);
                    }
                    // Account for validation data (_validate_* fields) in cbuffer
                    if (cs->validation_count() > 0) {
                        uniform_size += cs->validation_count() * sizeof(uint);
                    }
                } break;
                case Command::Tag::ECustomCommand: {
                    if (static_cast<CustomCommand const *>(command.get())->custom_cmd_uuid() ==
                        to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE)) {
                        auto c = static_cast<DrawRasterSceneCommand const *>(command.get());
                        auto rs = reinterpret_cast<RasterShader *>(c->handle());
                        uniform_size = CalcAlign(uniform_size, 32);
                        for (auto &&i : c->arguments()) {
                            add_size(*c, i);
                        }
                        if (rs->validation_count() > 0) {
                            uniform_size += rs->validation_count() * sizeof(uint);
                        }
                    }
                } break;
            }
        }
        // Upload CBuffers
        if (uniform_size > 0) {
            visitor.arg_buffer = allocator->get_temp_upload_buffer(uniform_size, 32);
        } else {
            visitor.arg_buffer = {};
        }
        auto cmdLists = reorder.command_lists();
        ID3D12DescriptorHeap *h[2] = {
            device->global_heap->GetHeap(),
            device->sampler_heap->GetHeap()};
        tracker->restoreStates.clear();
        if (device->device_settings) {
            auto after_states = device->device_settings->after_states(reinterpret_cast<uint64_t>(this));
            auto before_states = device->device_settings->before_states(reinterpret_cast<uint64_t>(this));
            for (auto &i : before_states) {
                tracker->SetRes(get_resource_view(i.resource),
                                i.sync, i.access, i.texture_layout);
            }
            for (auto &i : after_states) {
                tracker->restoreStates.emplace(
                    reinterpret_cast<Resource const *>(luisa::visit([](auto &&t) { return t.handle; }, i.resource)),
                    EnhancedBarrierTracker::ResotreStates{
                        get_resource_view(i.resource),
                        i.sync,
                        i.access,
                        i.texture_layout});
            }
        }
        // for (auto &&command : commands) {
        for (auto &&lst : cmdLists) {
            if (alloc_type != D3D12_COMMAND_LIST_TYPE_COPY) {
                cmd_buffer->cmd_list()->SetDescriptorHeaps(vstd::array_count(h), h);
            }

            // Clear caches
            pp_visitor.arg_vecs->clear();
            pp_visitor.accel_offset->clear();
            pp_visitor.bottom_accel_datas->clear();
            pp_visitor.build_accel_size = 0;
            // Preprocess: record resources' states
            for (auto i = lst; i != nullptr; i = i->p_next) {
                if (i->cmd)
                    i->cmd->accept(pp_visitor);
            }
            // command->accept(pp_visitor);
            visitor.bottom_accel_data = pp_visitor.bottom_accel_datas->data();
            DefaultBuffer const *accelScratchBuffer;
            if (pp_visitor.build_accel_size) {
                accelScratchBuffer = allocator->allocate_scratch_buffer(pp_visitor.build_accel_size);
                visitor.accel_scratch_offsets = pp_visitor.accel_offset->data();
                visitor.accel_scratch_buffer = accelScratchBuffer;
                tracker->Record(accelScratchBuffer, EnhancedBarrierTracker::Usage::BuildAccelScratch);
            }

            tracker->UpdateState(
                &barrier_callback);
            visitor.buffer_vec = pp_visitor.arg_vecs->data();
            // Execute commands
            for (auto i = lst; i != nullptr; i = i->p_next) {
                i->cmd->accept(visitor);
            }
            // command->accept(visitor);

            if (!updateAccel.empty()) {
                tracker->Record(
                    BufferView(accelScratchBuffer),
                    EnhancedBarrierTracker::Usage::CopySource);
                tracker->UpdateState(&barrier_callback);
                for (auto &&i : updateAccel) {
                    i.accel.visit([&](auto &&p) {
                        p->FinalCopy(
                            cmd_builder,
                            BufferView(
                                accelScratchBuffer,
                                i.offset,
                                i.size));
                    });
                }
                tracker->RestoreState(&barrier_callback);
                auto local_update_accel = std::move(updateAccel);
                lck.unlock();
                queue.force_sync(
                    allocator,
                    *cmd_buffer);
                for (auto &&i : local_update_accel) {
                    i.accel.visit([&](auto &&p) {
                        p->CheckAccel(cmd_builder);
                    });
                }
                lck.lock();
            }
        }
        if (visitor.arg_buffer.buffer) {
            static_cast<UploadBuffer const *>(visitor.arg_buffer.buffer)
                ->CopyData(
                    visitor.arg_buffer.offset,
                    {reinterpret_cast<uint8_t const *>(
                         argBuffer.data()),
                     argBuffer.size()});
            auto aligned_size = CalcAlign(argBuffer.size(), 32);
            uniform_size = CalcAlign(uniform_size, 32);
            LUISA_DEBUG_ASSERT(aligned_size == uniform_size);
        }

        for (auto &&present : presents) {
            auto swapchain = reinterpret_cast<LCSwapChain *>(present.chain->handle());
            auto &&rt = &swapchain->render_targets[swapchain->frame_index];
            auto img = reinterpret_cast<TextureBase *>(present.frame.handle());
            auto mip = present.frame.level();
            tracker->Record(
                rt,
                EnhancedBarrierTracker::Range(0, 1),
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_LAYOUT_COPY_DEST);
            tracker->Record(
                img,
                EnhancedBarrierTracker::Range(mip, 1),
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE);
            // tracker->UpdateState(&barrier_callback);
        }
        if (!presents.empty()) {
            tracker->UpdateState(&barrier_callback);
        }
        for (auto &&present : presents) {
            auto swapchain = reinterpret_cast<LCSwapChain *>(present.chain->handle());
            auto &&rt = &swapchain->render_targets[swapchain->frame_index];
            auto img = reinterpret_cast<TextureBase *>(present.frame.handle());
            auto mip = present.frame.level();
            swapchain->frame_index += 1;
            swapchain->frame_index %= swapchain->frame_count;
            D3D12_TEXTURE_COPY_LOCATION sourceLocation;
            sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            sourceLocation.SubresourceIndex = mip;
            D3D12_TEXTURE_COPY_LOCATION destLocation;
            destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destLocation.SubresourceIndex = 0;
            sourceLocation.pResource = img->GetResource();
            destLocation.pResource = rt->GetResource();
            cmd_buffer->cmd_list()->CopyTextureRegion(
                &destLocation,
                0, 0, 0,
                &sourceLocation,
                nullptr);
            tracker->Record(
                rt,
                EnhancedBarrierTracker::Range(0, 1),
                D3D12_BARRIER_SYNC_ALL,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_LAYOUT_PRESENT);
            present_swapchains.emplace_back(swapchain->swap_chain.Get(), swapchain->vsync);
        }
        tracker->RestoreState(&barrier_callback);
    }
    queue.execute(std::move(allocator), std::move(funcs), luisa::span{present_swapchains}, cmd_list_is_empty);
}
void LCCmdBuffer::Sync() {
    queue.complete();
}
void LCCmdBuffer::Present(
    LCSwapChain *swapchain,
    TextureBase *img,
    uint mip,
    size_t max_alloc) {
    auto alloc = queue.create_allocator(max_alloc);
    {
        std::lock_guard lck{mtx};
        // swapchain->frame_index = swapchain->swap_chain->GetCurrentBackBufferIndex();
        auto &&rt = &swapchain->render_targets[swapchain->frame_index];
        swapchain->frame_index += 1;
        swapchain->frame_index %= swapchain->frame_count;
        auto cb = alloc->get_buffer();
        auto bd = cb->build();
        auto cmdList = cb->cmd_list();
        {
            D3D12_RESOURCE_BARRIER barriers[2];
            D3D12_RESOURCE_BARRIER &img_barrier = barriers[0];
            D3D12_RESOURCE_BARRIER &rt_barrier = barriers[1];

            rt_barrier = D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE};
            img_barrier = D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE};
            rt_barrier.Transition.pResource = rt->GetResource();
            rt_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            rt_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            rt_barrier.Transition.Subresource = 0;
            img_barrier.Transition.pResource = img->GetResource();
            img_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            img_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            img_barrier.Transition.Subresource = mip;
            bd.get_cb()->cmd_list()->ResourceBarrier(vstd::array_count(barriers), barriers);
        }
        // tracker->Record(
        //     EnhancedBarrierTracker::ResourceView(rt), EnhancedBarrierTracker::Usage::CopyDest);
        // tracker->Record(
        //     EnhancedBarrierTracker::ResourceView(img), EnhancedBarrierTracker::Usage::CopySource);
        // tracker->UpdateState(bd);
        D3D12_TEXTURE_COPY_LOCATION sourceLocation;
        sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        sourceLocation.SubresourceIndex = mip;
        D3D12_TEXTURE_COPY_LOCATION destLocation;
        destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destLocation.SubresourceIndex = 0;
        sourceLocation.pResource = img->GetResource();
        destLocation.pResource = rt->GetResource();
        cmdList->CopyTextureRegion(
            &destLocation,
            0, 0, 0,
            &sourceLocation,
            nullptr);
        {
            D3D12_RESOURCE_BARRIER barriers[2];
            D3D12_RESOURCE_BARRIER &img_barrier = barriers[0];
            D3D12_RESOURCE_BARRIER &rt_barrier = barriers[1];

            rt_barrier = D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE};
            img_barrier = D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE};
            rt_barrier.Transition.pResource = rt->GetResource();
            rt_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            rt_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            rt_barrier.Transition.Subresource = 0;
            img_barrier.Transition.pResource = img->GetResource();
            img_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            img_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
            img_barrier.Transition.Subresource = mip;
            bd.get_cb()->cmd_list()->ResourceBarrier(vstd::array_count(barriers), barriers);
        }
    }
    std::pair<IDXGISwapChain *, bool> sw{swapchain->swap_chain.Get(), swapchain->vsync};
    queue.execute(std::move(alloc), {}, {&sw, 1}, false);
}
void LCCmdBuffer::CompressBC(
    TextureBase *rt,
    uint level,
    luisa::compute::BufferView<uint> const &result,
    bool is_hdr,
    float alpha_importance,
    GpuAllocator *allocator,
    size_t max_alloc) {
    if (!tracker) {
        if (device->feature_check.enhanced_barriers_supported()) {
            tracker = luisa::make_unique<EnhancedBarrierTrackerImpl>();
        } else {
            tracker = luisa::make_unique<EnhancedBarrierTrackerBackup>();
        }
    }
    alpha_importance = std::max<float>(std::min<float>(alpha_importance, 1), 0);// clamp<float>(alpha_importance, 0, 1);
    struct BCCBuffer {
        uint g_mip_level;
        uint g_tex_width;
        uint g_num_block_x;
        uint g_format;
        uint g_mode_id;
        uint g_start_block_id;
        uint g_num_total_blocks;
        float g_alpha_weight;
    };
    uint width = rt->Width() >> level;
    uint height = rt->Height() >> level;
    uint x_blocks = std::max<uint>(1, (width + 3) >> 2);
    uint y_blocks = std::max<uint>(1, (height + 3) >> 2);
    uint num_blocks = x_blocks * y_blocks;
    uint num_total_blocks = num_blocks;
    static constexpr size_t BLOCK_SIZE = 16;
    if (result.size_bytes() != BLOCK_SIZE * num_blocks) [[unlikely]] {
        LUISA_ERROR("Texture compress output buffer incorrect size!");
    }
    DefaultBuffer back_buffer(
        device,
        BLOCK_SIZE * num_blocks,
        allocator,
        D3D12_RESOURCE_STATE_COMMON);
    auto out_buffer_ptr = reinterpret_cast<Buffer *>(result.handle());
    BufferView outBuffer{
        out_buffer_ptr,
        result.offset_bytes(),
        result.size_bytes()};

    constexpr uint MAX_BATCH = 1024 * 1024;
    auto batch_num = static_cast<int>((num_total_blocks + MAX_BATCH - 1) / MAX_BATCH);
    uint start_block_id = 0;
    for (int batch = 0; batch < batch_num; batch++) {
        auto target = static_cast<int>((batch + 1) * MAX_BATCH);
        auto alloc = queue.create_allocator(max_alloc);
        {
            std::lock_guard lck{mtx};
            tracker->listType = alloc->type();
            // auto bufferReadState = tracker->ReadState(ResourceReadUsage::Srv);
            auto cmd_buffer = alloc->get_buffer();
            auto cmd_builder = cmd_buffer->build();
            GraphicsCmdlistBarrierCallback barrier_callback(cmd_builder);
            ID3D12DescriptorHeap *h[2] = {
                device->global_heap->GetHeap(),
                device->sampler_heap->GetHeap()};
            cmd_buffer->cmd_list()->SetDescriptorHeaps(vstd::array_count(h), h);

            BCCBuffer cb_data{
                .g_mip_level = level};
            tracker->Record(
                EnhancedBarrierTracker::TexView(rt, level),
                EnhancedBarrierTracker::Usage::ComputeRead);
            auto run_compute_shader = [&](ComputeShader const *cs, uint dispatch_count, BufferView const &in_buffer, BufferView const &outBuffer) {
                auto cbuffer = alloc->get_temp_upload_buffer(sizeof(BCCBuffer), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
                static_cast<UploadBuffer const *>(cbuffer.buffer)->CopyData(cbuffer.offset, {reinterpret_cast<uint8_t const *>(&cb_data), sizeof(BCCBuffer)});
                tracker->Record(
                    in_buffer,
                    EnhancedBarrierTracker::Usage::ComputeRead);
                tracker->Record(
                    outBuffer,
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                tracker->UpdateState(&barrier_callback);
                BindProperty prop[4];
                prop[0] = cbuffer;
                prop[1] = DescriptorHeapView(device->global_heap.get(), rt->GetGlobalSRVIndex());
                prop[2] = in_buffer;
                prop[3] = outBuffer;
                cmd_builder.dispatch_compute(
                    cs,
                    uint3(dispatch_count, 1, 1),
                    {prop, 4});
            };
            constexpr uint MAX_BLOCK_BATCH = 1024u * 32u;
            if (is_hdr)//bc6
            {
                BufferView err1_buffer{&back_buffer};
                BufferView err2_buffer{outBuffer};
                auto bc6_try_mode_g10 = device->bc6_try_mode_g10.get(device);
                auto bc6_try_mode_le10 = device->bc6_try_mode_le10.get(device);
                auto bc6Encode = device->bc6_encode_block.get(device);
                while (num_blocks > 0 && start_block_id < target) {
                    uint n = std::min<uint>(num_blocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cb_data.g_tex_width = width;
                    cb_data.g_num_block_x = x_blocks;
                    cb_data.g_format = is_hdr ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cb_data.g_start_block_id = start_block_id;
                    cb_data.g_alpha_weight = alpha_importance;
                    cb_data.g_num_total_blocks = num_total_blocks;
                    run_compute_shader(
                        bc6_try_mode_g10,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2_buffer,
                        err1_buffer);
                    for (auto i : vstd::range(10)) {
                        cb_data.g_mode_id = i;
                        run_compute_shader(
                            bc6_try_mode_le10,
                            std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                            ((i & 1) != 0) ? err2_buffer : err1_buffer,
                            ((i & 1) != 0) ? err1_buffer : err2_buffer);
                    }
                    run_compute_shader(
                        bc6Encode,
                        std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                        err1_buffer,
                        err2_buffer);
                    start_block_id += n;
                    num_blocks -= n;
                }

            } else {
                BufferView err1_buffer{outBuffer};
                BufferView err2_buffer{&back_buffer};
                auto bc7Try137Mode = device->bc7_try_mode_137.get(device);
                auto bc7Try02Mode = device->bc7_try_mode_02.get(device);
                auto bc7Try456Mode = device->bc7_try_mode_456.get(device);
                auto bc7Encode = device->bc7_encode_block.get(device);
                while (num_blocks > 0 && start_block_id < target) {
                    uint n = std::min<uint>(num_blocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cb_data.g_tex_width = width;
                    cb_data.g_num_block_x = x_blocks;
                    cb_data.g_format = is_hdr ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cb_data.g_start_block_id = start_block_id;
                    cb_data.g_alpha_weight = alpha_importance;
                    cb_data.g_num_total_blocks = num_total_blocks;
                    run_compute_shader(bc7Try456Mode, std::max<uint>((uThreadGroupCount + 3) / 4, 1), err2_buffer, err1_buffer);
                    //137
                    {
                        uint modes[] = {1, 3, 7};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cb_data.g_mode_id = modes[i];
                            run_compute_shader(
                                bc7Try137Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err2_buffer : err1_buffer,
                                ((i & 1) != 0) ? err1_buffer : err2_buffer);
                        }
                    }
                    //02
                    {
                        uint modes[] = {0, 2};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cb_data.g_mode_id = modes[i];
                            run_compute_shader(
                                bc7Try02Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err1_buffer : err2_buffer,
                                ((i & 1) != 0) ? err2_buffer : err1_buffer);
                        }
                    }
                    run_compute_shader(
                        bc7Encode,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2_buffer,
                        err1_buffer);
                    //TODO
                    start_block_id += n;
                    num_blocks -= n;
                }
            }
            tracker->Record(outBuffer, EnhancedBarrierTracker::Usage::CopySource);
            tracker->RestoreState(&barrier_callback);
        }
        if (batch == batch_num - 1) {
            vstd::vector<vstd::function<void()>> callbacks;
            callbacks.emplace_back([back_buffer = std::move(back_buffer)] {});
            queue.execute(
                std::move(alloc),
                std::move(callbacks), {}, false);
        } else {
            queue.execute(std::move(alloc), {}, {}, false);
        }
    }
}
}// namespace lc::dx
