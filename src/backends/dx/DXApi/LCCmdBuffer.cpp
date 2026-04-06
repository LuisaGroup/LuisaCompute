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
#include <luisa/backends/ext/work_graph_cmd.h>
#include <Shader/WorkGraphProgram.h>
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
    : Resource{device}, tag{tag},
      logCallback([](luisa::string_view str) {
          LUISA_INFO("[DEVICE] {}", str);
      }) {}
using Argument = luisa::compute::Argument;
static bool is_device_buffer(Resource const *res) {
    auto tag = res->GetTag();
    return (tag != Resource::Tag::UploadBuffer) && (tag != Resource::Tag::ReadbackBuffer);
}
template<typename Visitor>
void DecodeCmd(vstd::span<const Argument> args, Visitor &&visitor) {
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
    EnhancedBarrierTracker *stateTracker{};
    vstd::vector<std::pair<size_t, size_t>> *argVecs{};
    vstd::vector<uint8_t> *argBuffer{};
    vstd::vector<BottomAccelData> *bottomAccelDatas{};
    vstd::fixed_vector<std::pair<size_t, size_t>, 4> *accelOffset{};
    size_t buildAccelSize = 0;
    void AddBuildAccel(size_t size) {
        size = CalcAlign(size, 256);
        accelOffset->emplace_back(buildAccelSize, size);
        buildAccelSize += size;
    }
    void UniformAlign(size_t align) const {
        luisa::vector_resize(*argBuffer, CalcAlign(argBuffer->size(), align));
    }
    template<typename T>
    void EmplaceData(T const &data, size_t alignment) {
        size_t sz = argBuffer->size();
        alignment -= 1;
        auto aligned_size = (sz + alignment) & (~alignment);
        luisa::enlarge_by(*argBuffer, sizeof(T) + aligned_size - sz);
        using PlaceHolder = luisa::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(argBuffer->data() + aligned_size) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void EmplaceData(T const *data, size_t size, size_t alignment) {
        alignment -= 1;
        size_t sz = argBuffer->size();
        auto aligned_size = (sz + alignment) & (~alignment);
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(*argBuffer, byteSize + aligned_size - sz);
        std::memcpy(argBuffer->data() + aligned_size, data, byteSize);
    }
    struct Visitor {
        LCPreProcessVisitor *self;
        SavedArgument const *arg;
        ShaderDispatchCommandBase const &cmd;
        EnhancedBarrierTracker::Usage uav_usage;
        EnhancedBarrierTracker::Usage read_usage;
        EnhancedBarrierTracker::Usage accel_read_usage;
        Visitor(
            LCPreProcessVisitor *self,
            SavedArgument const *arg,
            ShaderDispatchCommandBase const &cmd,
            bool is_raster) : self(self), arg(arg), cmd(cmd) {
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
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
                self->stateTracker->Record(
                    BufferView{res, bf.offset, bf.size},
                    uav_usage);
                // self->stateTracker->RecordState(
                //     res,
                //     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                //     true);
            } else {
                if (is_device_buffer(res))
                    self->stateTracker->Record(
                        BufferView{res, bf.offset, bf.size},
                        read_usage);
                else {
                    LUISA_ASSERT(res->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as shader's resource.");
                }
            }
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                if (!rt->AllowUAV()) [[unlikely]] {
                    LUISA_ERROR("Texture not allowed for Unordered-Access.");
                }
                self->stateTracker->Record(
                    EnhancedBarrierTracker::TexView{rt, bf.level},
                    uav_usage);
            }
            // SRV
            else {
                self->stateTracker->Record(
                    EnhancedBarrierTracker::TexView{rt, bf.level},
                    read_usage);
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            vstd::fixed_vector<vstd::HashMap<Resource const *, size_t>::Index, 16> writeMap;
            auto &write_state_map = self->stateTracker->WriteStateMap();
            arr->Lock();
            for (auto iter = write_state_map.begin(); iter != write_state_map.end(); ++iter) {
                auto &i = *iter;
                if (arr->IsPtrInBindless(reinterpret_cast<size_t>(i.first))) {
                    writeMap.emplace_back(write_state_map.get_index(iter));
                }
            }
            arr->Unlock();

            for (auto &&iter : writeMap) {
                self->stateTracker->Record(
                    iter.key(),
                    EnhancedBarrierTracker::Range(0, iter.value()),
                    read_usage);
                write_state_map.remove(iter);
            }
            self->stateTracker->Record(
                BufferView(arr->BindlessBuffer()),
                read_usage);
            ++arg;
        }
        void operator()(Argument::Uniform const &a) {
            auto bf = cmd.uniform(a);
            self->EmplaceData(bf.data(), bf.size_bytes(), a.alignment);
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if (accel->GetInstBuffer()) [[likely]] {
                if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                    self->stateTracker->Record(
                        BufferView{accel->GetInstBuffer(), 0, accel->GetInstBuffer()->GetByteSize()},
                        uav_usage);
                } else {
                    self->stateTracker->Record(
                        BufferView{accel->GetInstBuffer(), 0, accel->GetInstBuffer()->GetByteSize()},
                        read_usage);
                    auto accelBuffer = accel->GetAccelBuffer();
                    self->stateTracker->Record(
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
            stateTracker->Record(res_view, i.required_state);
        }
        for (auto i : const_cast<DXCustomCmd *>(cmd)->get_enhanced_resource_usages()) {
            auto res_view = get_resource_view(i.resource);
            stateTracker->Record(
                res_view,
                i.sync,
                i.access,
                i.texture_layout);
        }
    }
    void visit(const BufferUploadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res)) {
            stateTracker->Record(
                BufferView(res, cmd->offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopyDest);
            // stateTracker->RecordState(res, D3D12_RESOURCE_STATE_COPY_DEST);
        } else {
            LUISA_ERROR("Host-buffer should not be used to upload.");
        }
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res)) {
            stateTracker->Record(
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
            stateTracker->Record(
                BufferView(srcBf, cmd->src_offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopySource);
        } else {
            LUISA_ASSERT(srcBf->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
        if (is_device_buffer(dstBf)) {
            stateTracker->Record(
                BufferView(dstBf, cmd->dst_offset(), cmd->size()),
                EnhancedBarrierTracker::Usage::CopyDest);
        } else {
            LUISA_ASSERT(dstBf->GetTag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopyDest);
        if (is_device_buffer(bf)) {
            stateTracker->Record(
                BufferView(bf, cmd->buffer_offset(), pixel_storage_size(cmd->storage(), cmd->size())),
                EnhancedBarrierTracker::Usage::CopySource);
        } else {
            LUISA_ASSERT(bf->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
    }

    void visit(const TextureUploadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopyDest);
    }
    void visit(const ClearDepthCommand *cmd) noexcept {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, 0),
            EnhancedBarrierTracker::Usage::DepthWrite);
    }
    void visit(const ClearRenderTargetCommand *cmd) noexcept {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::RenderTarget);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopySource);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(src, cmd->src_level()),
            EnhancedBarrierTracker::Usage::CopySource);
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(dst, cmd->dst_level()),
            EnhancedBarrierTracker::Usage::CopyDest);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->Record(
            EnhancedBarrierTracker::TexView(rt, cmd->level()),
            EnhancedBarrierTracker::Usage::CopySource);
        if (is_device_buffer(bf)) {
            stateTracker->Record(
                BufferView(bf, cmd->buffer_offset(), pixel_storage_size(cmd->storage(), cmd->size())),
                EnhancedBarrierTracker::Usage::CopyDest);
        } else {
            LUISA_ASSERT(bf->GetTag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        auto cs = reinterpret_cast<ComputeShader *>(cmd->handle());
        UniformAlign(32);
        size_t beforeSize = argBuffer->size();
        Visitor visitor{this, cs->Args().data(), *cmd, false};
        DecodeCmd(cs->ArgBindings(), visitor);
        DecodeCmd(cmd->arguments(), visitor);
        size_t afterSize = argBuffer->size();
        argVecs->emplace_back(beforeSize, afterSize - beforeSize);
        if (cmd->is_indirect()) {
            auto buffer = reinterpret_cast<Buffer *>(cmd->indirect_dispatch().handle);
            stateTracker->Record(
                BufferView(buffer, cmd->indirect_dispatch().offset / ComputeShader::DispatchIndirectStride, cmd->indirect_dispatch().max_dispatch_size / ComputeShader::DispatchIndirectStride), EnhancedBarrierTracker::Usage::IndirectArgs);
        }
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        if (!cmd->update_instance_buffer_only()) {
            AddBuildAccel(
                accel->PreProcess(
                    *stateTracker,
                    *bd,
                    cmd->instance_count(),
                    cmd->modifications(),
                    cmd->request() == AccelBuildRequest::PREFER_UPDATE));
        } else {
            accel->PreProcessInst(
                *stateTracker,
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
        AddBuildAccel(
            accel->PreProcessStates(
                *bd,
                *stateTracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                meshOptions,
                bottomAccelDatas->emplace_back()));
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
        AddBuildAccel(
            accel->PreProcessStates(
                *bd,
                *stateTracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                aabbOptions,
                bottomAccelDatas->emplace_back()));
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        // reinterpret_cast<BindlessArray *>(cmd->handle())->Bind(cmd->modifications());
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        if (!cmd->empty())
            arr->PreProcessStates(
                *bd,
                *stateTracker);
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
            case to_underlying(CustomCommandUUID::WORK_GRAPH_DISPATCH):
                visit(static_cast<WorkGraphDispatchCommand const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }

    void visit(const WorkGraphDispatchCommand *cmd) noexcept {
        auto program = reinterpret_cast<WorkGraphProgram const *>(cmd->handle());
        auto savedArg = program->Args().data();
        struct WGBarrierVisitor {
            LCPreProcessVisitor *self;
            SavedArgument const *&arg;
            void operator()(Argument::Buffer const &bf) {
                auto res = reinterpret_cast<Buffer const *>(bf.handle);
                if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                    LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
                    self->stateTracker->Record(BufferView{res, bf.offset, bf.size}, EnhancedBarrierTracker::Usage::ComputeUAV);
                } else if (is_device_buffer(res)) {
                    self->stateTracker->Record(BufferView{res, bf.offset, bf.size}, EnhancedBarrierTracker::Usage::ComputeRead);
                }
                ++arg;
            }
            void operator()(Argument::Texture const &bf) {
                auto rt = reinterpret_cast<TextureBase *>(bf.handle);
                if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                    self->stateTracker->Record(EnhancedBarrierTracker::TexView{rt, bf.level}, EnhancedBarrierTracker::Usage::ComputeUAV);
                } else {
                    self->stateTracker->Record(EnhancedBarrierTracker::TexView{rt, bf.level}, EnhancedBarrierTracker::Usage::ComputeRead);
                }
                ++arg;
            }
            void operator()(Argument::Uniform const &) {}
            void operator()(Argument::BindlessArray const &) {}
            void operator()(Argument::Accel const &) {}
        };
        DecodeCmd(program->ArgBindings(), WGBarrierVisitor{this, savedArg});
    }

    void visit(const DrawRasterSceneCommand *cmd) noexcept {
        auto cs = reinterpret_cast<RasterShader *>(cmd->handle());
        UniformAlign(32);
        size_t beforeSize = argBuffer->size();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        DecodeCmd(cmd->arguments(), Visitor{this, cs->Args().data(), *cmd, true});
        size_t afterSize = argBuffer->size();
        argVecs->emplace_back(beforeSize, afterSize - beforeSize);

        for (auto &&mesh : cmd->scene()) {
            for (auto &&v : mesh.vertex_buffers()) {
                stateTracker->Record(
                    BufferView(reinterpret_cast<Buffer *>(v.handle()), v.offset(), v.size()),
                    EnhancedBarrierTracker::Usage::VertexRead);
            }
            auto &&i = mesh.index();
            if (i.index() == 0) {
                auto &&idx = luisa::get<0>(i);
                stateTracker->Record(
                    BufferView(reinterpret_cast<Buffer *>(idx.handle()), idx.offset_bytes(), idx.size_bytes()),
                    EnhancedBarrierTracker::Usage::IndexRead);
            }
        }
        for (auto &&i : rtvs) {
            stateTracker->Record(
                EnhancedBarrierTracker::TexView(
                    reinterpret_cast<TextureBase *>(i.handle),
                    i.level),
                EnhancedBarrierTracker::Usage::RenderTarget);
        }
        if (dsv.handle != ~0ull) {
            stateTracker->Record(
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
    EnhancedBarrierTracker *stateTracker{};
    BufferView argBuffer{};
    Buffer const *accelScratchBuffer{};
    std::pair<size_t, size_t> *accelScratchOffsets{};
    std::pair<size_t, size_t> *bufferVec{};
    vstd::vector<BindProperty> *bindProps{};
    vstd::vector<ButtomCompactCmd> *updateAccel{};
    vstd::vector<D3D12_VERTEX_BUFFER_VIEW> *vbv{};
    BottomAccelData *bottomAccelData{};
    vstd::func_ptr_t<void(Device *, CommandBufferBuilder *)>
        after_custom_cmd{};

    void visit(const BufferUploadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Upload(bf, cmd->data());
    }

    void visit(const BufferDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Readback(
            bf,
            cmd->data());
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        bd->CopyBuffer(
            srcBf,
            dstBf,
            cmd->src_offset(),
            cmd->dst_offset(),
            cmd->size());
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer copy to texture");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
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

            self->bindProps->emplace_back(
                BufferView(res, bf.offset));
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                self->bindProps->emplace_back(
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalUAVIndex(bf.level)));
            }
            // SRV
            else {
                self->bindProps->emplace_back(
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalSRVIndex(bf.level)));
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            auto res = arr->BindlessBuffer();
            self->bindProps->emplace_back(
                BufferView(res, 0));
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if ((static_cast<uint>(arg->varUsage) & static_cast<uint>(Usage::WRITE)) == 0) {
                self->bindProps->emplace_back(
                    accel);
            }
            self->bindProps->emplace_back(
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
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Shader dispatch");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        bindProps->clear();
        auto shader = reinterpret_cast<ComputeShader const *>(cmd->handle());
        auto &&tempBuffer = *bufferVec;
        bufferVec++;
        auto cs = static_cast<ComputeShader const *>(shader);
        BufferView readback_count_buffer;
        BufferView readback_buffer;
        BufferView count_buffer;
        BufferView data_buffer;
        CommandAllocator *alloc{nullptr};
        auto BeforeDispatch = [&]() {
            bindProps->emplace_back(DescriptorHeapView(device->samplerHeap.get()));
            if (tempBuffer.second > 0) {
                bindProps->emplace_back(BufferView(argBuffer.buffer, argBuffer.offset + tempBuffer.first, tempBuffer.second));
            }
            DescriptorHeapView globalHeapView(device->globalHeap.get());
            vstd::push_back_func(*bindProps, shader->BindlessCount(), [&] { return globalHeapView; });
            Visitor visitor{this, cs->Args().data()};
            DecodeCmd(shader->ArgBindings(), visitor);
            DecodeCmd(cmd->arguments(), visitor);
            auto printers = shader->Printers();
            if (!printers.empty()) [[unlikely]] {
                alloc = bd->GetCB()->GetAlloc();
                static const uint zero = 0;
                auto upload_buffer = alloc->GetTempUploadBuffer(sizeof(uint), 16);
                count_buffer = alloc->GetTempDefaultBuffer(sizeof(uint), 16);
                readback_count_buffer = alloc->GetTempReadbackBuffer(sizeof(uint), 16);
                data_buffer = alloc->GetTempDefaultBuffer(1024ull * 1024ull, 16);
                readback_buffer = alloc->GetTempReadbackBuffer(1024ull * 1024ull, 16);
                static_cast<UploadBuffer const *>(upload_buffer.buffer)->CopyData(upload_buffer.offset, {reinterpret_cast<uint8_t const *>(&zero), sizeof(uint)});
                stateTracker->Record(
                    BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::CopyDest);
                stateTracker->UpdateState(&barrier_callback);
                bd->CopyBuffer(upload_buffer.buffer, count_buffer.buffer, upload_buffer.offset, count_buffer.offset, sizeof(uint));
                stateTracker->Record(
                    BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                stateTracker->Record(
                    BufferView(data_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                stateTracker->UpdateState(&barrier_callback);
                bindProps->emplace_back(count_buffer);
                bindProps->emplace_back(data_buffer);
            }
        };
        if (cmd->is_indirect()) {
            auto &&t = cmd->indirect_dispatch();
            auto buffer = reinterpret_cast<Buffer *>(t.handle);
            bindProps->emplace_back();
            BeforeDispatch();
            bd->DispatchComputeIndirect(cs, *buffer, t.offset, t.max_dispatch_size, *bindProps);
        } else if (cmd->is_multiple_dispatch()) {
            size_t bindCount = bindProps->size();
            bindProps->emplace_back();
            BeforeDispatch();
            auto sizes = cmd->dispatch_sizes();
            bd->DispatchCompute(
                cs,
                sizes,
                bindCount,
                *bindProps);
        } else {
            auto &&t = cmd->dispatch_size();
            bindProps->emplace_back(4, make_uint4(t, 0));
            BeforeDispatch();
            bd->DispatchCompute(
                cs,
                t,
                *bindProps);
        }
        if (logger && data_buffer.buffer != nullptr) [[unlikely]] {
            stateTracker->Record(
                BufferView(count_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                EnhancedBarrierTracker::Usage::CopySource);
            stateTracker->Record(
                BufferView(data_buffer.buffer, count_buffer.offset, count_buffer.byteSize),
                EnhancedBarrierTracker::Usage::CopySource);
            stateTracker->UpdateState(&barrier_callback);
            bd->CopyBuffer(count_buffer.buffer, readback_count_buffer.buffer, count_buffer.offset, readback_count_buffer.offset, sizeof(uint));
            bd->CopyBuffer(data_buffer.buffer, readback_buffer.buffer, data_buffer.offset, readback_buffer.offset, data_buffer.byteSize);
            alloc->ExecuteAfterComplete([logger = this->logger, shader, readback_count_buffer, readback_buffer]() {
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
                auto printers = shader->Printers();
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
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->size());
        auto bfView = bd->GetCB()->GetAlloc()->GetTempUploadBuffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
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
        bd->CopyBufferTexture(
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
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Clear depth");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto cmdList = bd->GetCB()->CmdList();
        auto alloc = bd->GetCB()->GetAlloc();
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
        auto chunk = alloc->dsvAllocator.allocate(1);
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
        auto cmdList = bd->GetCB()->CmdList();
        FLOAT values[4];
        float4 c_values = cmd->value();
        std::memcpy(values, &c_values, sizeof(float4));

        auto alloc = bd->GetCB()->GetAlloc();
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
        auto chunk = alloc->rtvAllocator.allocate(1);
        auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
        rtvHandle = descHeap->hCPU(chunk.offset);
        auto viewDesc = rt->GetRenderTargetDesc(cmd->level());
        device->device->CreateRenderTargetView(rt->GetResource(), &viewDesc, rtvHandle);
        cmdList->ClearRenderTargetView(rtvHandle, values, 0, nullptr);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->size());
        auto alloc = bd->GetCB()->GetAlloc();
        auto bfView = alloc->GetTempReadbackBuffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);

        if (copyInfo.alignedBufferSize == copyInfo.bufferSize) {
            alloc->ExecuteAfterComplete(
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
            alloc->ExecuteAfterComplete(
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
        bd->CopyBufferTexture(
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
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        bd->CopyTexture(
            src,
            0,
            cmd->src_level(),
            dst,
            0,
            cmd->dst_level());
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture copy to buffer");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
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
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Accel build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        vstd::optional<BufferView> scratch;
        if (!cmd->update_instance_buffer_only()) {
            scratch.create(BufferView(accelScratchBuffer, accelScratchOffsets->first, accelScratchOffsets->second));
            if (accel->RequireCompact()) {
                updateAccel->emplace_back(ButtomCompactCmd{
                    .accel = accel,
                    .offset = accelScratchOffsets->first,
                    .size = accelScratchOffsets->second});
            }
            accelScratchOffsets++;
        }
        accel->Build(
            *stateTracker,
            *bd,
            scratch.has_value() ? scratch.ptr() : nullptr);
    }
    void BottomBuild(uint64 handle) {
        auto accel = reinterpret_cast<BottomAccel *>(handle);
        accel->UpdateStates(
            *stateTracker,
            *bd,
            BufferView(accelScratchBuffer, accelScratchOffsets->first, accelScratchOffsets->second),
            *bottomAccelData);
        if (accel->RequireCompact()) {
            updateAccel->emplace_back(ButtomCompactCmd{
                .accel = accel,
                .offset = accelScratchOffsets->first,
                .size = accelScratchOffsets->second});
        }
        accelScratchOffsets++;
        bottomAccelData++;
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
        LUISA_NOT_IMPLEMENTED();
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Mesh build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BottomBuild(cmd->handle());
    }
    void visit(const ProceduralPrimitiveBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Procedural build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BottomBuild(cmd->handle());
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Bindless-array update");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        cmd->visit_modifications([&]<typename T>(T const &t) {
            if constexpr (std::is_same_v<T, luisa::vector<BindlessArrayUpdateCommand::Texture3DModification>>) {
                arr->UpdateStates(
                    *bd,
                    *stateTracker,
                    luisa::span{
                        reinterpret_cast<BindlessArrayUpdateCommand::Texture2DModification const *>(t.data()),
                        t.size()});
            } else {
                arr->UpdateStates(
                    *bd,
                    *stateTracker,
                    luisa::span{t});
            }
        });
    }
    void visit(const DXCustomCmd *cmd) noexcept {
        cmd->execute(
            device->adapter.Get(),
            device->dxgiFactory.Get(),
            device->device.Get(),
            bd->GetCB()->CmdList());
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
            case to_underlying(CustomCommandUUID::WORK_GRAPH_DISPATCH):
                visit(static_cast<WorkGraphDispatchCommand const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }
    void visit(const WorkGraphDispatchCommand *cmd) noexcept {
        auto program = reinterpret_cast<WorkGraphProgram const *>(cmd->handle());

        // Build bind properties: only captured resources (work graph root sig has no sampler/bindless slots)
        bindProps->clear();
        Visitor visitor{this, program->Args().data()};
        DecodeCmd(program->ArgBindings(), visitor);

        D3D12_DISPATCH_GRAPH_DESC dispatchDesc{};
        dispatchDesc.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
        dispatchDesc.NodeCPUInput.EntrypointIndex = 0;
        dispatchDesc.NodeCPUInput.NumRecords = static_cast<UINT>(cmd->record_count());
        dispatchDesc.NodeCPUInput.RecordStrideInBytes = static_cast<UINT>(cmd->record_stride());
        dispatchDesc.NodeCPUInput.pRecords = cmd->records();
        bd->DispatchWorkGraph(program, dispatchDesc, *bindProps);
    }
    void visit(const DrawRasterSceneCommand *cmd) noexcept {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Draw raster command");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        bindProps->clear();
        auto cmdList = bd->GetCB()->CmdList();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        DepthFormat dsvFormat{DepthFormat::None};
        // TODO:Set render target
        // Set viewport
        auto alloc = bd->GetCB()->GetAlloc();
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
                auto chunk = alloc->rtvAllocator.allocate(rtvs.size());
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
                auto chunk = alloc->dsvAllocator.allocate(1);
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
        auto pso = shader->GetPSO({rtvFormats, rtvs.size()}, cmd->mesh_format(), dsvFormat, rasterState);
        auto &&tempBuffer = *bufferVec;
        bufferVec++;
        bindProps->emplace_back(DescriptorHeapView(device->samplerHeap.get()));
        if (tempBuffer.second > 0) {
            bindProps->emplace_back(BufferView(argBuffer.buffer, argBuffer.offset + tempBuffer.first, tempBuffer.second));
        }
        auto globalHeapView = DescriptorHeapView(device->globalHeap.get());
        vstd::push_back_func(*bindProps, shader->BindlessCount(), [&] { return globalHeapView; });
        DecodeCmd(cmd->arguments(), Visitor{this, shader->Args().data()});
        bd->SetRasterShader(shader, pso, *bindProps);
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
        auto propCount = shader->Properties().size();
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
    size_t maxAlloc) {
    auto allocator = queue.CreateAllocator(maxAlloc);
    auto allocType = allocator->Type();
    bool cmdListIsEmpty = commands.empty();
    luisa::fixed_vector<std::pair<IDXGISwapChain *, bool>, 4> present_swapchains;
    present_swapchains.reserve(presents.size());
    {
        std::unique_lock lck{mtx};
        LCPreProcessVisitor ppVisitor;
        ppVisitor.argVecs = &argVecs;
        ppVisitor.argBuffer = &argBuffer;
        ppVisitor.bottomAccelDatas = &bottomAccelDatas;
        ppVisitor.accelOffset = &accelOffset;
        argVecs.clear();
        argBuffer.clear();
        bottomAccelDatas.clear();
        accelOffset.clear();

        LCCmdVisitor visitor;
        if (logCallback) {
            visitor.logger = &logCallback;
        } else {
            visitor.logger = nullptr;
        }
        visitor.bindProps = &bindProps;
        visitor.updateAccel = &updateAccel;
        visitor.vbv = &vbv;
        visitor.device = device;
        visitor.after_custom_cmd = [](Device *device, CommandBufferBuilder *bd) {
            ID3D12DescriptorHeap *h[2] = {
                device->globalHeap->GetHeap(),
                device->samplerHeap->GetHeap()};
            auto cb = bd->GetCB();
            if (cb->GetAlloc()->Type() != D3D12_COMMAND_LIST_TYPE_COPY) {
                cb->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);
            }
        };
        auto cmdBuffer = allocator->GetBuffer();
        auto cmdBuilder = cmdBuffer->Build();
        GraphicsCmdlistBarrierCallback barrier_callback(cmdBuilder);
        if (!tracker) {
            if (device->feature_check.enhanced_barriers_supported()) {
                tracker = luisa::make_unique<EnhancedBarrierTrackerImpl>();
            } else {
                tracker = luisa::make_unique<EnhancedBarrierTrackerBackup>();
            }
        }
        tracker->listType = allocator->Type();
        visitor.stateTracker = tracker.get();
        ppVisitor.stateTracker = tracker.get();
        visitor.bd = &cmdBuilder;
        ppVisitor.bd = &cmdBuilder;
        reorder.clear();
        size_t uniformSize = 0;
        auto addSize = [&](auto const &c, Argument const &a) {
            if (a.tag != Argument::Tag::UNIFORM) [[likely]]
                return;

            // uniform_buffer_size +=
            uniformSize = CalcAlign(uniformSize, a.uniform.alignment);
            auto bf = c.uniform(a.uniform);
            uniformSize += std::max<size_t>(4, bf.size_bytes());
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
                    uniformSize = CalcAlign(uniformSize, 32);
                    for (auto &&i : cs->ArgBindings()) {
                        addSize(*c, i);
                    }
                    for (auto &&i : c->arguments()) {
                        addSize(*c, i);
                    }
                } break;
                case Command::Tag::ECustomCommand: {
                    if (static_cast<CustomCommand const *>(command.get())->custom_cmd_uuid() ==
                        to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE)) {
                        auto c = static_cast<DrawRasterSceneCommand const *>(command.get());
                        uniformSize = CalcAlign(uniformSize, 32);
                        for (auto &&i : c->arguments()) {
                            addSize(*c, i);
                        }
                    }
                } break;
            }
        }
        // Upload CBuffers
        if (uniformSize > 0) {
            visitor.argBuffer = allocator->GetTempUploadBuffer(uniformSize, 32);
        } else {
            visitor.argBuffer = {};
        }
        auto cmdLists = reorder.command_lists();
        ID3D12DescriptorHeap *h[2] = {
            device->globalHeap->GetHeap(),
            device->samplerHeap->GetHeap()};
        tracker->restoreStates.clear();
        if (device->deviceSettings) {
            auto after_states = device->deviceSettings->after_states(reinterpret_cast<uint64_t>(this));
            auto before_states = device->deviceSettings->before_states(reinterpret_cast<uint64_t>(this));
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
            if (allocType != D3D12_COMMAND_LIST_TYPE_COPY) {
                cmdBuffer->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);
            }

            // Clear caches
            ppVisitor.argVecs->clear();
            ppVisitor.accelOffset->clear();
            ppVisitor.bottomAccelDatas->clear();
            ppVisitor.buildAccelSize = 0;
            // Preprocess: record resources' states
            for (auto i = lst; i != nullptr; i = i->p_next) {
                if (i->cmd)
                    i->cmd->accept(ppVisitor);
            }
            // command->accept(ppVisitor);
            visitor.bottomAccelData = ppVisitor.bottomAccelDatas->data();
            DefaultBuffer const *accelScratchBuffer;
            if (ppVisitor.buildAccelSize) {
                accelScratchBuffer = allocator->AllocateScratchBuffer(ppVisitor.buildAccelSize);
                visitor.accelScratchOffsets = ppVisitor.accelOffset->data();
                visitor.accelScratchBuffer = accelScratchBuffer;
                tracker->Record(accelScratchBuffer, EnhancedBarrierTracker::Usage::BuildAccelScratch);
            }

            tracker->UpdateState(
                &barrier_callback);
            visitor.bufferVec = ppVisitor.argVecs->data();
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
                            cmdBuilder,
                            BufferView(
                                accelScratchBuffer,
                                i.offset,
                                i.size));
                    });
                }
                tracker->RestoreState(&barrier_callback);
                auto localUpdateAccel = std::move(updateAccel);
                lck.unlock();
                queue.ForceSync(
                    allocator,
                    *cmdBuffer);
                for (auto &&i : localUpdateAccel) {
                    i.accel.visit([&](auto &&p) {
                        p->CheckAccel(cmdBuilder);
                    });
                }
                lck.lock();
            }
        }
        if (visitor.argBuffer.buffer) {
            static_cast<UploadBuffer const *>(visitor.argBuffer.buffer)
                ->CopyData(
                    visitor.argBuffer.offset,
                    {reinterpret_cast<uint8_t const *>(
                         argBuffer.data()),
                     argBuffer.size()});
            auto aligned_size = CalcAlign(argBuffer.size(), 32);
            uniformSize = CalcAlign(uniformSize, 32);
            LUISA_DEBUG_ASSERT(aligned_size == uniformSize);
        }

        for (auto &&present : presents) {
            auto swapchain = reinterpret_cast<LCSwapChain *>(present.chain->handle());
            auto &&rt = &swapchain->m_renderTargets[swapchain->frameIndex];
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
            auto &&rt = &swapchain->m_renderTargets[swapchain->frameIndex];
            auto img = reinterpret_cast<TextureBase *>(present.frame.handle());
            auto mip = present.frame.level();
            swapchain->frameIndex += 1;
            swapchain->frameIndex %= swapchain->frameCount;
            D3D12_TEXTURE_COPY_LOCATION sourceLocation;
            sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            sourceLocation.SubresourceIndex = mip;
            D3D12_TEXTURE_COPY_LOCATION destLocation;
            destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destLocation.SubresourceIndex = 0;
            sourceLocation.pResource = img->GetResource();
            destLocation.pResource = rt->GetResource();
            cmdBuffer->CmdList()->CopyTextureRegion(
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
            present_swapchains.emplace_back(swapchain->swapChain.Get(), swapchain->vsync);
        }
        tracker->RestoreState(&barrier_callback);
    }
    queue.Execute(std::move(allocator), std::move(funcs), luisa::span{present_swapchains}, cmdListIsEmpty);
}
void LCCmdBuffer::Sync() {
    queue.Complete();
}
void LCCmdBuffer::Present(
    LCSwapChain *swapchain,
    TextureBase *img,
    uint mip,
    size_t maxAlloc) {
    auto alloc = queue.CreateAllocator(maxAlloc);
    {
        std::lock_guard lck{mtx};
        // swapchain->frameIndex = swapchain->swapChain->GetCurrentBackBufferIndex();
        auto &&rt = &swapchain->m_renderTargets[swapchain->frameIndex];
        swapchain->frameIndex += 1;
        swapchain->frameIndex %= swapchain->frameCount;
        auto cb = alloc->GetBuffer();
        auto bd = cb->Build();
        auto cmdList = cb->CmdList();
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
            bd.GetCB()->CmdList()->ResourceBarrier(vstd::array_count(barriers), barriers);
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
            bd.GetCB()->CmdList()->ResourceBarrier(vstd::array_count(barriers), barriers);
        }
    }
    std::pair<IDXGISwapChain *, bool> sw{swapchain->swapChain.Get(), swapchain->vsync};
    queue.Execute(std::move(alloc), {}, {&sw, 1}, false);
}
void LCCmdBuffer::CompressBC(
    TextureBase *rt,
    uint level,
    luisa::compute::BufferView<uint> const &result,
    bool isHDR,
    float alphaImportance,
    GpuAllocator *allocator,
    size_t maxAlloc) {
    if (!tracker) {
        if (device->feature_check.enhanced_barriers_supported()) {
            tracker = luisa::make_unique<EnhancedBarrierTrackerImpl>();
        } else {
            tracker = luisa::make_unique<EnhancedBarrierTrackerBackup>();
        }
    }
    alphaImportance = std::max<float>(std::min<float>(alphaImportance, 1), 0);// clamp<float>(alphaImportance, 0, 1);
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
    uint xBlocks = std::max<uint>(1, (width + 3) >> 2);
    uint yBlocks = std::max<uint>(1, (height + 3) >> 2);
    uint numBlocks = xBlocks * yBlocks;
    uint numTotalBlocks = numBlocks;
    static constexpr size_t BLOCK_SIZE = 16;
    if (result.size_bytes() != BLOCK_SIZE * numBlocks) [[unlikely]] {
        LUISA_ERROR("Texture compress output buffer incorrect size!");
    }
    DefaultBuffer backBuffer(
        device,
        BLOCK_SIZE * numBlocks,
        allocator,
        D3D12_RESOURCE_STATE_COMMON);
    auto outBufferPtr = reinterpret_cast<Buffer *>(result.handle());
    BufferView outBuffer{
        outBufferPtr,
        result.offset_bytes(),
        result.size_bytes()};

    constexpr uint MAX_BATCH = 1024 * 1024;
    auto batchNum = static_cast<int>((numTotalBlocks + MAX_BATCH - 1) / MAX_BATCH);
    uint startBlockID = 0;
    for (int batch = 0; batch < batchNum; batch++) {
        auto target = static_cast<int>((batch + 1) * MAX_BATCH);
        auto alloc = queue.CreateAllocator(maxAlloc);
        {
            std::lock_guard lck{mtx};
            tracker->listType = alloc->Type();
            // auto bufferReadState = tracker->ReadState(ResourceReadUsage::Srv);
            auto cmdBuffer = alloc->GetBuffer();
            auto cmdBuilder = cmdBuffer->Build();
            GraphicsCmdlistBarrierCallback barrier_callback(cmdBuilder);
            ID3D12DescriptorHeap *h[2] = {
                device->globalHeap->GetHeap(),
                device->samplerHeap->GetHeap()};
            cmdBuffer->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);

            BCCBuffer cbData{
                .g_mip_level = level};
            tracker->Record(
                EnhancedBarrierTracker::TexView(rt, level),
                EnhancedBarrierTracker::Usage::ComputeRead);
            auto RunComputeShader = [&](ComputeShader const *cs, uint dispatchCount, BufferView const &inBuffer, BufferView const &outBuffer) {
                auto cbuffer = alloc->GetTempUploadBuffer(sizeof(BCCBuffer), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
                static_cast<UploadBuffer const *>(cbuffer.buffer)->CopyData(cbuffer.offset, {reinterpret_cast<uint8_t const *>(&cbData), sizeof(BCCBuffer)});
                tracker->Record(
                    inBuffer,
                    EnhancedBarrierTracker::Usage::ComputeRead);
                tracker->Record(
                    outBuffer,
                    EnhancedBarrierTracker::Usage::ComputeUAV);
                tracker->UpdateState(&barrier_callback);
                BindProperty prop[4];
                prop[0] = cbuffer;
                prop[1] = DescriptorHeapView(device->globalHeap.get(), rt->GetGlobalSRVIndex());
                prop[2] = inBuffer;
                prop[3] = outBuffer;
                cmdBuilder.DispatchCompute(
                    cs,
                    uint3(dispatchCount, 1, 1),
                    {prop, 4});
            };
            constexpr uint MAX_BLOCK_BATCH = 1024u * 32u;
            if (isHDR)//bc6
            {
                BufferView err1Buffer{&backBuffer};
                BufferView err2Buffer{outBuffer};
                auto bc6TryModeG10 = device->bc6TryModeG10.Get(device);
                auto bc6TryModeLE10 = device->bc6TryModeLE10.Get(device);
                auto bc6Encode = device->bc6EncodeBlock.Get(device);
                while (numBlocks > 0 && startBlockID < target) {
                    uint n = std::min<uint>(numBlocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cbData.g_tex_width = width;
                    cbData.g_num_block_x = xBlocks;
                    cbData.g_format = isHDR ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cbData.g_start_block_id = startBlockID;
                    cbData.g_alpha_weight = alphaImportance;
                    cbData.g_num_total_blocks = numTotalBlocks;
                    RunComputeShader(
                        bc6TryModeG10,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2Buffer,
                        err1Buffer);
                    for (auto i : vstd::range(10)) {
                        cbData.g_mode_id = i;
                        RunComputeShader(
                            bc6TryModeLE10,
                            std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                            ((i & 1) != 0) ? err2Buffer : err1Buffer,
                            ((i & 1) != 0) ? err1Buffer : err2Buffer);
                    }
                    RunComputeShader(
                        bc6Encode,
                        std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                        err1Buffer,
                        err2Buffer);
                    startBlockID += n;
                    numBlocks -= n;
                }

            } else {
                BufferView err1Buffer{outBuffer};
                BufferView err2Buffer{&backBuffer};
                auto bc7Try137Mode = device->bc7TryMode137.Get(device);
                auto bc7Try02Mode = device->bc7TryMode02.Get(device);
                auto bc7Try456Mode = device->bc7TryMode456.Get(device);
                auto bc7Encode = device->bc7EncodeBlock.Get(device);
                while (numBlocks > 0 && startBlockID < target) {
                    uint n = std::min<uint>(numBlocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cbData.g_tex_width = width;
                    cbData.g_num_block_x = xBlocks;
                    cbData.g_format = isHDR ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cbData.g_start_block_id = startBlockID;
                    cbData.g_alpha_weight = alphaImportance;
                    cbData.g_num_total_blocks = numTotalBlocks;
                    RunComputeShader(bc7Try456Mode, std::max<uint>((uThreadGroupCount + 3) / 4, 1), err2Buffer, err1Buffer);
                    //137
                    {
                        uint modes[] = {1, 3, 7};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cbData.g_mode_id = modes[i];
                            RunComputeShader(
                                bc7Try137Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err2Buffer : err1Buffer,
                                ((i & 1) != 0) ? err1Buffer : err2Buffer);
                        }
                    }
                    //02
                    {
                        uint modes[] = {0, 2};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cbData.g_mode_id = modes[i];
                            RunComputeShader(
                                bc7Try02Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err1Buffer : err2Buffer,
                                ((i & 1) != 0) ? err2Buffer : err1Buffer);
                        }
                    }
                    RunComputeShader(
                        bc7Encode,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2Buffer,
                        err1Buffer);
                    //TODO
                    startBlockID += n;
                    numBlocks -= n;
                }
            }
            tracker->Record(outBuffer, EnhancedBarrierTracker::Usage::CopySource);
            tracker->RestoreState(&barrier_callback);
        }
        if (batch == batchNum - 1) {
            vstd::vector<vstd::function<void()>> callbacks;
            callbacks.emplace_back([backBuffer = std::move(backBuffer)] {});
            queue.Execute(
                std::move(alloc),
                std::move(callbacks), {}, false);
        } else {
            queue.Execute(std::move(alloc), {}, {}, false);
        }
    }
}
}// namespace lc::dx
