#include <algorithm>
#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>

#include "fallback_stream.h"
#include "fallback_accel.h"
#include "fallback_bindless_array.h"
#include "fallback_mesh.h"
#include "fallback_curve.h"
#include "fallback_proc_prim.h"
#include "fallback_motion_instance.h"
#include "fallback_texture.h"
#include "fallback_shader.h"
#include "fallback_buffer.h"

namespace luisa::compute::fallback {

void FallbackStream::_enqueue(luisa::unique_ptr<BufferUploadCommand> cmd) noexcept {
    auto temp_buffer = luisa::allocate_with_allocator<std::byte>(cmd->size());
    std::memcpy(temp_buffer, cmd->data(), cmd->size());
    auto dst = reinterpret_cast<FallbackBuffer *>(cmd->handle())->view(cmd->offset(), cmd->size());
    queue()->enqueue([src = temp_buffer, dst] {
        std::memcpy(dst.ptr, src, dst.size);
        luisa::deallocate_with_allocator(src);
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<BufferDownloadCommand> cmd) noexcept {
    auto src = reinterpret_cast<FallbackBuffer *>(cmd->handle())->view(cmd->offset(), cmd->size());
    queue()->enqueue([dst = cmd->data(), src] { std::memcpy(dst, src.ptr, src.size); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<BufferCopyCommand> cmd) noexcept {
    auto src = reinterpret_cast<FallbackBuffer *>(cmd->src_handle())->view(cmd->src_offset(), cmd->size());
    auto dst = reinterpret_cast<FallbackBuffer *>(cmd->dst_handle())->view(cmd->dst_offset(), cmd->size());
    queue()->enqueue([src, dst = dst.ptr] { std::memcpy(dst, src.ptr, src.size); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<BufferToTextureCopyCommand> cmd) noexcept {
    auto buffer = reinterpret_cast<FallbackBuffer *>(cmd->buffer());
    auto texture = reinterpret_cast<FallbackTexture *>(cmd->texture());
    auto src = buffer->view_with_offset(cmd->buffer_offset());
    auto dst = texture->view(cmd->level());
    queue()->enqueue([src, dst] { dst.copy_from(src.ptr); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<ShaderDispatchCommand> cmd) noexcept {
    auto shader = reinterpret_cast<FallbackShader *>(cmd->handle());
    shader->dispatch(queue(), std::move(cmd));
}

void FallbackStream::_enqueue(luisa::unique_ptr<TextureUploadCommand> cmd) noexcept {
    auto tex = reinterpret_cast<FallbackTexture *>(cmd->handle())->view(cmd->level());
    auto byte_size = pixel_storage_size(tex.storage(), tex.size3d());
    auto temp_buffer = luisa::allocate_with_allocator<std::byte>(byte_size);
    std::memcpy(temp_buffer, cmd->data(), byte_size);
    queue()->enqueue([tex, temp_buffer] {
        auto byte_size = pixel_storage_size(tex.storage(), tex.size3d());
        std::memcpy(const_cast<std::byte *>(tex.data()), temp_buffer, byte_size);
        luisa::deallocate_with_allocator(temp_buffer);
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<TextureDownloadCommand> cmd) noexcept {
    auto tex = reinterpret_cast<FallbackTexture *>(cmd->handle())->view(cmd->level());
    queue()->enqueue([=, dst = cmd->data()] { tex.copy_to(dst); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<TextureCopyCommand> cmd) noexcept {
    auto src_tex = reinterpret_cast<FallbackTexture *>(cmd->src_handle())->view(cmd->src_level());
    auto dst_tex = reinterpret_cast<FallbackTexture *>(cmd->dst_handle())->view(cmd->dst_level());
    queue()->enqueue([=] { dst_tex.copy_from(src_tex); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<TextureToBufferCopyCommand> cmd) noexcept {
    auto src = reinterpret_cast<FallbackTexture *>(cmd->texture())->view(cmd->level());
    auto dst = reinterpret_cast<FallbackBuffer *>(cmd->buffer())->view_with_offset(cmd->buffer_offset());
    queue()->enqueue([src, dst = dst.ptr] { src.copy_to(dst); });
}

void FallbackStream::_enqueue(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto accel = reinterpret_cast<FallbackAccel *>(cmd->handle());
        accel->build(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<MeshBuildCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto mesh = reinterpret_cast<FallbackMesh *>(cmd->handle());
        mesh->build(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<CurveBuildCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto curve = reinterpret_cast<FallbackCurve *>(cmd->handle());
        curve->build(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<ProceduralPrimitiveBuildCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto prim = reinterpret_cast<FallbackProceduralPrim *>(cmd->handle());
        prim->build(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<MotionInstanceBuildCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto instance = reinterpret_cast<FallbackMotionInstance *>(cmd->handle());
        instance->build(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<BindlessArrayUpdateCommand> cmd) noexcept {
    queue()->enqueue([cmd = std::move(cmd)]() mutable noexcept {
        auto bindless = reinterpret_cast<FallbackBindlessArray *>(cmd->handle());
        bindless->update(std::move(cmd));
    });
}

void FallbackStream::_enqueue(luisa::unique_ptr<CustomCommand> cmd) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void FallbackStream::dispatch(CommandList &&cmd_list) noexcept {
    static const bool orig_stream_mode = []() noexcept {
        if (const auto *s = std::getenv("LUISA_FALLBACK_BACKEND_SCHEDULER")) {
            if (std::string_view{s} == "akari_orig") { return true; }
        }
        if (const auto *q = std::getenv("LUISA_FALLBACK_QUEUE_MODE")) {
            if (std::string_view{q} == "orig") { return true; }
        }
        return false;
    }();
    auto cmds = cmd_list.steal_commands();
    for (auto &&cmd : cmds) {
#define LUISA_FALLBACK_STREAM_CAST_AND_ENQUEUE_COMMAND_CASE(COMMAND_TYPE) \
    case Command::Tag::E##COMMAND_TYPE: {                                 \
        auto derived_cmd = static_cast<COMMAND_TYPE *>(cmd.release());    \
        _enqueue(luisa::unique_ptr<COMMAND_TYPE>{derived_cmd});           \
        break;                                                            \
    }
        switch (cmd->tag()) {
            LUISA_MAP(LUISA_FALLBACK_STREAM_CAST_AND_ENQUEUE_COMMAND_CASE,
                      LUISA_COMPUTE_RUNTIME_COMMANDS)
        }
#undef LUISA_FALLBACK_STREAM_CAST_AND_ENQUEUE_COMMAND_CASE
    }
    auto callbacks = cmd_list.steal_callbacks();
    if (orig_stream_mode || !callbacks.empty()) {
        dispatch([callbacks = std::move(callbacks)] {
            for (auto &&cb : callbacks) { cb(); }
        });
    }
}

void FallbackStream::dispatch(luisa::move_only_function<void()> &&f) noexcept {
    queue()->enqueue(std::move(f));
}

FallbackStream::FallbackStream(size_t in_flight_limit) noexcept
    : _queue{in_flight_limit, 0u} {}

}// namespace luisa::compute::fallback
