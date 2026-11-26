//
// Created by mike on 1/9/26.
//

#pragma once

#include <hip/hip_runtime.h>
#include <luisa/runtime/rhi/command.h>

#include "hip_stream.h"

namespace luisa::compute::hip {

class HIPCommandEncoder : public MutableCommandVisitor {

private:
    HIPStream *_stream;
    luisa::vector<HIPCallbackContext *> _callbacks;
    luisa::vector<HIPStream::DeferredDownloadCopy> _deferred_downloads;
    luisa::vector<MeshBuildCommand *> _pending_mesh_builds;

    void _flush_pending_mesh_builds() noexcept;

public:
    explicit HIPCommandEncoder(HIPStream *stream) noexcept
        : _stream{stream} {}

    [[nodiscard]] auto stream() const noexcept { return _stream; }
    void add_callback(HIPCallbackContext *cb) noexcept { _callbacks.emplace_back(cb); }
    void commit(CommandList::CallbackContainer &&user_callbacks) noexcept;

    void visit(BufferUploadCommand *) noexcept override;
    void visit(BufferDownloadCommand *) noexcept override;
    void visit(BufferCopyCommand *) noexcept override;
    void visit(BufferToTextureCopyCommand *) noexcept override;
    void visit(ShaderDispatchCommand *) noexcept override;
    void visit(TextureUploadCommand *) noexcept override;
    void visit(TextureDownloadCommand *) noexcept override;
    void visit(TextureCopyCommand *) noexcept override;
    void visit(TextureToBufferCopyCommand *) noexcept override;
    void visit(AccelBuildCommand *) noexcept override;
    void visit(MeshBuildCommand *) noexcept override;
    void visit(CurveBuildCommand *) noexcept override;
    void visit(ProceduralPrimitiveBuildCommand *) noexcept override;
    void visit(MotionInstanceBuildCommand *) noexcept override;
    void visit(BindlessArrayUpdateCommand *) noexcept override;
    void visit(CustomCommand *) noexcept override;

    template<typename F>
    void with_upload_buffer(size_t size, F &&f) noexcept {
        auto upload_buffer = _stream->upload_pool()->allocate(size);
        f(upload_buffer);
        _callbacks.emplace_back(upload_buffer);
    }

    template<typename F>
    void with_download_pool(size_t size, F &&f) noexcept {
        auto download_buffer = _stream->download_pool()->allocate(size);
        f(download_buffer);
        _callbacks.emplace_back(download_buffer);
    }

    template<typename F>
    void with_upload_pool_no_fallback(size_t size, F &&f) noexcept {
        auto upload_buffer = _stream->upload_pool()->allocate(size, false);
        f(upload_buffer);
        if (upload_buffer) {
            _callbacks.emplace_back(upload_buffer);
        }
    }

    template<typename F>
    void with_download_pool_no_fallback(size_t size, F &&f) noexcept {
        auto download_buffer = _stream->download_pool()->allocate(size, false);
        f(download_buffer);
        if (download_buffer) {
            _callbacks.emplace_back(download_buffer);
        }
    }
};

}// namespace luisa::compute::hip
