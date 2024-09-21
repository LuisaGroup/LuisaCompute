#pragma once

#include <cuda.h>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/runtime/rhi/command.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include "cuda_stream.h"

namespace luisa::compute::cuda {

class CUDADevice;

/**
 * @brief Command encoder of CUDA
 * 
 */
class CUDACommandEncoder : public MutableCommandVisitor {

private:
    CUDAStream *_stream;
    luisa::vector<CUDACallbackContext *> _callbacks;

private:
    void visit(DStorageReadCommand *command) noexcept;

public:
    explicit CUDACommandEncoder(CUDAStream *stream) noexcept
        : _stream{stream} {}

    [[nodiscard]] auto stream() const noexcept { return _stream; }
    void add_callback(CUDACallbackContext *cb) noexcept { _callbacks.emplace_back(cb); }

    void visit(BufferUploadCommand *command) noexcept override;
    void visit(BufferDownloadCommand *command) noexcept override;
    void visit(BufferCopyCommand *command) noexcept override;
    void visit(BufferToTextureCopyCommand *command) noexcept override;
    void visit(ShaderDispatchCommand *command) noexcept override;
    void visit(TextureUploadCommand *command) noexcept override;
    void visit(TextureDownloadCommand *command) noexcept override;
    void visit(TextureCopyCommand *command) noexcept override;
    void visit(TextureToBufferCopyCommand *command) noexcept override;
    void visit(AccelBuildCommand *command) noexcept override;
    void visit(MeshBuildCommand *command) noexcept override;
    void visit(CurveBuildCommand *command) noexcept override;
    void visit(BindlessArrayUpdateCommand *command) noexcept override;
    void visit(ProceduralPrimitiveBuildCommand *command) noexcept override;
    void visit(CustomCommand *command) noexcept override;
    void visit(MotionInstanceBuildCommand *command) noexcept override;
    void commit(CommandList::CallbackContainer &&user_callbacks) noexcept;

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

}// namespace luisa::compute::cuda

