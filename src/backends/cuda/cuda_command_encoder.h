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
    // Staging views, reclaimed by a stream event instead of a host callback.
    luisa::vector<CUDACallbackContext *> _deferred;
    // Seeded from the stream at the start of a submission and advanced after
    // every command, so the launch policy always sees the true predecessor of
    // the grid it is about to launch.
    bool _pdl_predecessor_is_kernel{false};

private:
    void visit(DStorageReadCommand *command) noexcept;

public:
    explicit CUDACommandEncoder(CUDAStream *stream) noexcept
        : _stream{stream} {}

    [[nodiscard]] auto stream() const noexcept { return _stream; }
    void add_callback(CUDACallbackContext *cb) noexcept { _callbacks.emplace_back(cb); }

    // The one entry point every command goes through.
    void visit_command(Command *command) noexcept {
        // A command that launches no grid (a zero-sized dispatch, a copy, a
        // build, ...) is not a kernel predecessor.
        _pdl_predecessor_is_kernel = false;
        command->accept(*this);
    }
    void seed_pdl_predecessor(bool is_kernel) noexcept { _pdl_predecessor_is_kernel = is_kernel; }
    [[nodiscard]] bool pdl_predecessor_is_kernel() const noexcept { return _pdl_predecessor_is_kernel; }
    // Called by the shader launch path when it actually enqueued a grid.
    void note_pdl_grid_launched() noexcept { _pdl_predecessor_is_kernel = true; }
    // A command that splices nested commands into the stream but is itself not a
    // kernel launch (the software-ray-tracing build paths).
    void note_pdl_non_kernel_command() noexcept { _pdl_predecessor_is_kernel = false; }

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
        _deferred.emplace_back(upload_buffer);
    }

    template<typename F>
    void with_download_pool(size_t size, F &&f) noexcept {
        auto download_buffer = _stream->download_pool()->allocate(size);
        f(download_buffer);
        _deferred.emplace_back(download_buffer);
    }

    template<typename F>
    void with_upload_pool_no_fallback(size_t size, F &&f) noexcept {
        auto upload_buffer = _stream->upload_pool()->allocate(size, false);
        f(upload_buffer);
        if (upload_buffer) {
            _deferred.emplace_back(upload_buffer);
        }
    }

    template<typename F>
    void with_download_pool_no_fallback(size_t size, F &&f) noexcept {
        auto download_buffer = _stream->download_pool()->allocate(size, false);
        f(download_buffer);
        if (download_buffer) {
            _deferred.emplace_back(download_buffer);
        }
    }
};

}// namespace luisa::compute::cuda

