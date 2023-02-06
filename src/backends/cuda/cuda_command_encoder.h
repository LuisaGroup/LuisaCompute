//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <cuda.h>
#include <runtime/command.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;

/**
 * @brief Command encoder of CUDA
 * 
 */
class CUDACommandEncoder : public CommandVisitor {

private:
    CUDADevice *_device;
    CUDAStream *_stream;

public:
    explicit CUDACommandEncoder(CUDADevice *device, CUDAStream *stream) noexcept
        : _device{device}, _stream{stream} {}
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;
    void visit(const ShaderDispatchCommand *command) noexcept override;
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
    void visit(const BindlessArrayUpdateCommand *command) noexcept override;
    void visit(const ShaderDispatchExCommand *command) noexcept override;

    template<typename F>
    void with_upload_buffer(size_t size, F &&f) noexcept;
};

}// namespace luisa::compute::cuda
