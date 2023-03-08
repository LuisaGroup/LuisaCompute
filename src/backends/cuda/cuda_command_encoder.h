//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <cuda.h>
#include <runtime/rhi/command.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;

/**
 * @brief Command encoder of CUDA
 * 
 */
class CUDACommandEncoder : public CommandVisitor {

private:
    CUDAStream *_stream;

public:
    explicit CUDACommandEncoder(CUDAStream *stream) noexcept: _stream{stream} {}
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
    void visit(const ProceduralPrimitiveBuildCommand *command) noexcept override;
    void visit(const CustomCommand *command) noexcept override;
    void visit(const DrawRasterSceneCommand *command) noexcept override;
    void visit(const ClearDepthCommand *command) noexcept override;
};

}// namespace luisa::compute::cuda
