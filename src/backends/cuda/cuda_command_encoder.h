//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <cuda.h>
#include <runtime/command.h>

namespace luisa::compute::cuda {

class CUDACommandEncoder : public CommandVisitor {

private:
    CUstream _stream;

public:
    explicit CUDACommandEncoder(CUstream stream) noexcept : _stream{stream} {}
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;
    void visit(const ShaderDispatchCommand *command) noexcept override;
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;
    void visit(const AccelUpdateCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;
    void visit(const MeshUpdateCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
};

}// namespace luisa::compute::cuda
