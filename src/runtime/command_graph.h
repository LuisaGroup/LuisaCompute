//
// Created by Mike Smith on 2022/4/20.
//

#pragma once

#include <core/stl.h>
#include <runtime/command.h>
#include <runtime/command_list.h>
#include <runtime/device.h>

namespace luisa::compute {

class CommandGraph final : CommandVisitor {

private:
    const Device::Interface *_device;
    luisa::vector<Command *> _commands;
    luisa::vector<luisa::vector<uint>> _edges;// _edges[0] stores indices of all commands that depends on _commands[0]
    luisa::vector<uint> _dependency_count;    // number of dependencies of each command

private:
    [[nodiscard]] bool _check_buffer_read(uint64_t handle, size_t offset, size_t size, const Command *command) const noexcept;
    [[nodiscard]] bool _check_buffer_write(uint64_t handle, size_t offset, size_t size, const Command *command) const noexcept;
    [[nodiscard]] bool _check_texture_read(uint64_t handle, uint level, const Command *command) const noexcept;
    [[nodiscard]] bool _check_texture_write(uint64_t handle, uint level, const Command *command) const noexcept;
    [[nodiscard]] bool _check_mesh_write(uint64_t handle, const Command *command) const noexcept;
    [[nodiscard]] bool _check_accel_read(uint64_t handle, const Command *command) const noexcept;
    [[nodiscard]] bool _check_accel_write(uint64_t handle, const Command *command) const noexcept;
    [[nodiscard]] bool _check_bindless_array_read(uint64_t handle, const Command *command) const noexcept;
    [[nodiscard]] bool _check_bindless_array_write(uint64_t handle, const Command *command) const noexcept;

private:
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

public:
    explicit CommandGraph(const Device::Interface *device) noexcept;
    void add(Command *command) noexcept;
    [[nodiscard]] luisa::vector<CommandList> schedule() noexcept;
};

}// namespace luisa::compute