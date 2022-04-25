//
// Created by Mike Smith on 2022/4/20.
//

#include "core/logging.h"
#include <algorithm>
#include <bitset>
#include <runtime/command_scheduler.h>

namespace luisa::compute {

CommandScheduler::CommandScheduler(const Device::Interface *device, size_t window_size) noexcept
    : _device{device}, _window_size{window_size} {}

void CommandScheduler::_schedule_step() noexcept {
    CommandList list;
    list.reserve(_free_nodes.size());
    for (auto i : _free_nodes) {
        list.append(_commands[i]);
        _pending_nodes.erase(i);
        for (auto j : _edges[i]) {
            if (--_dependency_count[j] == 0u) {
                _free_nodes_swap.emplace_back(j);
            }
        }
    }
    _command_lists.emplace_back(std::move(list));
    _free_nodes.swap(_free_nodes_swap);
    _free_nodes_swap.clear();
}

void CommandScheduler::add(Command *command) noexcept {
    auto node = static_cast<uint>(_commands.size());
    // determine dependencies...
    _dependency_count.emplace_back(0u);
    command->accept(*this);
    // add the command
    _commands.emplace_back(command);
    _edges.emplace_back();
    _pending_nodes.emplace(node);
    if (_dependency_count[node] == 0u) {// free node
        _free_nodes.emplace_back(node);
    }
    // perform a schedule step if exceeds the window size
    if (_pending_nodes.size() >= _window_size) {
        _schedule_step();
    }
}

luisa::vector<CommandList> CommandScheduler::schedule() noexcept {
    // schedule all pending nodes
    while (!_pending_nodes.empty()) { _schedule_step(); }
    auto total = 0u;
    for (auto &list : _command_lists) { total += list.size(); }
    LUISA_ASSERT(total == _commands.size(),
                 "CommandGraph::schedule: command count "
                 "mismatch (expected {}, got {}).",
                 _commands.size(), total);
    _commands.clear();
    _edges.clear();
    _dependency_count.clear();
    _pending_nodes.clear();
    _free_nodes.clear();
    _free_nodes_swap.clear();
    luisa::vector<CommandList> lists;
    lists.swap(_command_lists);
    return lists;
}

[[nodiscard]] auto buffers_overlap(uint64_t h1, size_t o1, size_t s1,
                                   uint64_t h2, size_t o2, size_t s2) noexcept {
    return h1 == h2 && (o1 < o2 + s2) && (o2 < o1 + s1);
}

[[nodiscard]] auto textures_overlap(uint64_t h1, uint l1, uint64_t h2, uint l2) noexcept {
    return h1 == h2 && l1 == l2;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

bool CommandScheduler::_check_buffer_read(uint64_t handle, size_t offset, size_t size, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBufferUploadCommand: {
            auto other = static_cast<const BufferUploadCommand *>(command);
            return buffers_overlap(handle, offset, size, other->handle(), other->offset(), other->size());
        }
        case Command::Tag::EBufferCopyCommand: {
            auto other = static_cast<const BufferCopyCommand *>(command);
            return buffers_overlap(handle, offset, size, other->dst_handle(), other->dst_offset(), other->size());
        }
        case Command::Tag::EShaderDispatchCommand: {
            auto other = static_cast<const ShaderDispatchCommand *>(command);
            auto overlap = false;
            other->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                    if (auto usage = other->kernel().variable_usage(argument.variable_uid);
                        (usage == Usage::WRITE || usage == Usage::READ_WRITE) &&
                        buffers_overlap(handle, offset, size, argument.handle, argument.offset, argument.size)) {
                        overlap = true;
                        return;
                    }
                }
            });
            return overlap;
        }
        case Command::Tag::ETextureToBufferCopyCommand: {
            auto other = static_cast<const TextureToBufferCopyCommand *>(command);
            return buffers_overlap(handle, offset, size, other->buffer(), other->buffer_offset(),
                                   other->size().x * other->size().y * other->size().z *
                                       pixel_storage_size(other->storage()));
        }
        default: break;// impossible to overlap...
    }
    return false;
}

bool CommandScheduler::_check_buffer_write(uint64_t handle, size_t offset, size_t size, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBufferUploadCommand: {
            auto other = static_cast<const BufferUploadCommand *>(command);
            return buffers_overlap(handle, offset, size, other->handle(), other->offset(), other->size());
        }
        case Command::Tag::EBufferDownloadCommand: {
            auto other = static_cast<const BufferDownloadCommand *>(command);
            return buffers_overlap(handle, offset, size, other->handle(), other->offset(), other->size());
        }
        case Command::Tag::EBufferCopyCommand: {
            auto other = static_cast<const BufferCopyCommand *>(command);
            return buffers_overlap(handle, offset, size, other->src_handle(), other->src_offset(), other->size()) ||
                   buffers_overlap(handle, offset, size, other->dst_handle(), other->dst_offset(), other->size());
        }
        case Command::Tag::EBufferToTextureCopyCommand: {
            auto other = static_cast<const BufferToTextureCopyCommand *>(command);
            return buffers_overlap(handle, offset, size, other->buffer(), other->buffer_offset(),
                                   other->size().x * other->size().y * other->size().z *
                                       pixel_storage_size(other->storage()));
        }
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            static_cast<const ShaderDispatchCommand *>(command)->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                    if (buffers_overlap(handle, offset, size, argument.handle, argument.offset, argument.size)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                    if (_device->is_resource_in_bindless_array(argument.handle, handle)) {
                        overlap = true;
                        return;
                    }
                }
            });
            return overlap;
        }
        case Command::Tag::ETextureToBufferCopyCommand: {
            auto other = static_cast<const TextureToBufferCopyCommand *>(command);
            return buffers_overlap(handle, offset, size, other->buffer(), other->buffer_offset(),
                                   other->size().x * other->size().y * other->size().z *
                                       pixel_storage_size(other->storage()));
        }
        case Command::Tag::EMeshBuildCommand: {
            auto other = static_cast<const MeshBuildCommand *>(command);
            return buffers_overlap(handle, offset, size, other->vertex_buffer(),
                                   other->vertex_buffer_offset(), other->vertex_buffer_size()) ||
                   buffers_overlap(handle, offset, size, other->triangle_buffer(),
                                   other->triangle_buffer_offset(), other->triangle_buffer_size());
        }
        default: break;// impossible to overlap...
    }
    return false;
}

bool CommandScheduler::_check_texture_read(uint64_t handle, uint level, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBufferToTextureCopyCommand: {
            auto other = static_cast<const BufferToTextureCopyCommand *>(command);
            return textures_overlap(handle, level, other->texture(), other->level());
        }
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            auto other = static_cast<const ShaderDispatchCommand *>(command);
            other->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                    if (auto usage = other->kernel().variable_usage(argument.variable_uid);
                        (usage == Usage::WRITE || usage == Usage::READ_WRITE) &&
                        textures_overlap(handle, level, argument.handle, argument.level)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                }
            });
            return overlap;
        }
        case Command::Tag::ETextureUploadCommand: {
            auto other = static_cast<const TextureUploadCommand *>(command);
            return textures_overlap(handle, level, other->handle(), other->level());
        }
        case Command::Tag::ETextureCopyCommand: {
            auto other = static_cast<const TextureCopyCommand *>(command);
            return textures_overlap(handle, level, other->dst_handle(), other->dst_level());
        }
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_texture_write(uint64_t handle, uint level, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBufferToTextureCopyCommand: {
            auto other = static_cast<const BufferToTextureCopyCommand *>(command);
            return textures_overlap(handle, level, other->texture(), other->level());
        }
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            auto other = static_cast<const ShaderDispatchCommand *>(command);
            other->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                    if (textures_overlap(handle, level, argument.handle, argument.level)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                    if (_device->is_resource_in_bindless_array(argument.handle, handle)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                }
            });
            return overlap;
        }
        case Command::Tag::ETextureUploadCommand: {
            auto other = static_cast<const TextureUploadCommand *>(command);
            return textures_overlap(handle, level, other->handle(), other->level());
        }
        case Command::Tag::ETextureDownloadCommand: {
            auto other = static_cast<const TextureDownloadCommand *>(command);
            return textures_overlap(handle, level, other->handle(), other->level());
        }
        case Command::Tag::ETextureCopyCommand: {
            auto other = static_cast<const TextureCopyCommand *>(command);
            return textures_overlap(handle, level, other->dst_handle(), other->dst_level());
        }
        case Command::Tag::ETextureToBufferCopyCommand: {
            auto other = static_cast<const TextureToBufferCopyCommand *>(command);
            return textures_overlap(handle, level, other->texture(), other->level());
        }
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_mesh_write(uint64_t handle, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EAccelBuildCommand: return true;
        case Command::Tag::EMeshBuildCommand:
            return handle == static_cast<const MeshBuildCommand *>(command)->handle();
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_accel_read(uint64_t handle, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EAccelBuildCommand:
            return handle == static_cast<const AccelBuildCommand *>(command)->handle();
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            auto other = static_cast<const ShaderDispatchCommand *>(command);
            other->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
                    if (auto usage = other->kernel().variable_usage(argument.variable_uid);
                        (usage == Usage::WRITE || usage == Usage::READ_WRITE) && handle == argument.handle) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                }
            });
            return overlap;
        }
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_accel_write(uint64_t handle, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EAccelBuildCommand:
            return handle == static_cast<const AccelBuildCommand *>(command)->handle();
        case Command::Tag::EMeshBuildCommand: return true;
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            static_cast<const ShaderDispatchCommand *>(command)->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
                    overlap = true;
                    return;// do not add the edge twice
                }
            });
            return overlap;
        }
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_bindless_array_read(uint64_t handle, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBufferUploadCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const BufferUploadCommand *>(command)->handle());
        case Command::Tag::EBufferCopyCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const BufferCopyCommand *>(command)->dst_handle());
        case Command::Tag::EBufferToTextureCopyCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const BufferToTextureCopyCommand *>(command)->texture());
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            auto other = static_cast<const ShaderDispatchCommand *>(command);
            other->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                    if (auto usage = other->kernel().variable_usage(argument.variable_uid);
                        (usage == Usage::WRITE || usage == Usage::READ_WRITE) &&
                        _device->is_resource_in_bindless_array(handle, argument.handle)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                    if (auto usage = other->kernel().variable_usage(argument.variable_uid);
                        (usage == Usage::WRITE || usage == Usage::READ_WRITE) &&
                        _device->is_resource_in_bindless_array(handle, argument.handle)) {
                        overlap = true;
                        return;// do not add the edge twice
                    }
                }
            });
            return overlap;
        }
        case Command::Tag::ETextureUploadCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const TextureUploadCommand *>(command)->handle());
        case Command::Tag::ETextureCopyCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const TextureCopyCommand *>(command)->dst_handle());
        case Command::Tag::ETextureToBufferCopyCommand:
            return _device->is_resource_in_bindless_array(
                handle, static_cast<const TextureToBufferCopyCommand *>(command)->buffer());
        case Command::Tag::EBindlessArrayUpdateCommand:
            return handle == static_cast<const BindlessArrayUpdateCommand *>(command)->handle();
        default: break;
    }
    return false;
}

bool CommandScheduler::_check_bindless_array_write(uint64_t handle, const Command *command) const noexcept {
    switch (command->tag()) {
        case Command::Tag::EBindlessArrayUpdateCommand:
            return handle == static_cast<const BindlessArrayUpdateCommand *>(command)->handle();
        case Command::Tag::EShaderDispatchCommand: {
            auto overlap = false;
            static_cast<const ShaderDispatchCommand *>(command)->decode([&](auto argument) noexcept {
                using T = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                    overlap = true;
                    return;// do not add the edge twice
                }
            });
            return overlap;
        }
        default: break;
    }
    return false;
}

void CommandScheduler::visit(const BufferUploadCommand *command) noexcept {
    // writes into the buffer range
    auto curr = static_cast<uint>(_commands.size());
    auto buffer = command->handle();
    auto offset = command->offset();
    auto size = command->size();
    for (auto i : _pending_nodes) {
        if (_check_buffer_write(buffer, offset, size, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const BufferDownloadCommand *command) noexcept {
    // reads from the buffer range
    auto curr = static_cast<uint>(_commands.size());
    auto buffer = command->handle();
    auto offset = command->offset();
    auto size = command->size();
    for (auto i : _pending_nodes) {
        if (_check_buffer_read(buffer, offset, size, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const BufferCopyCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto src = command->src_handle();
    auto src_offset = command->src_offset();
    auto dst = command->dst_handle();
    auto dst_offset = command->dst_offset();
    auto size = command->size();
    for (auto i : _pending_nodes) {
        if (_check_buffer_read(src, src_offset, size, _commands[i]) ||
            _check_buffer_write(dst, dst_offset, size, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto texture = command->texture();
    auto texture_level = command->level();
    auto buffer = command->buffer();
    auto buffer_offset = command->buffer_offset();
    auto buffer_size = command->size().x * command->size().y * command->size().z *
                       pixel_storage_size(command->storage());
    for (auto i : _pending_nodes) {
        if (_check_buffer_read(buffer, buffer_offset, buffer_size, _commands[i]) ||
            _check_texture_write(texture, texture_level, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const ShaderDispatchCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto kernel = command->kernel();
    for (auto i : _pending_nodes) {
        auto overlap = false;
        command->decode([&](auto argument) noexcept {
            using T = std::decay_t<decltype(argument)>;
            if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                if (auto usage = kernel.variable_usage(argument.variable_uid); usage == Usage::READ) {
                    if (_check_buffer_read(argument.handle, argument.offset, argument.size, _commands[i])) {
                        overlap = true;
                        return;
                    }
                } else if (usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                    if (_check_buffer_write(argument.handle, argument.offset, argument.size, _commands[i])) {
                        overlap = true;
                        return;
                    }
                }
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                if (auto usage = kernel.variable_usage(argument.variable_uid); usage == Usage::READ) {
                    if (_check_texture_read(argument.handle, argument.level, _commands[i])) {
                        overlap = true;
                        return;
                    }
                } else if (usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                    if (_check_texture_write(argument.handle, argument.level, _commands[i])) {
                        overlap = true;
                        return;
                    }
                }
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                if (_check_bindless_array_read(argument.handle, _commands[i])) {
                    overlap = true;
                    return;
                }
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
                if (auto usage = kernel.variable_usage(argument.variable_uid); usage == Usage::READ) {
                    if (_check_accel_read(argument.handle, _commands[i])) {
                        overlap = true;
                        return;
                    }
                } else if (usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                    if (_check_accel_write(argument.handle, _commands[i])) {
                        overlap = true;
                        return;
                    }
                }
            }
        });
        if (overlap) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const TextureUploadCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto texture = command->handle();
    auto level = command->level();
    for (auto i : _pending_nodes) {
        if (_check_texture_write(texture, level, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const TextureDownloadCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto texture = command->handle();
    auto level = command->level();
    for (auto i : _pending_nodes) {
        if (_check_texture_read(texture, level, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const TextureCopyCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto src = command->src_handle();
    auto dst = command->dst_handle();
    auto src_level = command->src_level();
    auto dst_level = command->dst_level();
    for (auto i : _pending_nodes) {
        if (_check_texture_read(src, src_level, _commands[i]) ||
            _check_texture_write(dst, dst_level, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto texture = command->texture();
    auto texture_level = command->level();
    auto buffer = command->buffer();
    auto buffer_offset = command->buffer_offset();
    auto buffer_size = command->size().x * command->size().y * command->size().z *
                       pixel_storage_size(command->storage());
    for (auto i : _pending_nodes) {
        if (_check_texture_read(texture, texture_level, _commands[i]) ||
            _check_buffer_write(buffer, buffer_offset, buffer_size, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const AccelBuildCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto accel = command->handle();
    for (auto i : _pending_nodes) {
        if (_check_accel_write(accel, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const MeshBuildCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto mesh = command->handle();
    auto vertex_buffer = command->vertex_buffer();
    auto vertex_buffer_offset = command->vertex_buffer_offset();
    auto vertex_buffer_size = command->vertex_buffer_size();
    auto triangle_buffer = command->triangle_buffer();
    auto triangle_buffer_offset = command->triangle_buffer_offset();
    auto triangle_buffer_size = command->triangle_buffer_size();
    for (auto i : _pending_nodes) {
        if (_check_buffer_read(vertex_buffer, vertex_buffer_offset, vertex_buffer_size, _commands[i]) ||
            _check_buffer_read(triangle_buffer, triangle_buffer_offset, triangle_buffer_size, _commands[i]) ||
            _check_mesh_write(mesh, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

void CommandScheduler::visit(const BindlessArrayUpdateCommand *command) noexcept {
    auto curr = static_cast<uint>(_commands.size());
    auto array = command->handle();
    for (auto i : _pending_nodes) {
        if (_check_bindless_array_write(array, _commands[i])) {
            _edges[i].emplace_back(curr);
            _dependency_count[curr]++;
        }
    }
}

#pragma clang diagnostic pop

}// namespace luisa::compute
