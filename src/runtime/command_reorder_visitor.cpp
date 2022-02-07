//
// Created by ChenXin on 2021/12/7.
//

#include <core/mathematics.h>
#include <runtime/command_reorder_visitor.h>
#include <runtime/stream.h>

namespace luisa::compute {

thread_local luisa::vector<luisa::vector<CommandReorderVisitor::CommandRelation>> CommandReorderVisitor::_commandRelationData;

CommandReorderVisitor::ShaderDispatchCommandVisitor::ShaderDispatchCommandVisitor(
    CommandReorderVisitor::CommandRelation *commandRelation, Function *kernel) {
    this->commandRelation = commandRelation;
    this->kernel = kernel;
}

void CommandReorderVisitor::ShaderDispatchCommandVisitor::operator()(uint32_t vid,
                                                                     ShaderDispatchCommand::BufferArgument argument) {
    commandRelation->sourceSet.insert(
        CommandSource{argument.handle, argument.offset, size_t(-1),
                      kernel->variable_usage(vid), CommandType::BUFFER});
}

void CommandReorderVisitor::ShaderDispatchCommandVisitor::operator()(uint32_t vid,
                                                                     ShaderDispatchCommand::TextureArgument argument) {
    commandRelation->sourceSet.insert(
        CommandSource{argument.handle, size_t(-1), size_t(-1),
                      kernel->variable_usage(vid), CommandType::TEXTURE});
}

void CommandReorderVisitor::ShaderDispatchCommandVisitor::operator()(uint32_t vid,
                                                                     ShaderDispatchCommand::BindlessArrayArgument argument) {
    commandRelation->sourceSet.insert(
        CommandSource{argument.handle, size_t(-1), size_t(-1),
                      kernel->variable_usage(vid), CommandType::BINDLESS_ARRAY});
}

void CommandReorderVisitor::ShaderDispatchCommandVisitor::operator()(uint32_t vid,
                                                                     ShaderDispatchCommand::AccelArgument argument) {
    commandRelation->sourceSet.insert(
        CommandSource{argument.handle, size_t(-1), size_t(-1),
                      kernel->variable_usage(vid), CommandType::ACCEL});
}

bool CommandReorderVisitor::Overlap(CommandSource sourceA, CommandSource sourceB) {

    // no Usage::NONE by default
    if (sourceA.usage == Usage::NONE || sourceB.usage == Usage::NONE) {
        return false;
    }
    if (sourceA.usage == Usage::READ && sourceB.usage == Usage::READ) {
        return false;
    }

    if (sourceA.type == sourceB.type) {
        // the same type
        if (sourceA.handle != sourceB.handle) {
            return false;
        }
        if (sourceA.offset == size_t(-1) || sourceB.offset == size_t(-1) || sourceA.size == size_t(-1) || sourceB.size == size_t(-1)) {
            return true;
        }
        return (sourceA.offset >= sourceB.offset && sourceA.offset <= sourceB.offset + sourceB.size) ||
               (sourceA.offset + sourceA.size >= sourceB.offset && sourceA.offset + sourceA.size <= sourceB.offset + sourceB.size) ||
               (sourceB.offset >= sourceA.offset && sourceB.offset <= sourceA.offset + sourceA.size) ||
               (sourceB.offset + sourceB.size >= sourceA.offset && sourceB.offset + sourceB.size <= sourceA.offset + sourceA.size);
    }
    // sourceA will be set to higher level
    if (sourceB.type == CommandType::BINDLESS_ARRAY || sourceB.type == CommandType::ACCEL ||
        (sourceB.type == CommandType::MESH && (sourceA.type == CommandType::BUFFER || sourceA.type == CommandType::TEXTURE))) {
        std::swap(sourceA, sourceB);
    }
    if (sourceA.type == CommandType::ACCEL) {
        // accel - xxx
        if (sourceB.type == CommandType::MESH) {
            // accel - mesh
            return (sourceB.usage == Usage::WRITE || sourceB.usage == Usage::READ_WRITE) &&
                   device->is_mesh_in_accel(sourceA.handle, sourceB.handle);
        }
        if (sourceB.type == CommandType::BUFFER) {
            // accel - buffer
            return (sourceB.usage == Usage::WRITE || sourceB.usage == Usage::READ_WRITE) &&
                   device->is_buffer_in_accel(sourceA.handle, sourceB.handle);
        }
    } else if (sourceA.type == CommandType::MESH) {
        if (sourceB.type == CommandType::BUFFER) {
            // mesh - buffer
            return (sourceB.usage == Usage::WRITE || sourceB.usage == Usage::READ_WRITE) &&
                   (sourceB.handle == device->get_vertex_buffer_from_mesh(sourceA.handle) ||
                    sourceB.handle == device->get_triangle_buffer_from_mesh(sourceA.handle));
        }
    } else if (sourceA.type == CommandType::BINDLESS_ARRAY) {
        if (sourceB.type == CommandType::TEXTURE) {
            // bindless_array - texture
            return (sourceB.usage == Usage::WRITE || sourceB.usage == Usage::READ_WRITE) &&
                   device->is_texture_in_bindless_array(sourceA.handle, sourceB.handle);
        }
        if (sourceB.type == CommandType::BUFFER) {
            // bindless_array - buffer
            return (sourceB.usage == Usage::WRITE || sourceB.usage == Usage::READ_WRITE) &&
                   device->is_buffer_in_bindless_array(sourceA.handle, sourceB.handle);
        }
    }
    return false;
}

void CommandReorderVisitor::processNewCommandRelation(CommandReorderVisitor::CommandRelation &&commandRelation) noexcept {
    // check all relations by reversed index if they overlap with the command under processing
    int insertIndex = 0;
    for (int i = int(_commandRelationData.size()) - 1; i >= std::max(0, int(_commandRelationData.size()) - windowSize); --i) {
        for (auto &j : _commandRelationData[i]) {
            CommandRelation *lastCommandRelation = &j;
            bool overlap = false;
            // check every condition
            for (const auto &source : commandRelation.sourceSet) {
                for (const auto &lastSource : lastCommandRelation->sourceSet)
                    if (Overlap(lastSource, source)) {
                        overlap = true;
                        break;
                    }
                if (overlap)
                    break;
            }
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                insertIndex = i + 1;
                break;
            }
        }
        if (insertIndex > 0)
            break;
    }

    if (insertIndex == _commandRelationData.size())
        _commandRelationData.push_back({commandRelation});
    else
        _commandRelationData[insertIndex].push_back(commandRelation);
}

luisa::vector<CommandList> CommandReorderVisitor::getCommandLists() noexcept {
    luisa::vector<CommandList> ans;

    for (auto &i : _commandRelationData) {
        CommandList commandList;
        for (auto &j : i) {
            commandList.append(j.command->clone());
        }
        if (!commandList.empty()) {
            ans.push_back(std::move(commandList));
        }
    }

    _commandRelationData.clear();

    LUISA_VERBOSE_WITH_LOCATION("Reordered command list size = {}", ans.size());
    auto index = 0;
    for (auto &commandList : ans) {
        LUISA_VERBOSE_WITH_LOCATION(
            "List {} : size = {}",
            index++, commandList.size());
    }
    return ans;
}

void CommandReorderVisitor::visit(const BufferUploadCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), command->offset(), command->size(), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const BufferDownloadCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), command->offset(), command->size(), Usage::READ, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const BufferCopyCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->src_handle(), command->src_offset(), command->size(), Usage::READ, CommandType::BUFFER});
    commandRelation.sourceSet.insert(CommandSource{
        command->dst_handle(), command->dst_offset(), command->size(), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const BufferToTextureCopyCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->buffer(), size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});
    commandRelation.sourceSet.insert(CommandSource{
        command->texture(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const ShaderDispatchCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    Function kernel = command->kernel();
    ShaderDispatchCommandVisitor shaderDispatchCommandVisitor(&commandRelation, &kernel);
    command->decode(shaderDispatchCommandVisitor);

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const TextureUploadCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const TextureDownloadCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const TextureCopyCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->src_handle(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});
    commandRelation.sourceSet.insert(CommandSource{
        command->dst_handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const TextureToBufferCopyCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->texture(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});
    commandRelation.sourceSet.insert(CommandSource{
        command->buffer(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const BindlessArrayUpdateCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::BINDLESS_ARRAY});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const AccelUpdateCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::ACCEL});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const AccelBuildCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::ACCEL});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const MeshUpdateCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::MESH});
    // TODO : whether triangle and vertex are read ?
    uint64_t triangle_buffer = device->get_triangle_buffer_from_mesh(command->handle()),
             vertex_buffer = device->get_vertex_buffer_from_mesh(command->handle());
    commandRelation.sourceSet.insert(CommandSource{
        triangle_buffer, size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});
    commandRelation.sourceSet.insert(CommandSource{
        vertex_buffer, size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

void CommandReorderVisitor::visit(const MeshBuildCommand *command) noexcept {
    // generate CommandRelation data
    CommandRelation commandRelation{(Command *)command};

    // get source set
    commandRelation.sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::MESH});
    // TODO : whether triangle and vertex are read ?
    uint64_t triangle_buffer = device->get_triangle_buffer_from_mesh(command->handle()),
             vertex_buffer = device->get_vertex_buffer_from_mesh(command->handle());
    commandRelation.sourceSet.insert(CommandSource{
        triangle_buffer, size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});
    commandRelation.sourceSet.insert(CommandSource{
        vertex_buffer, size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});

    processNewCommandRelation(std::move(commandRelation));
}

CommandReorderVisitor::CommandReorderVisitor(Device::Interface *device, size_t size) {
    this->device = device;
    if (size > _commandRelationData.capacity())
        _commandRelationData.reserve(next_pow2(size));
}

}// namespace luisa::compute