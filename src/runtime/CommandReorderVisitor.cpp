//
// Created by 12437 on 2021/12/7.
//

#include <runtime/CommandReorderVisitor.h>
#include <runtime/stream.h>

namespace luisa::compute {

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

void CommandReorderVisitor::ShaderDispatchCommandVisitor::operator()(uint32_t vid,
                                                                     ShaderDispatchCommand::UniformArgument argument) {
    return;
}

bool CommandReorderVisitor::Overlap(const CommandSource &sourceA, const CommandSource &sourceB) {

    if (sourceA.handle != sourceB.handle)
        return false;
    //    // no Usage::NONE by default
    //    if (sourceA.usage == Usage::NONE || sourceB.usage == Usage::NONE)
    //        return false;
    if (sourceA.usage == Usage::READ && sourceB.usage == Usage::READ)
        return false;
    if (sourceA.offset == size_t(-1) || sourceB.offset == size_t(-1) || sourceA.size == size_t(-1) || sourceB.size == size_t(-1))
        return true;
    return (sourceA.offset >= sourceB.offset && sourceA.offset <= sourceB.offset + sourceB.size) ||
           (sourceA.offset + sourceA.size >= sourceB.offset && sourceA.offset + sourceA.size <= sourceB.offset + sourceB.size) ||
           (sourceB.offset >= sourceA.offset && sourceB.offset <= sourceA.offset + sourceA.size) ||
           (sourceB.offset + sourceB.size >= sourceA.offset && sourceB.offset + sourceB.size <= sourceA.offset + sourceA.size);
}

void CommandReorderVisitor::processNewCommandRelation(CommandReorderVisitor::CommandRelation *commandRelation) noexcept {
    // 1. check all tails if they overlap with the command under processing
    for (int i = 0; i < _tail.size(); ++i) {
        CommandRelation *lastCommandRelation = _tail[i];
        bool overlap = false;
        // check every condition
        for (const auto &source : commandRelation->sourceSet) {
            for (const auto &lastSource : lastCommandRelation->sourceSet)
                // TODO : check command type
                if (Overlap(lastSource, source)) {
                    overlap = true;
                    break;
                }
            if (overlap)
                break;
        }
        // if overlapping: add relation, tail command is not tail anymore
        if (overlap) {
            lastCommandRelation->next.push_back(commandRelation);
            commandRelation->prev.push_back(lastCommandRelation);
            _tail.erase(_tail.begin() + i);
            --i;
        }
    }

    // 2. new command must be a tail
    _tail.push_back(commandRelation);

    // 3. check new command if it's a head
    if (commandRelation->prev.empty()) {
        _head.push_back(commandRelation);
    }
}

std::vector<CommandList> CommandReorderVisitor::getCommandLists() noexcept {
    std::vector<CommandList> ans;
    // 1 command list per loop
    while (!_head.empty()) {
        CommandList commandList;
        size_t index = _head.size();
        // get all heads
        for (size_t i = 0; i < index; ++i) {
            auto commandRelation = _head[i];
            commandList.append(commandRelation->command->clone());
            // prepare next loop
            for (auto nextCommandRelation : commandRelation->next) {
                nextCommandRelation->prev.erase(
                    std::find(nextCommandRelation->prev.begin(),
                              nextCommandRelation->prev.end(),
                              commandRelation));
                // prev empty means it becomes a new head
                if (nextCommandRelation->prev.empty()) {
                    _head.push_back(nextCommandRelation);
                }
            }
        }
        ans.push_back(std::move(commandList));
        _head.erase(_head.begin(), _head.begin() + index);
    }
    return ans;
}

void CommandReorderVisitor::visit(const BufferUploadCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), command->offset(), command->size(), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const BufferDownloadCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), command->offset(), command->size(), Usage::READ, CommandType::BUFFER});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const BufferCopyCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->src_handle(), command->src_offset(), command->size(), Usage::READ, CommandType::BUFFER});
    commandRelation->sourceSet.insert(CommandSource{
        command->dst_handle(), command->dst_offset(), command->size(), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const BufferToTextureCopyCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->buffer(), size_t(-1), size_t(-1), Usage::READ, CommandType::BUFFER});
    commandRelation->sourceSet.insert(CommandSource{
        command->texture(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const ShaderDispatchCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    ShaderDispatchCommandVisitor shaderDispatchCommandVisitor(commandRelation, &command->kernel());
    command->decode(shaderDispatchCommandVisitor);

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const TextureUploadCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const TextureDownloadCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const TextureCopyCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->src_handle(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});
    commandRelation->sourceSet.insert(CommandSource{
        command->dst_handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::TEXTURE});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const TextureToBufferCopyCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->texture(), size_t(-1), size_t(-1), Usage::READ, CommandType::TEXTURE});
    commandRelation->sourceSet.insert(CommandSource{
        command->buffer(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::BUFFER});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const BindlessArrayUpdateCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::BINDLESS_ARRAY});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const AccelUpdateCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::ACCEL});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const AccelBuildCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE, CommandType::ACCEL});

    processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const MeshUpdateCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    uint64_t triangle_buffer = Device::Interface::get_triangle_buffer_from_mesh(command->handle()),
             vertex_buffer = Device::Interface::get_vertex_buffer_from_mesh(command->handle());
    commandRelation->sourceSet.insert(CommandSource{
        triangle_buffer, size_t(-1), size_t(-1), Usage::READ});
    commandRelation->sourceSet.insert(CommandSource{
        vertex_buffer, size_t(-1), size_t(-1), Usage::READ});
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE})

        processNewCommandRelation(commandRelation);
}

void CommandReorderVisitor::visit(const MeshBuildCommand *command) noexcept {
    // save data
    _commandRelationData.push_back({(Command *)command});
    CommandRelation *commandRelation = &_commandRelationData.back();

    // get source set
    uint64_t triangle_buffer = Device::Interface::get_triangle_buffer_from_mesh(command->handle()),
             vertex_buffer = Device::Interface::get_vertex_buffer_from_mesh(command->handle());
    commandRelation->sourceSet.insert(CommandSource{
        triangle_buffer, size_t(-1), size_t(-1), Usage::READ});
    commandRelation->sourceSet.insert(CommandSource{
        vertex_buffer, size_t(-1), size_t(-1), Usage::READ});
    commandRelation->sourceSet.insert(CommandSource{
        command->handle(), size_t(-1), size_t(-1), Usage::WRITE});

    processNewCommandRelation(commandRelation);
}

}// namespace luisa::compute