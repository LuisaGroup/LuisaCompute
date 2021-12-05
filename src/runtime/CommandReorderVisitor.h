//
// Created by 12437 on 2021/12/3.
//

#ifndef LUISACOMPUTE_COMMANDREORDERVISITOR_H
#define LUISACOMPUTE_COMMANDREORDERVISITOR_H

#include <runtime/device.h>
#include <vector>
#include <unordered_set>

namespace luisa::compute {

class CommandReorderVisitor : public CommandVisitor {
    struct CommandSource {
        uint64_t handle;
        size_t offset, size;
        Usage usage;
    };

    struct CommandRelation {
        Command *command;
        std::vector<CommandRelation *> prev, next;
        std::unordered_set<CommandSource> sourceSet;
    };

private:
    std::vector<CommandRelation *> _head, _tail;
    std::vector<CommandRelation> _commandRelationData;

private:
    static inline bool Overlap(const CommandSource &sourceA, const CommandSource &sourceB) {
        if (sourceA.handle != sourceB.handle)
            return false;
        if (sourceA.usage == Usage::READ && sourceB.usage == Usage::READ)
            return false;
        if (sourceA.offset == size_t(-1) || sourceB.offset == size_t(-1) || sourceA.size == size_t(-1) || sourceB.size == size_t(-1))
            return true;
        return (sourceA.offset >= sourceB.offset && sourceA.offset <= sourceB.offset + sourceB.size) ||
               (sourceA.offset + sourceA.size >= sourceB.offset && sourceA.offset + sourceA.size <= sourceB.offset + sourceB.size) ||
               (sourceB.offset >= sourceA.offset && sourceB.offset <= sourceA.offset + sourceA.size) ||
               (sourceB.offset + sourceB.size >= sourceA.offset && sourceB.offset + sourceB.size <= sourceA.offset + sourceA.size);
    }

    void processNewCommandRelation(CommandRelation *commandRelation) noexcept {
        // 1. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            bool overlap = false;
            // check every condition
            for (const auto &source : commandRelation->sourceSet) {
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

public:
    [[nodiscard]] std::vector<CommandList> getCommandLists() noexcept {
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
    };

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->handle(), command->offset(), command->size(), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->handle(), command->offset(), command->size(), Usage::READ});

        processNewCommandRelation(commandRelation);
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->src_handle(), command->src_offset(), command->size(), Usage::READ});
        commandRelation->sourceSet.insert(CommandSource{
            command->dst_handle(), command->dst_offset(), command->size(), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->buffer(), size_t(-1), size_t(-1), Usage::READ});
        commandRelation->sourceSet.insert(CommandSource{
            command->texture(), size_t(-1), size_t(-1), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }

    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->handle(), size_t(-1), size_t(-1), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->handle(), size_t(-1), size_t(-1), Usage::READ});

        processNewCommandRelation(commandRelation);
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->src_handle(), size_t(-1), size_t(-1), Usage::READ});
        commandRelation->sourceSet.insert(CommandSource{
            command->dst_handle(), size_t(-1), size_t(-1), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        commandRelation->sourceSet.insert(CommandSource{
            command->texture(), size_t(-1), size_t(-1), Usage::READ});
        commandRelation->sourceSet.insert(CommandSource{
            command->buffer(), size_t(-1), size_t(-1), Usage::WRITE});

        processNewCommandRelation(commandRelation);
    }

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }

    // Accel : ray tracing resource, ignored
    void visit(const AccelUpdateCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }
    void visit(const AccelBuildCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }

    // Mesh : ray tracing resource, ignored
    void visit(const MeshUpdateCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }
    void visit(const MeshBuildCommand *command) noexcept override {
        // save data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // get source set
        // TODO

        processNewCommandRelation(commandRelation);
    }
};

}// namespace luisa::compute

#endif//LUISACOMPUTE_COMMANDREORDERVISITOR_H
