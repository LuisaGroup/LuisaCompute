//
// Created by 12437 on 2021/12/3.
//

#ifndef LUISACOMPUTE_COMMANDREORDERVISITOR_H
#define LUISACOMPUTE_COMMANDREORDERVISITOR_H

#include <runtime/device.h>

namespace luisa::compute {

namespace basicFunction {
inline bool Overlap(uint64_t handleA, size_t offsetA, size_t sizeA,
                    uint64_t handleB, size_t offsetB, size_t sizeB) {
    return handleA == handleB && sizeA > 0 && sizeB > 0 &&
           ((offsetA >= offsetB && offsetA <= offsetB + sizeB) ||
            (offsetA + sizeA >= offsetB && offsetA + sizeA <= offsetB + sizeB) ||
            (offsetB >= offsetA && offsetB <= offsetA + sizeA) ||
            (offsetB + sizeB >= offsetA && offsetB + sizeB <= offsetA + sizeA));
}

inline bool Include(uint64_t handleA, size_t offsetA, size_t sizeA,
                    uint64_t handleB, size_t offsetB, size_t sizeB) {
    return handleA == handleB && offsetA >= offsetB && offsetA + sizeA <= offsetB + sizeB;
}
}// namespace basicFunction

class CommandReorderVisitor : public CommandVisitor {
    struct CommandRelation {
        Command *command;
        std::vector<CommandRelation *> prev, next;
    };

    std::vector<CommandRelation *> _head, _tail;
    std::vector<CommandRelation> _commandRelationData;

public:
    [[nodiscard]] std::vector<CommandList> getCommandLists() noexcept {
        std::vector<CommandList> ans;
        // 1 command list per loop
        while (!_head.empty()) {
            CommandList commandList;
            size_t index = _head.size();
            // get all heads
            for (auto i = 0; i < index; ++i) {
                auto commandRelation = _head[i];
                commandList.append(commandRelation->command);
                // prepare next loop
                for (auto nextCommandRelation : commandRelation->next) {
                    nextCommandRelation->prev.erase(
                        std::find(nextCommandRelation->prev.begin(),
                                  nextCommandRelation->prev.end(),
                                  commandRelation));
                    if (nextCommandRelation->prev.empty()) {
                        _head.push_back(nextCommandRelation);
                    }
                }
            }
            _head.erase(_head.begin(), _head.begin() + index);
        }
        return ans;
    };

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            bool overlap = false;
            // check every condition
            // TODO
            if (auto *lastCommand = dynamic_cast<BufferUploadCommand *>(lastCommandRelation->command)) {
                overlap = basicFunction::Overlap(lastCommand->handle(), lastCommand->offset(), lastCommand->size(),
                                                 command->handle(), command->offset(), command->size()) &&
                          !basicFunction::Include(lastCommand->handle(), lastCommand->offset(), lastCommand->size(),
                                                  command->handle(), command->offset(), command->size());
            }
            if (auto *lastCommand = dynamic_cast<BufferDownloadCommand *>(lastCommandRelation->command)) {
                overlap = basicFunction::Overlap(lastCommand->handle(), lastCommand->offset(), lastCommand->size(),
                                                 command->handle(), command->offset(), command->size());
            }
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }

    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }

    // Accel : ray tracing resource, ignored
    void visit(const AccelUpdateCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const AccelBuildCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }

    // Mesh : ray tracing resource, ignored
    void visit(const MeshUpdateCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
    void visit(const MeshBuildCommand *command) noexcept override {
        // 1. data
        _commandRelationData.push_back({(Command *)command});
        CommandRelation *commandRelation = &_commandRelationData.back();

        // 2. check all tails if they overlap with the command under processing
        for (int i = 0; i < _tail.size(); ++i) {
            CommandRelation *lastCommandRelation = _tail[i];
            // TODO
            // assume it has relationship with all command now
            bool overlap = true;
            // if overlapping: add relation, tail command is not tail anymore
            if (overlap) {
                lastCommandRelation->next.push_back(commandRelation);
                commandRelation->prev.push_back(lastCommandRelation);
                _tail.erase(_tail.begin() + i);
                --i;
            }
        }

        // 3. new command must be a tail
        _tail.push_back(commandRelation);

        // 4. check new command if it's a head
        if (commandRelation->prev.empty()) {
            _head.push_back(commandRelation);
        }
    }
};

}// namespace luisa::compute

#endif//LUISACOMPUTE_COMMANDREORDERVISITOR_H
