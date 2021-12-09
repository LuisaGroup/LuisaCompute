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
    enum struct CommandType : uint32_t {
        BUFFER = 0x1u,
        TEXTURE = 0x2u,
        MESH = 0x4u,
        ACCEL = 0x8u,
        SHADER = 0x10u,
        BINDLESS_ARRAY = 0x20u,
    };

    struct CommandSource {
        uint64_t handle;
        size_t offset, size;
        Usage usage;
        CommandType type;

        bool operator==(const CommandSource &b) const {
            return handle == b.handle && offset == b.offset &&
                   size == b.size && usage == b.usage &&
                   type == b.type;
        }
    };
    struct HashCommandSource {
        size_t operator()(const CommandSource &source) const {
            return ((source.handle << 57) | (source.handle >> 7)) ^
                   ((source.offset << 43) | (source.offset >> 11)) ^
                   ((source.size << 31) | (source.size >> 33)) ^
                   ((uint32_t(source.usage) << 19) | (uint32_t(source.usage) >> 13)) ^
                   ((uint32_t(source.type) << 11) | (uint32_t(source.type) >> 21));
        }
    };

    struct CommandRelation {
        Command *command;
        std::vector<CommandRelation *> prev, next;
        std::unordered_set<CommandSource, HashCommandSource> sourceSet;
    };

    class ShaderDispatchCommandVisitor {
        CommandRelation *commandRelation;
        Function *kernel;

    public:
        explicit ShaderDispatchCommandVisitor(CommandRelation *commandRelation, Function *kernel);

        void operator()(uint32_t vid, ShaderDispatchCommand::BufferArgument argument);
        void operator()(uint32_t vid, ShaderDispatchCommand::TextureArgument argument);
        void operator()(uint32_t vid, ShaderDispatchCommand::BindlessArrayArgument argument);
        void operator()(uint32_t vid, ShaderDispatchCommand::AccelArgument argument);
        template<typename UnknownArgument>
        void operator()(uint32_t vid, UnknownArgument argument);
    };

private:
    Device::Interface *device;
    std::vector<CommandRelation *> _head, _tail;
    std::vector<CommandRelation> _commandRelationData;

private:
    inline bool Overlap(CommandSource sourceA, CommandSource sourceB);

    void processNewCommandRelation(CommandRelation *commandRelation) noexcept;

public:
    explicit CommandReorderVisitor(Device::Interface *device);

    [[nodiscard]] std::vector<CommandList> getCommandLists() noexcept;

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferToTextureCopyCommand *command) noexcept override;

    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override;

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override;
    void visit(const TextureDownloadCommand *command) noexcept override;
    void visit(const TextureCopyCommand *command) noexcept override;
    void visit(const TextureToBufferCopyCommand *command) noexcept override;

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override;

    // Accel : conclude meshes and their buffer
    void visit(const AccelUpdateCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;

    // Mesh : conclude vertex and triangle buffers
    void visit(const MeshUpdateCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
};

}// namespace luisa::compute

#endif//LUISACOMPUTE_COMMANDREORDERVISITOR_H
