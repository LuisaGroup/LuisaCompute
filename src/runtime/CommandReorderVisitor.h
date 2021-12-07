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
        BUFFER = 1u,
        TEXTURE = 2u,
        MESH = 3u,
        ACCEL = 4u,
        SHADER = 5u,
        BINDLESS_ARRAY = 6u,
    };

    struct CommandSource {
        uint64_t handle;
        size_t offset, size;
        Usage usage;
        CommandType type;
    };

    struct CommandRelation {
        Command *command;
        std::vector<CommandRelation *> prev, next;
        std::unordered_set<CommandSource> sourceSet;
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
        void operator()(uint32_t vid, ShaderDispatchCommand::UniformArgument argument);
    };

private:
    std::vector<CommandRelation *> _head, _tail;
    std::vector<CommandRelation> _commandRelationData;

private:
    static inline bool Overlap(const CommandSource &sourceA, const CommandSource &sourceB);

    void processNewCommandRelation(CommandRelation *commandRelation) noexcept;

public:
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

    // Accel : ray tracing resource, ignored
    void visit(const AccelUpdateCommand *command) noexcept override;
    void visit(const AccelBuildCommand *command) noexcept override;

    // Mesh : ray tracing resource, ignored
    void visit(const MeshUpdateCommand *command) noexcept override;
    void visit(const MeshBuildCommand *command) noexcept override;
};

}// namespace luisa::compute

#endif//LUISACOMPUTE_COMMANDREORDERVISITOR_H
