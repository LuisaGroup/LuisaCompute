//
// Created by ChenXin on 2021/12/3.
//

#ifndef LUISACOMPUTE_COMMAND_REORDER_VISITOR_H
#define LUISACOMPUTE_COMMAND_REORDER_VISITOR_H

#include <runtime/device.h>
#include <vector>
#include <unordered_set>
#include <core/hash.h>

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
        inline auto hash() const {
            return ((handle << 57) | (handle >> 7)) ^
                   ((offset << 43) | (offset >> 11)) ^
                   ((size << 31) | (size >> 33)) ^
                   ((uint32_t(usage) << 19) | (uint32_t(usage) >> 13)) ^
                   ((uint32_t(type) << 11) | (uint32_t(type) >> 21));
        }
    };

    struct CommandRelation {
        Command *command;
        std::unordered_set<CommandSource, Hash64> sourceSet;
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
        void operator()(uint32_t vid, UnknownArgument argument) {
            // include ShaderDispatchCommand::UniformArgument
        }
    };

private:
    Device::Interface *device;
    int windowSize = 5;
    static thread_local std::vector<std::vector<CommandRelation>> _commandRelationData;

private:
    inline bool Overlap(CommandSource sourceA, CommandSource sourceB);

    void processNewCommandRelation(CommandRelation &&commandRelation) noexcept;

public:
    explicit CommandReorderVisitor(Device::Interface *device, size_t size);

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

#endif//LUISACOMPUTE_COMMAND_REORDER_VISITOR_H
