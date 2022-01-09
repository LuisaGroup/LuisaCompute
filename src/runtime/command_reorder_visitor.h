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
            return memcmp(this, &b, sizeof(CommandSource)) == 0;
        }

        struct Hash {
            [[nodiscard]] auto operator()(CommandSource cs) const {
                return luisa::detail::xxh3_hash64(&cs, sizeof(cs), 19980810u);
            }
        };
    };

    struct CommandRelation {
        Command *command;
        luisa::unordered_set<CommandSource, CommandSource::Hash> sourceSet;
        explicit CommandRelation(Command *command) noexcept : command{command} {}
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
    static constexpr int windowSize = 16;
    static thread_local luisa::vector<luisa::vector<CommandRelation>> _commandRelationData;

private:
    inline bool Overlap(CommandSource sourceA, CommandSource sourceB);

    void processNewCommandRelation(CommandRelation &&commandRelation) noexcept;

public:
    explicit CommandReorderVisitor(Device::Interface *device, size_t size);

    [[nodiscard]] luisa::vector<CommandList> getCommandLists() noexcept;

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
