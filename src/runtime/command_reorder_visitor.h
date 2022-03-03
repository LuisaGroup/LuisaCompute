//
// Created by ChenXin on 2021/12/3.
//
#pragma once

#include <runtime/device.h>
#include <vector>
#include <core/hash.h>

namespace luisa::compute {

class CommandReorderVisitor : public MutableCommandVisitor {
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
            [[nodiscard]] auto operator()(CommandSource const& cs) const {
                return luisa::detail::xxh3_hash64(&cs, sizeof(cs), 19980810u);
            }
        };
    };

    struct CommandRelation {
        Command *command;
        luisa::unordered_set<CommandSource, CommandSource::Hash> sourceSet;
        explicit CommandRelation(Command *command) noexcept : command{command} {}
        void clear() {
            sourceSet.clear();
        }
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
    luisa::vector<luisa::vector<CommandRelation*>> _commandRelationData;
    Pool<CommandRelation> relationPool;
    luisa::vector<CommandRelation *> pooledRelations;

private:
    CommandRelation *allocate_relation(Command *cmd);
    void deallocate_relation(CommandRelation * v);

    inline bool Overlap(CommandSource sourceA, CommandSource sourceB);

    void processNewCommandRelation(CommandRelation *commandRelation) noexcept;

public:
    explicit CommandReorderVisitor(Device::Interface *device);
    ~CommandReorderVisitor();
    void reserve(size_t size);
    [[nodiscard]] luisa::vector<CommandList> getCommandLists() noexcept;
    void clear() noexcept;
    // Buffer : resource
    void visit(BufferUploadCommand *command) noexcept override;
    void visit(BufferDownloadCommand *command) noexcept override;
    void visit(BufferCopyCommand *command) noexcept override;
    void visit(BufferToTextureCopyCommand *command) noexcept override;

    // Shader : function, read/write multi resources
    void visit(ShaderDispatchCommand *command) noexcept override;

    // Texture : resource
    void visit(TextureUploadCommand *command) noexcept override;
    void visit(TextureDownloadCommand *command) noexcept override;
    void visit(TextureCopyCommand *command) noexcept override;
    void visit(TextureToBufferCopyCommand *command) noexcept override;

    // BindlessArray : read multi resources
    void visit(BindlessArrayUpdateCommand *command) noexcept override;

    // Accel : conclude meshes and their buffer
    void visit(AccelUpdateCommand *command) noexcept override;
    void visit(AccelBuildCommand *command) noexcept override;

    // Mesh : conclude vertex and triangle buffers
    void visit(MeshUpdateCommand *command) noexcept override;
    void visit(MeshBuildCommand *command) noexcept override;
};

}// namespace luisa::compute
