#pragma once
#include <runtime/device.h>
#include <core/hash.h>
#include <vstl/Common.h>
namespace luisa::compute {
class CommandReorderVisitor : public CommandVisitor {
    enum class ResourceRW : uint8_t {
        Read,
        Write
    };
    enum class ResourceType : uint8_t {
        Texture,
        Buffer,
        Mesh,
        Bindless,
        Accel
    };
    struct ResourceHandle {
        uint64_t handle;
        int64_t readLayer = -1;
        int64_t writeLayer = -1;
        ResourceType type;
    };
    vstd::Pool<ResourceHandle, true> handlePool;
    eastl::vector<ResourceHandle *> allocatedHandles;
    vstd::HashMap<uint64_t, ResourceHandle *> resMap;
    vstd::HashMap<uint64_t, ResourceHandle *> bindlessMap;
    vstd::HashMap<uint64_t, ResourceHandle *> accelMap;
    vstd::HashMap<uint64_t, ResourceHandle *> meshMap;
    eastl::vector<CommandList> commandLists;
    size_t layerCount = 0;
    ResourceHandle *GetHandle(
        uint64_t target_handle,
        ResourceType target_type);
    uint GetLastLayerWrite(ResourceHandle *handle) const;
    uint GetLastLayerRead(ResourceHandle *handle) const;
    void AddCommand(Command const *cmd, uint layer);
    uint SetRead(
        uint64_t handle,
        ResourceType type);
    uint SetWrite(
        uint64_t handle,
        ResourceType type);
    uint SetRW(
        uint64_t read_handle,
        ResourceType read_type,
        uint64_t write_handle,
        ResourceType write_type);
    uint SetMesh(
        uint64_t handle,
        uint64_t vb,
        uint64_t ib);
    eastl::vector<ResourceHandle *> dispatchReadHandle;
    eastl::vector<ResourceHandle *> dispatchWriteHandle;
    Variable const *arg;
    Function f;
    uint dispatchLayer;
    void AddDispatchHandle(
        uint64_t handle,
        ResourceType type,
        bool isWrite);
    Device::Interface *device = nullptr;

public:
    eastl::span<CommandList const> command_lists() const {
        return {commandLists.data(), layerCount};
    }
    CommandReorderVisitor(Device::Interface *device);
    ~CommandReorderVisitor();
    void clear();

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


    void operator()(uint uid, ShaderDispatchCommand::BufferArgument const &bf);
    void operator()(uint uid, ShaderDispatchCommand::TextureArgument const &bf);
    void operator()(uint uid, ShaderDispatchCommand::BindlessArrayArgument const &bf);
    void operator()(uint uid, vstd::span<std::byte const> bf);
    void operator()(uint uid, ShaderDispatchCommand::AccelArgument const &bf);
};
}// namespace luisa::compute::maxwell