#pragma once

#include <cstdint>

#include <core/hash.h>
#include <core/stl.h>
#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

class CommandReorderVisitor : public CommandVisitor {
public:
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
    struct Range {
        int64_t min;
        int64_t max;
        Range() {
            min = std::numeric_limits<int64_t>::min();
            max = std::numeric_limits<int64_t>::max();
        }
        Range(int64_t value) {
            min = value;
            max = value + 1;
        }
        Range(int64_t min, int64_t size)
            : min(min), max(size + min) {}
        bool collide(Range const &r) const;
        bool operator==(Range const &r) const;
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    struct RangeHash {
        uint64_t operator()(Range const &r) const {
            auto x = std::make_pair(r.min, r.max);
            return luisa::detail::murmur2_hash64(&x, sizeof(x), Hash64::default_seed);
        }
    };
    struct ResourceView {
        int64_t readLayer = -1;
        int64_t writeLayer = -1;
    };
    struct ResourceHandle {
        uint64_t handle;
        ResourceType type;
    };
    struct RangeHandle : public ResourceHandle {
        luisa::unordered_map<Range, ResourceView, RangeHash> views;
    };
    struct NoRangeHandle : public ResourceHandle {
        ResourceView view;
    };
    struct BindlessHandle : public ResourceHandle {
        ResourceView view;
    };

private:
    Pool<RangeHandle, false> rangePool;
    Pool<NoRangeHandle, false> noRangePool;
    Pool<BindlessHandle, false> bindlessHandlePool;
    luisa::unordered_map<uint64_t, RangeHandle *> resMap;
    luisa::unordered_map<uint64_t, NoRangeHandle *> noRangeResMap;
    luisa::unordered_map<uint64_t, BindlessHandle *> bindlessMap;
    int64_t bindlessMaxLayer = -1;
    int64_t maxMeshLevel = -1;
    int64_t maxAccelReadLevel = -1;
    int64_t maxAccelWriteLevel = -1;
    luisa::vector<CommandList> commandLists;
    size_t layerCount = 0;
    bool useBindlessInPass;
    bool useAccelInPass;
    ResourceHandle *GetHandle(
        uint64_t target_handle,
        ResourceType target_type);
    size_t GetLastLayerWrite(RangeHandle *handle, Range range);
    size_t GetLastLayerWrite(NoRangeHandle *handle);
    size_t GetLastLayerWrite(BindlessHandle *handle);
    size_t GetLastLayerRead(RangeHandle *handle, Range range);
    size_t GetLastLayerRead(NoRangeHandle *handle);
    size_t GetLastLayerRead(BindlessHandle *handle);
    void AddCommand(Command const *cmd, size_t layer);
    size_t SetRead(
        uint64_t handle,
        Range range,
        ResourceType type);
    size_t SetRead(
        ResourceHandle *handle,
        Range range);
    size_t SetWrite(
        ResourceHandle *handle,
        Range range);
    size_t SetWrite(
        uint64_t handle,
        Range range,
        ResourceType type);
    size_t SetRW(
        uint64_t read_handle,
        Range read_range,
        ResourceType read_type,
        uint64_t write_handle,
        Range write_range,
        ResourceType write_type);
    size_t SetMesh(
        uint64_t handle,
        uint64_t vb,
        Range vb_range,
        uint64_t ib,
        Range ib_range);

    void SetReadLayer(
        ResourceHandle *handle,
        Range range,
        int64_t layer);
    void SetWriteLayer(
        ResourceHandle *handle,
        Range range,
        int64_t layer);
    luisa::vector<std::pair<Range, ResourceHandle *>> dispatchReadHandle;
    luisa::vector<std::pair<Range, ResourceHandle *>> dispatchWriteHandle;
    Variable const *arg;
    Function f;
    size_t dispatchLayer;
    void AddDispatchHandle(
        uint64_t handle,
        ResourceType type,
        Range range,
        bool isWrite);
    Device::Interface *device = nullptr;

public:
    explicit CommandReorderVisitor(Device::Interface *device) noexcept;
    ~CommandReorderVisitor() noexcept = default;
    void clear() noexcept;
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{commandLists.data(), layerCount};
    }

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
    void visit(const AccelBuildCommand *command) noexcept override;

    // Mesh : conclude vertex and triangle buffers
    void visit(const MeshBuildCommand *command) noexcept override;

    void operator()(ShaderDispatchCommand::BufferArgument const &bf);
    void operator()(ShaderDispatchCommand::TextureArgument const &bf);
    void operator()(ShaderDispatchCommand::BindlessArrayArgument const &bf);
    void operator()(ShaderDispatchCommand::UniformArgument const &bf);
    void operator()(ShaderDispatchCommand::AccelArgument const &bf);
};

}// namespace luisa::compute
