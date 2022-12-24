#pragma once

#include <runtime/device.h>
#include <core/stl/hash.h>
#include <cstdint>
#include <vstl/common.h>
#include <runtime/command.h>
#include <runtime/buffer.h>
#include <raster/raster_scene.h>
namespace luisa::compute {
/*
struct FuncTable{
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept;
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept;
}
*/
template<typename FuncTable, bool supportConcurrentCopy>
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
        bool collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    struct RangeHash {
        size_t operator()(Range const &r) const {
            return hash64(this, sizeof(Range), 9527ull);
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
        vstd::unordered_map<Range, ResourceView, RangeHash> views;
    };
    struct NoRangeHandle : public ResourceHandle {
        ResourceView view;
    };
    struct BindlessHandle : public ResourceHandle {
        ResourceView view;
    };

private:
    static Range CopyRange(int64_t offset, int64_t size) {
        if constexpr (supportConcurrentCopy) {
            return Range(offset, size);
        } else {
            return Range();
        }
    }
    template<typename Func>
        requires(std::is_invocable_v<Func, CommandReorderVisitor::ResourceView const &>)
    void IterateMap(Func &&func, RangeHandle &handle, Range const &range) {
        for (auto &&r : handle.views) {
            if (r.first.collide(range)) {
                func(r.second);
            }
        }
    }
    vstd::Pool<RangeHandle, true> rangePool;
    vstd::Pool<NoRangeHandle, true> noRangePool;
    vstd::Pool<BindlessHandle, true> bindlessHandlePool;
    vstd::unordered_map<uint64_t, RangeHandle *> resMap;
    vstd::unordered_map<uint64_t, NoRangeHandle *> noRangeResMap;
    vstd::unordered_map<uint64_t, BindlessHandle *> bindlessMap;
    int64_t bindlessMaxLayer = -1;
    int64_t maxMeshLevel = -1;
    int64_t maxBufferReadLevel = -1;
    int64_t maxAccelReadLevel = -1;
    int64_t maxAccelWriteLevel = -1;
    vstd::vector<vstd::fixed_vector<Command const *, 4>> commandLists;
    size_t layerCount = 0;
    bool useBindlessInPass;
    bool useAccelInPass;
    ResourceHandle *GetHandle(
        uint64_t target_handle,
        ResourceType target_type) {
        auto func = [&](auto &&map, auto &&pool) {
            auto tryResult = map.try_emplace(
                target_handle);
            auto &&value = tryResult.first->second;
            if (tryResult.second) {
                value = pool.New();
                value->handle = target_handle;
                value->type = target_type;
            }
            return value;
        };
        switch (target_type) {
            case ResourceType::Bindless:
                return func(bindlessMap, bindlessHandlePool);
            case ResourceType::Mesh:
            case ResourceType::Accel:
                return func(noRangeResMap, noRangePool);
            default:
                return func(resMap, rangePool);
        }
    }
    // Texture, Buffer
    size_t GetLastLayerWrite(RangeHandle *handle, Range range) {
        size_t layer = 0;
        IterateMap(
            [&](auto &&handle) {
                layer = std::max<int64_t>(layer, std::max<int64_t>(handle.readLayer + 1, handle.writeLayer + 1));
            },
            *handle,
            range);
        if (bindlessMaxLayer >= layer) {
            for (auto &&i : bindlessMap) {
                if (funcTable.is_res_in_bindless(i.first, handle->handle)) {
                    layer = std::max<int64_t>(layer, i.second->view.readLayer + 1);
                }
            }
        }
        if (handle->type == ResourceType::Buffer) {
            layer = std::max<int64_t>(layer, maxBufferReadLevel + 1);
        }
        return layer;
    }
    // Mesh, Accel
    size_t GetLastLayerWrite(NoRangeHandle *handle) {
        size_t layer = std::max<int64_t>(handle->view.readLayer + 1, handle->view.writeLayer + 1);

        switch (handle->type) {
            case ResourceType::Mesh: {
                auto maxAccelLevel = std::max(maxAccelReadLevel, maxAccelWriteLevel);
                layer = std::max<int64_t>(layer, maxAccelLevel + 1);
            } break;
            case ResourceType::Accel: {
                auto maxAccelLevel = std::max(maxAccelReadLevel, maxAccelWriteLevel);
                layer = std::max<int64_t>(layer, maxAccelLevel + 1);
                layer = std::max<int64_t>(layer, maxMeshLevel + 1);
            } break;
            default: break;
        }
        return layer;
    }
    // Bindless
    size_t GetLastLayerWrite(BindlessHandle *handle) {
        return std::max<int64_t>(handle->view.readLayer + 1, handle->view.writeLayer + 1);
    }
    size_t GetLastLayerRead(RangeHandle *handle, Range range) {
        size_t layer = 0;
        IterateMap(
            [&](auto &&handle) {
                layer = std::max<int64_t>(layer, handle.writeLayer + 1);
            },
            *handle,
            range);
        return layer;
    }
    size_t GetLastLayerRead(NoRangeHandle *handle) {
        size_t layer = handle->view.writeLayer + 1;
        if (handle->type == ResourceType::Accel) {
            layer = std::max<int64_t>(layer, maxAccelWriteLevel + 1);
        }
        return layer;
    }
    size_t GetLastLayerRead(BindlessHandle *handle) {
        return handle->view.writeLayer + 1;
    }
    void AddCommand(Command const *cmd, size_t layer) {
        if (commandLists.size() <= layer) {
            commandLists.resize(layer + 1);
        }
        layerCount = std::max<int64_t>(layerCount, layer + 1);
        commandLists[layer].push_back(cmd);
    }
    size_t SetRead(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto srcHandle = GetHandle(
            handle,
            type);
        return SetRead(srcHandle, range);
    }
    size_t SetRead(
        ResourceHandle *srcHandle,
        Range range) {
        size_t layer = 0;
        switch (srcHandle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(srcHandle);
                layer = GetLastLayerRead(handle);
                handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(srcHandle);
                layer = GetLastLayerRead(handle);
                handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(srcHandle);
                layer = GetLastLayerRead(handle, range);
                auto ite = handle->views.try_emplace(range);
                if (ite.second)
                    ite.first->second.readLayer = std::max<int64_t>(ite.first->second.readLayer, layer);
                else
                    ite.first->second.readLayer = layer;
            } break;
        }
        return layer;
    }
    size_t SetWrite(
        ResourceHandle *dstHandle,
        Range range) {
        size_t layer = 0;
        switch (dstHandle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dstHandle);
                layer = GetLastLayerWrite(handle);
                handle->view.writeLayer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dstHandle);
                layer = GetLastLayerWrite(handle);
                handle->view.writeLayer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dstHandle);
                layer = GetLastLayerWrite(handle, range);
                auto ite = handle->views.try_emplace(range);
                ite.first->second.writeLayer = layer;
            } break;
        }

        return layer;
    }
    size_t SetWrite(
        uint64_t handle,
        Range range,
        ResourceType type) {
        auto dstHandle = GetHandle(
            handle,
            type);
        return SetWrite(dstHandle, range);
    }
    size_t SetRW(
        uint64_t read_handle,
        Range read_range,
        ResourceType read_type,
        uint64_t write_handle,
        Range write_range,
        ResourceType write_type) {

        size_t layer = 0;
        auto srcHandle = GetHandle(
            read_handle,
            read_type);
        auto dstHandle = GetHandle(
            write_handle,
            write_type);
        luisa::move_only_function<void()> setReadLayer;
        switch (read_type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(srcHandle);
                layer = GetLastLayerRead(handle);
                setReadLayer = [&]() {
                    auto handle = static_cast<NoRangeHandle *>(srcHandle);
                    handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
                };
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(srcHandle);
                layer = GetLastLayerRead(handle);
                setReadLayer = [&]() {
                    auto handle = static_cast<BindlessHandle *>(srcHandle);
                    handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
                };
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(srcHandle);
                layer = GetLastLayerRead(handle, read_range);
                auto ite = handle->views.try_emplace(read_range);
                if (ite.second) {
                    auto viewPtr = &ite.first->second;
                    setReadLayer = [viewPtr, &layer]() {
                        viewPtr->readLayer = std::max<int64_t>(viewPtr->readLayer, layer);
                    };
                } else {
                    auto viewPtr = &ite.first->second;
                    setReadLayer = [viewPtr, &layer]() {
                        viewPtr->readLayer = layer;
                    };
                }

            } break;
        }

        switch (write_type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dstHandle);
                layer = std::max<int64_t>(layer, GetLastLayerWrite(handle));
                handle->view.writeLayer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dstHandle);
                layer = std::max<int64_t>(layer, GetLastLayerWrite(handle));
                handle->view.writeLayer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dstHandle);
                layer = std::max<int64_t>(layer, GetLastLayerWrite(handle, write_range));
                auto ite = handle->views.try_emplace(write_range);
                ite.first->second.writeLayer = layer;
            } break;
        }
        setReadLayer();
        return layer;
    }
    size_t SetMesh(
        uint64_t handle,
        uint64_t vb,
        Range vb_range,
        uint64_t ib,
        Range ib_range) {

        auto vbHandle = GetHandle(
            vb,
            ResourceType::Buffer);
        auto meshHandle = GetHandle(
            handle,
            ResourceType::Mesh);
        auto layer = GetLastLayerRead(static_cast<RangeHandle *>(vbHandle), vb_range);
        layer = std::max<int64_t>(layer, GetLastLayerWrite(static_cast<NoRangeHandle *>(meshHandle)));
        auto SetHandle = [](auto &&handle, auto &&range, auto layer) {
            auto ite = handle->views.try_emplace(range);
            if (ite.second)
                ite.first->second.readLayer = layer;
            else
                ite.first->second.readLayer = std::max<int64_t>(layer, ite.first->second.readLayer);
        };
        auto ibHandle = GetHandle(
            ib,
            ResourceType::Buffer);
        auto rangeHandle = static_cast<RangeHandle *>(ibHandle);
        layer = std::max<int64_t>(layer, GetLastLayerRead(rangeHandle, ib_range));
        SetHandle(rangeHandle, ib_range, layer);
        SetHandle(static_cast<RangeHandle *>(vbHandle), vb_range, layer);
        static_cast<NoRangeHandle *>(meshHandle)->view.writeLayer = layer;
        maxMeshLevel = std::max<int64_t>(maxMeshLevel, layer);
        return layer;
    }
    size_t SetAABB(
        uint64_t handle,
        uint64_t aabb_buffer,
        Range aabb_range) {
        auto vbHandle = GetHandle(
            aabb_buffer,
            ResourceType::Buffer);
        auto meshHandle = GetHandle(
            handle,
            ResourceType::Mesh);
        auto layer = GetLastLayerRead(static_cast<RangeHandle *>(vbHandle), aabb_range);
        layer = std::max<int64_t>(layer, GetLastLayerWrite(static_cast<NoRangeHandle *>(meshHandle)));
        auto SetHandle = [](auto &&handle, auto &&range, auto layer) {
            auto ite = handle->views.try_emplace(range);
            if (ite.second)
                ite.first->second.readLayer = layer;
            else
                ite.first->second.readLayer = std::max<int64_t>(layer, ite.first->second.readLayer);
        };
        SetHandle(static_cast<RangeHandle *>(vbHandle), aabb_range, layer);
        static_cast<NoRangeHandle *>(meshHandle)->view.writeLayer = layer;
        maxMeshLevel = std::max<int64_t>(maxMeshLevel, layer);
        return layer;
    }

    void SetReadLayer(
        ResourceHandle *srcHandle,
        Range range,
        int64_t layer) {
        switch (srcHandle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(srcHandle);
                handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(srcHandle);
                handle->view.readLayer = std::max<int64_t>(layer, handle->view.readLayer);
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(srcHandle);
                auto emplaceResult = handle->views.try_emplace(range);
                if (emplaceResult.second) {
                    emplaceResult.first->second.readLayer = layer;
                } else {
                    emplaceResult.first->second.readLayer = std::max<int64_t>(emplaceResult.first->second.readLayer, layer);
                }
            } break;
        }
    }
    void SetWriteLayer(
        ResourceHandle *dstHandle,
        Range range,
        int64_t layer) {
        switch (dstHandle->type) {
            case ResourceType::Mesh:
            case ResourceType::Accel: {
                auto handle = static_cast<NoRangeHandle *>(dstHandle);
                handle->view.writeLayer = layer;
            } break;
            case ResourceType::Bindless: {
                auto handle = static_cast<BindlessHandle *>(dstHandle);
                handle->view.writeLayer = layer;
            } break;
            default: {
                auto handle = static_cast<RangeHandle *>(dstHandle);
                auto emplaceResult = handle->views.try_emplace(range);
                emplaceResult.first->second.writeLayer = layer;
            } break;
        }
    }
    vstd::vector<std::pair<Range, ResourceHandle *>> dispatchReadHandle;
    vstd::vector<std::pair<Range, ResourceHandle *>> dispatchWriteHandle;
    size_t argIndex;
    uint64_t shaderHandle;
    size_t dispatchLayer;
    void AddDispatchHandle(
        uint64_t handle,
        ResourceType type,
        Range range,
        bool isWrite) {
        if (isWrite) {
            auto h = GetHandle(
                handle,
                type);
            switch (type) {
                case ResourceType::Accel:
                case ResourceType::Mesh:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerWrite(static_cast<NoRangeHandle *>(h)));
                    break;
                case ResourceType::Buffer:
                case ResourceType::Texture:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerWrite(static_cast<RangeHandle *>(h), range));
                    break;
                case ResourceType::Bindless:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerWrite(static_cast<BindlessHandle *>(h)));
                    break;
            }
            dispatchWriteHandle.emplace_back(range, h);
        } else {
            auto h = GetHandle(
                handle,
                type);
            switch (type) {
                case ResourceType::Accel:
                case ResourceType::Mesh:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerRead(static_cast<NoRangeHandle *>(h)));
                    break;
                case ResourceType::Buffer:
                case ResourceType::Texture:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerRead(static_cast<RangeHandle *>(h), range));
                    break;
                case ResourceType::Bindless:
                    dispatchLayer = std::max<int64_t>(dispatchLayer, GetLastLayerRead(static_cast<BindlessHandle *>(h)));
                    break;
            }
            dispatchReadHandle.emplace_back(range, h);
        }
    }
    FuncTable funcTable;
    template<typename... Callbacks>
    void visit(const ShaderDispatchCommandBase *command, uint64_t shader_handle, Callbacks &&...callbacks) noexcept {
        dispatchReadHandle.clear();
        dispatchWriteHandle.clear();
        useBindlessInPass = false;
        useAccelInPass = false;
        dispatchLayer = 0;
        argIndex = 0;
        shaderHandle = shader_handle;
        command->decode(*this);
        if constexpr (sizeof...(callbacks) > 0) {
            auto cb = {(callbacks(), 0)...};
        }
        for (auto &&i : dispatchReadHandle) {
            SetReadLayer(i.second, i.first, dispatchLayer);
        }
        for (auto &&i : dispatchWriteHandle) {
            SetWriteLayer(i.second, i.first, dispatchLayer);
        }
        AddCommand(command, dispatchLayer);
        if (useBindlessInPass) {
            bindlessMaxLayer = std::max<int64_t>(bindlessMaxLayer, dispatchLayer);
        }
        if (useAccelInPass) {
            maxAccelReadLevel = std::max<int64_t>(maxAccelReadLevel, dispatchLayer);
        }
    }

public:
    explicit CommandReorderVisitor(FuncTable &&funcTable) noexcept
        : rangePool(256, true),
          noRangePool(256, true),
          bindlessHandlePool(32, true),
          funcTable(std::forward<FuncTable>(funcTable)) {
    }
    ~CommandReorderVisitor() noexcept = default;
    void clear() noexcept {
        for (auto &&i : resMap) {
            rangePool.Delete(i.second);
        }
        for (auto &&i : noRangeResMap) {
            noRangePool.Delete(i.second);
        }
        for (auto &&i : bindlessMap) {
            bindlessHandlePool.Delete(i.second);
        }

        resMap.clear();
        noRangeResMap.clear();
        bindlessMap.clear();
        bindlessMaxLayer = -1;
        maxAccelReadLevel = -1;
        maxBufferReadLevel = -1;
        maxAccelWriteLevel = -1;
        maxMeshLevel = -1;
        luisa::span<typename decltype(commandLists)::value_type> sp(commandLists.data(), layerCount);
        for (auto &&i : sp) {
            i.clear();
        }
        layerCount = 0;
    }
    [[nodiscard]] auto command_lists() const noexcept {
        return luisa::span{commandLists.data(), layerCount};
    }

    // Buffer : resource
    void visit(const BufferUploadCommand *command) noexcept override {
        AddCommand(command, SetWrite(command->handle(), CopyRange(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferDownloadCommand *command) noexcept override {
        AddCommand(command, SetRead(command->handle(), CopyRange(command->offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferCopyCommand *command) noexcept override {
        AddCommand(command, SetRW(command->src_handle(), CopyRange(command->src_offset(), command->size()), ResourceType::Buffer, command->dst_handle(), CopyRange(command->dst_offset(), command->size()), ResourceType::Buffer));
    }
    void visit(const BufferToTextureCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto binSize = pixel_storage_size(command->storage(), sz.x, sz.y, sz.z);
        AddCommand(command, SetRW(command->buffer(), CopyRange(command->buffer_offset(), binSize), ResourceType::Buffer, command->texture(), CopyRange(command->level(), 1), ResourceType::Texture));
    }

    // Shader : function, read/write multi resources
    void visit(const ShaderDispatchCommand *command) noexcept override {
        visit(command, command->handle(), [&] {
            luisa::visit(
                [&]<typename T>(T const &t) {
                    if constexpr (std::is_same_v<T, ShaderDispatchCommand::IndirectArg>) {
                        AddDispatchHandle(
                            t.handle,
                            ResourceType::Buffer,
                            Range(),
                            false);
                    }
                },
                command->dispatch_size());
        });
    }
    void visit(const DrawRasterSceneCommand *command) noexcept override {
        auto SetTexDst = [&](ShaderDispatchCommandBase::TextureArgument const &a) {
            AddDispatchHandle(
                a.handle,
                ResourceType::Texture,
                Range(a.level),
                true);
        };
        visit(command, command->handle(), [&] {
            auto &&rtv = command->rtv_texs();
            auto &&dsv = command->dsv_tex();
            for (auto &&i : rtv) {
                SetTexDst(i);
            }
            if (dsv.handle != ~0ull) {
                SetTexDst(dsv);
            }
            for (auto &&mesh : command->scene) {
                for (auto &&v : mesh.vertex_buffers()) {
                    AddDispatchHandle(
                        v.handle(),
                        ResourceType::Buffer,
                        Range(v.offset(), v.size()),
                        false);
                }
                auto &&i = mesh.index();
                if (i.index() == 0) {
                    auto idx = luisa::get<0>(i);
                    AddDispatchHandle(
                        idx.handle(),
                        ResourceType::Buffer,
                        Range(idx.offset_bytes(), idx.size_bytes()),
                        false);
                }
            }
        });
    }

    // Texture : resource
    void visit(const TextureUploadCommand *command) noexcept override {
        AddCommand(command, SetWrite(command->handle(), CopyRange(command->level(), 1), ResourceType::Texture));
    }
    void visit(const TextureDownloadCommand *command) noexcept override {
        AddCommand(command, SetRead(command->handle(), CopyRange(command->level(), 1), ResourceType::Texture));
    }
    void visit(const TextureCopyCommand *command) noexcept override {
        AddCommand(command, SetRW(command->src_handle(), CopyRange(command->src_level(), 1), ResourceType::Texture, command->dst_handle(), CopyRange(command->dst_level(), 1), ResourceType::Texture));
    }
    void visit(const TextureToBufferCopyCommand *command) noexcept override {
        auto sz = command->size();
        auto binSize = pixel_storage_size(command->storage(), sz.x, sz.y, sz.z);
        AddCommand(command, SetRW(command->texture(), CopyRange(command->level(), 1), ResourceType::Texture, command->buffer(), CopyRange(command->buffer_offset(), binSize), ResourceType::Buffer));
    }
    void visit(const ClearDepthCommand *command) noexcept override {
        AddCommand(command, SetWrite(command->handle(), Range{}, ResourceType::Texture));
    }

    // BindlessArray : read multi resources
    void visit(const BindlessArrayUpdateCommand *command) noexcept override {
        AddCommand(command, SetWrite(command->handle(), Range(), ResourceType::Bindless));
    }

    // Accel : conclude meshes and their buffer
    void visit(const AccelBuildCommand *command) noexcept override {
        auto layer = SetWrite(command->handle(), Range(), ResourceType::Accel);
        maxAccelWriteLevel = std::max<int64_t>(maxAccelWriteLevel, layer);
        AddCommand(command, layer);
    }

    // Mesh : conclude vertex and triangle buffers
    void visit(const MeshBuildCommand *command) noexcept override {
        AddCommand(
            command,
            SetMesh(
                command->handle(),
                command->vertex_buffer(),
                Range(command->vertex_buffer_offset(),
                      command->vertex_buffer_size()),
                command->triangle_buffer(),
                Range(command->triangle_buffer_offset(),
                      command->triangle_buffer_size())));
    }
    void visit(const PrimBuildCommand *command) noexcept override {
        auto stride = funcTable.aabb_stride();
        AddCommand(
            command,
            SetAABB(
                command->handle(),
                command->aabb_buffer(),
                Range(command->aabb_offset() * stride, command->aabb_count() * stride)));
    }

    void visit(const CustomCommand *command) noexcept override {
        dispatchReadHandle.clear();
        dispatchWriteHandle.clear();
        useBindlessInPass = false;
        useAccelInPass = false;
        dispatchLayer = 0;
        for (auto &&i : command->resources()) {
            bool isWrite = ((uint)i.usage & (uint)Usage::WRITE) != 0;
            luisa::visit(
                [&]<typename T>(T const &res) {
                    if constexpr (std::is_same_v<T, CustomCommand::BufferView>) {
                        AddDispatchHandle(
                            res.handle,
                            ResourceType::Buffer,
                            Range(res.start_byte, res.size_byte),
                            isWrite);
                    } else if constexpr (std::is_same_v<T, CustomCommand::TextureView>) {
                        AddDispatchHandle(
                            res.handle,
                            ResourceType::Texture,
                            Range(res.start_mip, res.size_mip),
                            isWrite);
                    } else if constexpr (std::is_same_v<T, CustomCommand::MeshView>) {
                        AddDispatchHandle(
                            res.handle,
                            ResourceType::Mesh,
                            Range(),
                            isWrite);
                    } else if constexpr (std::is_same_v<T, CustomCommand::AccelView>) {
                        AddDispatchHandle(
                            res.handle,
                            ResourceType::Accel,
                            Range(),
                            isWrite);
                    } else {
                        AddDispatchHandle(
                            res.handle,
                            ResourceType::Bindless,
                            Range(),
                            isWrite);
                    }
                },
                i.resource_view);
        }
        for (auto &&i : dispatchReadHandle) {
            SetReadLayer(i.second, i.first, dispatchLayer);
        }
        for (auto &&i : dispatchWriteHandle) {
            SetWriteLayer(i.second, i.first, dispatchLayer);
        }
        AddCommand(command, dispatchLayer);
        if (useBindlessInPass) {
            bindlessMaxLayer = std::max<int64_t>(bindlessMaxLayer, dispatchLayer);
        }
        if (useAccelInPass) {
            maxAccelReadLevel = std::max<int64_t>(maxAccelReadLevel, dispatchLayer);
        }
    }

    void operator()(ShaderDispatchCommandBase::BufferArgument const &bf) {
        AddDispatchHandle(
            bf.handle,
            ResourceType::Buffer,
            Range(bf.offset, bf.size),
            ((uint)funcTable.get_usage(shaderHandle, argIndex) & (uint)Usage::WRITE) != 0);
        ++argIndex;
    }
    void operator()(ShaderDispatchCommandBase::TextureArgument const &bf) {
        AddDispatchHandle(
            bf.handle,
            ResourceType::Texture,
            Range(bf.level),
            ((uint)funcTable.get_usage(shaderHandle, argIndex) & (uint)Usage::WRITE) != 0);
        ++argIndex;
    }
    void operator()(ShaderDispatchCommandBase::BindlessArrayArgument const &bf) {
        useBindlessInPass = true;
        AddDispatchHandle(
            bf.handle,
            ResourceType::Bindless,
            Range(),
            false);
        ++argIndex;
    }
    void operator()(ShaderDispatchCommandBase::UniformArgument const &) {
        ++argIndex;
    }
    void operator()(ShaderDispatchCommandBase::AccelArgument const &bf) {
        useAccelInPass = true;
        AddDispatchHandle(
            bf.handle,
            ResourceType::Accel,
            Range(),
            false);
        ++argIndex;
    }
};

}// namespace luisa::compute
