#include "runtime/command.h"
#include <core/mathematics.h>
#include <runtime/command_reorder_visitor.h>
#include <runtime/stream.h>

namespace luisa::compute {
template<typename Func>
    requires(std::is_invocable_v<Func, CommandReorderVisitor::ResourceView const &>)
void CommandReorderVisitor::IterateMap(Func &&func, RangeHandle &handle, Range const &range) {
    for (auto &&r : handle.views) {
        if (r.first.collide(range)) {
            func(r.second);
        }
    }
}
bool CommandReorderVisitor::Range::operator==(Range const &r) const {
    return min == r.min && max == r.max;
}
bool CommandReorderVisitor::Range::collide(Range const &r) const {
    return min < r.max && r.min < max;
}
CommandReorderVisitor::ResourceHandle *CommandReorderVisitor::GetHandle(
    uint64_t tarGetHandle,
    ResourceType target_type) {
    auto func = [&](auto &&map, auto &&pool) {
        auto tryResult = map.try_emplace(
            tarGetHandle);
        auto &&value = tryResult.first->second;
        if (tryResult.second) {
            value = pool.create();
            value->handle = tarGetHandle;
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
size_t CommandReorderVisitor::GetLastLayerWrite(RangeHandle *handle, Range range) {
    size_t layer = 0;
    IterateMap(
        [&](auto &&handle) {
            layer = std::max<int64_t>(layer, std::max<int64_t>(handle.readLayer + 1, handle.writeLayer + 1));
        },
        *handle,
        range);
    if (bindlessMaxLayer >= layer) {
        for (auto &&i : bindlessMap) {
            if (device->is_resource_in_bindless_array(i.first, handle->handle)) {
                layer = std::max<int64_t>(layer, i.second->view.readLayer + 1);
            }
        }
    }
    return layer;
}
size_t CommandReorderVisitor::GetLastLayerWrite(NoRangeHandle *handle) {
    size_t layer = std::max<int64_t>(handle->view.readLayer + 1, handle->view.writeLayer + 1);

    switch (handle->type) {
        case ResourceType::Mesh:
            layer = std::max<int64_t>(layer, maxAccelLevel + 1);
            break;
        case ResourceType::Accel:
            layer = std::max<int64_t>(layer, maxAccelLevel + 1);
            layer = std::max<int64_t>(layer, maxMeshLevel + 1);
    }
    return layer;
}
size_t CommandReorderVisitor::GetLastLayerWrite(BindlessHandle *handle) {
    return std::max<int64_t>(handle->view.readLayer + 1, handle->view.writeLayer + 1);
}
size_t CommandReorderVisitor::GetLastLayerRead(RangeHandle *handle, Range range) {
    size_t layer = 0;
    IterateMap(
        [&](auto &&handle) {
            layer = std::max<int64_t>(layer, handle.writeLayer + 1);
        },
        *handle,
        range);
    return layer;
}
size_t CommandReorderVisitor::GetLastLayerRead(NoRangeHandle *handle) {
    size_t layer = handle->view.writeLayer + 1;
    if (handle->type == ResourceType::Accel) {
        layer = std::max<int64_t>(layer, maxAccelLevel + 1);
    }
    return layer;
}
size_t CommandReorderVisitor::GetLastLayerRead(BindlessHandle *handle) {
    return handle->view.writeLayer + 1;
}
CommandReorderVisitor::CommandReorderVisitor(Device::Interface *device) noexcept
    : device{device} {
    resMap.reserve(256);
    bindlessMap.reserve(256);
}
size_t CommandReorderVisitor::SetRead(
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

size_t CommandReorderVisitor::SetRead(
    uint64_t handle,
    Range range,
    ResourceType type) {
    auto srcHandle = GetHandle(
        handle,
        type);
    return SetRead(srcHandle, range);
}
void CommandReorderVisitor::SetWriteLayer(
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
void CommandReorderVisitor::SetReadLayer(
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
size_t CommandReorderVisitor::SetWrite(
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
size_t CommandReorderVisitor::SetWrite(
    uint64_t handle,
    Range range,
    ResourceType type) {
    auto dstHandle = GetHandle(
        handle,
        type);
    return SetWrite(dstHandle, range);
}
size_t CommandReorderVisitor::SetRW(
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
//TODO: Most backend can not support copy & kernel-write at same time, disable copy's range
static CommandReorderVisitor::Range CopyRange(int64_t min = std::numeric_limits<int64_t>::min(), int64_t max = std::numeric_limits<int64_t>::max()) {
    return CommandReorderVisitor::Range();
}
void CommandReorderVisitor::visit(const BufferUploadCommand *command) noexcept {
    AddCommand(command, SetWrite(command->handle(), CopyRange(command->offset(), command->size()), ResourceType::Buffer));
}
void CommandReorderVisitor::visit(const BufferDownloadCommand *command) noexcept {
    AddCommand(command, SetRead(command->handle(), CopyRange(command->offset(), command->size()), ResourceType::Buffer));
}
void CommandReorderVisitor::visit(const BufferCopyCommand *command) noexcept {
    AddCommand(command, SetRW(command->src_handle(), CopyRange(command->src_offset(), command->size()), ResourceType::Buffer, command->dst_handle(), CopyRange(command->dst_offset(), command->size()), ResourceType::Buffer));
}
void CommandReorderVisitor::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto sz = command->size();
    auto binSize = pixel_storage_size(command->storage()) * sz.x * sz.y * sz.z;
    AddCommand(command, SetRW(command->buffer(), CopyRange(command->buffer_offset(), binSize), ResourceType::Buffer, command->texture(), CopyRange(command->level()), ResourceType::Texture));
}
// Texture : resource

void CommandReorderVisitor::visit(const TextureUploadCommand *command) noexcept {
    AddCommand(command, SetWrite(command->handle(), CopyRange(command->level()), ResourceType::Texture));
}
void CommandReorderVisitor::visit(const TextureDownloadCommand *command) noexcept {
    AddCommand(command, SetRead(command->handle(), CopyRange(command->level()), ResourceType::Texture));
}
void CommandReorderVisitor::visit(const TextureCopyCommand *command) noexcept {
    AddCommand(command, SetRW(command->src_handle(), CopyRange(command->src_level()), ResourceType::Texture, command->dst_handle(), CopyRange(command->dst_level()), ResourceType::Texture));
}
void CommandReorderVisitor::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto sz = command->size();
    auto binSize = pixel_storage_size(command->storage()) * sz.x * sz.y * sz.z;
    AddCommand(command, SetRW(command->texture(), CopyRange(command->level()), ResourceType::Texture, command->buffer(), CopyRange(command->buffer_offset(), binSize), ResourceType::Buffer));
}
// Shader : function, read/write multi resources
void CommandReorderVisitor::visit(const ShaderDispatchCommand *command) noexcept {
    dispatchReadHandle.clear();
    dispatchWriteHandle.clear();
    useBindlessInPass = false;
    useAccelInPass = false;
    f = command->kernel();
    arg = command->kernel().arguments().data();
    dispatchLayer = 0;
    command->decode(*this);
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
        maxAccelLevel = std::max<int64_t>(maxAccelLevel, dispatchLayer);
    }
}

// BindlessArray : read multi resources
void CommandReorderVisitor::visit(const BindlessArrayUpdateCommand *command) noexcept {
    AddCommand(command, SetWrite(command->handle(), Range(), ResourceType::Bindless));
}

// Accel : conclude meshes and their buffer
void CommandReorderVisitor::visit(const AccelBuildCommand *command) noexcept {
    auto layer = SetWrite(command->handle(), Range(), ResourceType::Accel);
    maxAccelLevel = std::max<int64_t>(maxAccelLevel, layer);
    AddCommand(command, layer);
}

size_t CommandReorderVisitor::SetMesh(
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
    if (ib != vb) {
        auto ibHandle = GetHandle(
            ib,
            ResourceType::Buffer);
        auto rangeHandle = static_cast<RangeHandle *>(ibHandle);
        layer = std::max<int64_t>(layer, GetLastLayerRead(rangeHandle, ib_range));
        SetHandle(rangeHandle, ib_range, layer);
    }
    SetHandle(static_cast<RangeHandle *>(vbHandle), vb_range, layer);
    static_cast<NoRangeHandle *>(meshHandle)->view.writeLayer = layer;
    maxMeshLevel = std::max<int64_t>(maxMeshLevel, layer);
    return layer;
}

// Mesh : conclude vertex and triangle buffers
void CommandReorderVisitor::visit(const MeshBuildCommand *command) noexcept {
    AddCommand(command, SetMesh(command->handle(), command->vertex_buffer(), Range(), command->triangle_buffer(), Range()));
}

void CommandReorderVisitor::clear() noexcept {
    for (auto &&i : resMap) {
        rangePool.recycle(i.second);
    }
    for (auto &&i : noRangeResMap) {
        noRangePool.recycle(i.second);
    }
    for (auto &&i : bindlessMap) {
        bindlessHandlePool.recycle(i.second);
    }

    resMap.clear();
    noRangeResMap.clear();
    bindlessMap.clear();
    bindlessMaxLayer = -1;
    maxAccelLevel = -1;
    maxMeshLevel = -1;
    luisa::span<CommandList> sp(commandLists.data(), layerCount);
    for (auto &&i : sp) {
        i.clear();
    }
    layerCount = 0;
}

void CommandReorderVisitor::AddCommand(Command const *cmd, size_t layer) {
    if (commandLists.size() <= layer) {
        commandLists.resize(layer + 1);
    }
    layerCount = std::max<int64_t>(layerCount, layer + 1);
    commandLists[layer].append(const_cast<Command *>(cmd));
}

void CommandReorderVisitor::AddDispatchHandle(
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
void CommandReorderVisitor::operator()(ShaderDispatchCommand::TextureArgument const &bf) {
    AddDispatchHandle(
        bf.handle,
        ResourceType::Texture,
        Range(bf.level),
        ((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0);
    arg++;
}
void CommandReorderVisitor::operator()(ShaderDispatchCommand::BufferArgument const &bf) {
    AddDispatchHandle(
        bf.handle,
        ResourceType::Buffer,
        Range(bf.offset, bf.size),
        ((uint)f.variable_usage(arg->uid()) & (uint)Usage::WRITE) != 0);
    arg++;
}
void CommandReorderVisitor::operator()(ShaderDispatchCommand::UniformArgument bf) {
    arg++;
}
void CommandReorderVisitor::operator()(ShaderDispatchCommand::BindlessArrayArgument const &bf) {
    useBindlessInPass = true;
    AddDispatchHandle(
        bf.handle,
        ResourceType::Bindless,
        Range(),
        false);
    arg++;
}
void CommandReorderVisitor::operator()(ShaderDispatchCommand::AccelArgument const &bf) {
    useAccelInPass = true;
    AddDispatchHandle(
        bf.handle,
        ResourceType::Accel,
        Range(),
        false);
    arg++;
}

}// namespace luisa::compute
