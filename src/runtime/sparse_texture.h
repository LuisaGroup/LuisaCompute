#pragma once

#include <runtime/rhi/resource.h>
#include <runtime/stream_event.h>
#include <runtime/rhi/command.h>
#include <runtime/rhi/tile_modification.h>

namespace luisa::compute {
template<typename T>
class Buffer;

template<typename T>
class BufferView;

class LC_RUNTIME_API SparseTexture : public Resource {

public:
    struct LC_RUNTIME_API UpdateTiles {
        uint64_t handle;
        luisa::vector<TileModification> tiles;
        void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
    };

protected:
    luisa::vector<TileModification> _tiles;
    SparseTexture(DeviceInterface *device, const ResourceCreationInfo &info) noexcept;
    SparseTexture(SparseTexture &&) noexcept = default;
    ~SparseTexture() noexcept override;

public:
    // deleted members should be public
    SparseTexture(const SparseTexture &) noexcept = delete;
    SparseTexture &operator=(SparseTexture &&) noexcept = delete;// use _move_from in derived classes
    SparseTexture &operator=(const SparseTexture &) noexcept = delete;

public:
    [[nodiscard]] UpdateTiles update() noexcept;
};

LUISA_MARK_STREAM_EVENT_TYPE(SparseTexture::UpdateTiles)

}// namespace luisa::compute
