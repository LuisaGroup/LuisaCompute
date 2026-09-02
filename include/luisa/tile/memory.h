#pragma once

#include <luisa/tile/value.h>

namespace luisa::compute::tile {

// An explicit addressable resource, independent of the execution hierarchy.
// Its declaration site owns the allocation; the state handle participates in
// structured capture, but never appears in the user-facing load/store syntax.
template<typename T>
class Memory final {

    static_assert(scalar_cpp_type<T> && !std::is_const_v<T>);

private:
    Value *_memory{nullptr};
    detail::ValueHandle _state;

public:
    explicit Memory(detail::DeclaredMemory declaration) noexcept
        : _memory{declaration.memory}, _state{std::move(declaration.state)} {}
    Memory(const Memory &) = delete;
    Memory(Memory &&) noexcept = default;
    Memory &operator=(const Memory &) = delete;
    Memory &operator=(Memory &&) = delete;

    [[nodiscard]] bool valid() const noexcept { return _memory != nullptr && static_cast<bool>(_state); }
    [[nodiscard]] Value *ir_value() const noexcept { return valid() ? _memory : nullptr; }
    [[nodiscard]] const IndexSpace &space() const noexcept {
        static const IndexSpace empty;
        return valid() ? *_memory->type().index_space() : empty;
    }

    // A load snapshots the current contents into Tile SSA. A later store to
    // this Memory cannot change an already loaded Tile.
    [[nodiscard]] Tile<T> load() const noexcept {
        return Tile<T>{detail::load_memory(_memory, _state)};
    }
    void store(const Tile<T> &value) noexcept {
        detail::store_memory(_memory, _state, value.ir_value());
    }
};

template<scalar_cpp_type T>
[[nodiscard]] Memory<T> memory(const IndexSpace &space, mem::Resource resource = mem::auto_) noexcept {
    return Memory<T>{detail::declare_memory(scalar_type_v<T>, space, resource)};
}

}// namespace luisa::compute::tile
