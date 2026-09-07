#pragma once

#include <utility>

#include <luisa/ast/coro_suspend.h>
#include <luisa/ast/type.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/var.h>

namespace luisa::compute {

struct CoroFrame;

namespace coro {

class CoroGraph;

/// A typed projection of one suspend-extension binding onto the coroutine's
/// existing interference-colored frame. This object is an access plan, not a
/// second storage owner: several bindings may intentionally contain identical
/// pieces and therefore read/write the same physical slots.
class LUISA_CORO_API CoroSlotAccess {

public:
    struct Piece {
        size_t frame_value_index{0u};
        size_t field_index{0u};
        luisa::vector<uint32_t> access_chain;
        const Type *logical_type{nullptr};
        const Type *physical_type{nullptr};
        luisa::optional<uint32_t> bit_offset;
    };

private:
    const Type *_type{nullptr};
    CoroSuspendBindingAccess _access{CoroSuspendBindingAccess::read};
    CoroSuspendBindingLifetime _lifetime{
        CoroSuspendBindingLifetime::boundary};
    luisa::vector<Piece> _pieces;
    // Binding-local logical effects and their physical projection. Stage-wide
    // liveness belongs to CoroGraph::Boundary::Stage, not to each binding.
    luisa::vector<size_t> _use_frame_values;
    luisa::vector<size_t> _def_frame_values;
    luisa::vector<size_t> _use_slots;
    luisa::vector<size_t> _def_slots;
    // Physical carriers that must be reconstructed even without a logical
    // use. Currently this is the shared uint of a packed Boolean write.
    luisa::vector<size_t> _rmw_slots;
    luisa::vector<size_t> _reconstruct_slots;

private:
    friend class CoroGraph;
    CoroSlotAccess(
        const Type *type, CoroSuspendBindingAccess access,
        CoroSuspendBindingLifetime lifetime,
        luisa::vector<Piece> pieces) noexcept;

    [[nodiscard]] const Expression *_read(CoroFrame &frame) const noexcept;
    void _write(CoroFrame &frame, const Expression *value) const noexcept;

public:
    CoroSlotAccess() noexcept = default;

    [[nodiscard]] const Type *type() const noexcept { return _type; }
    [[nodiscard]] CoroSuspendBindingAccess access() const noexcept {
        return _access;
    }
    [[nodiscard]] CoroSuspendBindingLifetime lifetime() const noexcept {
        return _lifetime;
    }
    [[nodiscard]] bool readable() const noexcept {
        return _access != CoroSuspendBindingAccess::write;
    }
    [[nodiscard]] bool writable() const noexcept {
        return _access != CoroSuspendBindingAccess::read;
    }
    [[nodiscard]] bool materialized() const noexcept {
        return !_pieces.empty();
    }
    [[nodiscard]] auto pieces() const noexcept {
        return luisa::span<const Piece>{_pieces};
    }
    [[nodiscard]] auto use_frame_values() const noexcept {
        return luisa::span<const size_t>{_use_frame_values};
    }
    [[nodiscard]] auto def_frame_values() const noexcept {
        return luisa::span<const size_t>{_def_frame_values};
    }
    /// Physical CoroFrameDesc field indices logically consumed by read or
    /// read-write access. Duplicate colored slots are removed.
    [[nodiscard]] auto use_slots() const noexcept {
        return luisa::span<const size_t>{_use_slots};
    }
    /// Physical fields dirtied by write or read-write access. Whether a dirty
    /// field must be written back is decided by stage-wide live-out analysis.
    [[nodiscard]] auto def_slots() const noexcept {
        return luisa::span<const size_t>{_def_slots};
    }
    [[nodiscard]] auto rmw_slots() const noexcept {
        return luisa::span<const size_t>{_rmw_slots};
    }
    /// Minimal fields that must initialize a partial CoroFrame before invoking
    /// this access: logical uses union physical RMW carriers.
    [[nodiscard]] auto reconstruct_slots() const noexcept {
        return luisa::span<const size_t>{_reconstruct_slots};
    }

    template<typename T>
    [[nodiscard]] Var<T> read(CoroFrame &frame) const noexcept {
        LUISA_ASSERT(readable(),
                     "Attempted to read a write-only coroutine binding.");
        LUISA_ASSERT(_type == Type::of<T>(),
                     "Coroutine binding read type mismatch (expected '{}', "
                     "got '{}').",
                     _type == nullptr ? "<null>" : _type->description(),
                     Type::of<T>()->description());
        return Var<T>{_read(frame)};
    }

    template<typename T>
    void write(CoroFrame &frame, Expr<T> value) const noexcept {
        LUISA_ASSERT(writable(),
                     "Attempted to write a read-only coroutine binding.");
        LUISA_ASSERT(_type == Type::of<T>(),
                     "Coroutine binding write type mismatch (expected '{}', "
                     "got '{}').",
                     _type == nullptr ? "<null>" : _type->description(),
                     Type::of<T>()->description());
        _write(frame, value.expression());
    }
};

}// namespace coro
}// namespace luisa::compute
