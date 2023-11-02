#pragma once

#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>

namespace luisa {

namespace compute {
class CallableLibrary;
namespace detail {
class FunctionBuilder;
}// namespace detail

/// Variable class
class LC_AST_API Variable {

public:
    /// Variable tags
    enum struct Tag : uint32_t {

        // data
        LOCAL,
        SHARED,

        // reference
        REFERENCE,

        // resources
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID,
        DISPATCH_SIZE,
        KERNEL_ID,
        WARP_LANE_COUNT,
        WARP_LANE_ID,
        OBJECT_ID
    };

private:
    const Type *_type{nullptr};
    uint32_t _uid{};
    Tag _tag{};

private:
    friend class detail::FunctionBuilder;
    friend class CallableLibrary;
    Variable(const Type *type, Tag tag, uint32_t uid) noexcept;

public:
    Variable() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
    [[nodiscard]] auto is_local() const noexcept { return _tag == Tag::LOCAL; }
    [[nodiscard]] auto is_shared() const noexcept { return _tag == Tag::SHARED; }
    [[nodiscard]] auto is_reference() const noexcept { return _tag == Tag::REFERENCE; }
    [[nodiscard]] auto is_resource() const noexcept {
        return _tag == Tag::BUFFER ||
               _tag == Tag::TEXTURE ||
               _tag == Tag::BINDLESS_ARRAY ||
               _tag == Tag::ACCEL;
    }
    [[nodiscard]] auto is_builtin() const noexcept {
        return _tag == Tag::THREAD_ID ||
               _tag == Tag::BLOCK_ID ||
               _tag == Tag::DISPATCH_ID ||
               _tag == Tag::DISPATCH_SIZE ||
               _tag == Tag::KERNEL_ID ||
               _tag == Tag::WARP_LANE_COUNT ||
               _tag == Tag::WARP_LANE_ID ||
               _tag == Tag::OBJECT_ID;
    }
};

}// namespace compute

template<>
struct hash<compute::Variable> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(compute::Variable v) const noexcept { return v.hash(); }
};

}// namespace luisa
