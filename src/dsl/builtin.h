//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(detail::Expr<Src> s) noexcept { return s.template as<Dest>(); }

[[nodiscard]] inline auto thread_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->thread_id()};
}

[[nodiscard]] inline auto thread_x() noexcept {
    return thread_id().x;
}

[[nodiscard]] inline auto thread_y() noexcept {
    return thread_id().y;
}

[[nodiscard]] inline auto thread_z() noexcept {
    return thread_id().z;
}

[[nodiscard]] inline auto block_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->block_id()};
}

[[nodiscard]] inline auto block_x() noexcept {
    return block_id().x;
}

[[nodiscard]] inline auto block_y() noexcept {
    return block_id().y;
}

[[nodiscard]] inline auto block_z() noexcept {
    return block_id().z;
}

[[nodiscard]] inline auto dispatch_id() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->dispatch_id()};
}

[[nodiscard]] inline auto dispatch_x() noexcept {
    return dispatch_id().x;
}

[[nodiscard]] inline auto dispatch_y() noexcept {
    return dispatch_id().y;
}

[[nodiscard]] inline auto dispatch_z() noexcept {
    return dispatch_id().z;
}

[[nodiscard]] inline auto launch_size() noexcept {
    return detail::Expr<uint3>{FunctionBuilder::current()->launch_size()};
}

[[nodiscard]] inline auto block_size() noexcept {
    return FunctionBuilder::current()->block_size();
}

inline void set_block_size(uint x, uint y = 1u, uint z = 1u) noexcept {
    FunctionBuilder::current()->set_block_size(
        uint3{std::max(x, 1u), std::max(y, 1u), std::max(z, 1u)});
}

template<typename... T>
[[nodiscard]] inline auto multiple(T &&...v) noexcept {
    return std::make_tuple(detail::Expr{v}...);
}

// math functions

// atomic functions

// sync functions

// make_vector functions

namespace detail {

template<typename T>
[[nodiscard]] constexpr auto make_vector_tag() noexcept {
    if constexpr (std::is_same_v<T, bool2>) {
        return CallOp::MAKE_BOOL2;
    } else if constexpr (std::is_same_v<T, bool3>) {
        return CallOp::MAKE_BOOL3;
    } else if constexpr (std::is_same_v<T, bool4>) {
        return CallOp::MAKE_BOOL4;
    } else if constexpr (std::is_same_v<T, int2>) {
        return CallOp::MAKE_INT2;
    } else if constexpr (std::is_same_v<T, int3>) {
        return CallOp::MAKE_INT3;
    } else if constexpr (std::is_same_v<T, int4>) {
        return CallOp::MAKE_INT4;
    } else if constexpr (std::is_same_v<T, uint2>) {
        return CallOp::MAKE_UINT2;
    } else if constexpr (std::is_same_v<T, uint3>) {
        return CallOp::MAKE_UINT3;
    } else if constexpr (std::is_same_v<T, uint4>) {
        return CallOp::MAKE_UINT4;
    } else if constexpr (std::is_same_v<T, float2>) {
        return CallOp::MAKE_FLOAT2;
    } else if constexpr (std::is_same_v<T, float3>) {
        return CallOp::MAKE_FLOAT3;
    } else if constexpr (std::is_same_v<T, float4>) {
        return CallOp::MAKE_FLOAT4;
    } else {
        static_assert(always_false_v<T>);
    }
}

}// namespace detail

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    using V = Vector<T, 2>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression()})};
}

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression()})};
}

template<typename T>
[[nodiscard]] inline auto make_vector(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression(), w.expression()})};
}

#define LUISA_MAKE_VECTOR(type)                                                                                \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> s) noexcept {                                  \
        return make_vector(s, s);                                                                              \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> x, detail::Expr<type> y) noexcept {            \
        return make_vector(x, y);                                                                              \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 3>> v) noexcept {                       \
        return make_vector(v.x, v.y);                                                                          \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector(v.x, v.y);                                                                          \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<T, 2>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y));                                                  \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##3(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z) noexcept {                           \
        return make_vector(x, y, z);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> s) noexcept {                                  \
        return make_vector(s, s, s);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 2>> v, detail::Expr<type> z) noexcept { \
        return make_vector(v.x, v.y, z);                                                                       \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> x, detail::Expr<Vector<type, 2>> v) noexcept { \
        return make_vector(x, v.x, v.y);                                                                       \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector(v.x, v.y, v.z);                                                                     \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<T, 3>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y), cast<type>(v.z));                                 \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z, detail::Expr<type> w) noexcept {     \
        return make_vector(x, y, z, w);                                                                        \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> v, detail::Expr<type> z, detail::Expr<type> w) noexcept {                \
        return make_vector(v.x, v.y, z, w);                                                                    \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 2>> yz, detail::Expr<type> w) noexcept {               \
        return make_vector(x, yz.x, yz.y, w);                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<Vector<type, 2>> zw) noexcept {               \
        return make_vector(x, y, zw.x, zw.y);                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> xy, detail::Expr<Vector<type, 2>> zw) noexcept {                         \
        return make_vector(xy.x, xy.y, zw.x, zw.y);                                                            \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 3>> xyz, detail::Expr<type> w) noexcept {                                    \
        return make_vector(xyz.x, xyz.y, xyz.z, w);                                                            \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 3>> yzw) noexcept {                                    \
        return make_vector(x, yzw.x, yzw.y, yzw.z);                                                            \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##4(detail::Expr<Vector<T, 4>> v) noexcept {                          \
        return make_vector(cast<type>(v.x), cast<type>(v.y), cast<type>(v.z), cast<type>(v.w));                \
    }
LUISA_MAKE_VECTOR(bool)
LUISA_MAKE_VECTOR(int)
LUISA_MAKE_VECTOR(uint)
LUISA_MAKE_VECTOR(float)
#undef LUISA_MAKE_VECTOR

}// namespace luisa::compute
