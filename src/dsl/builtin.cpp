#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/func.h>

namespace luisa::compute {

namespace detail {
LUISA_DSL_API void luisa_compute_validate_warp_size(uint8_t warp_size) noexcept {
    LUISA_ASSERT(
        warp_size == 1u ||
            (warp_size >= 4u && warp_size <= 128u &&
             luisa::next_pow2<uint>(warp_size) == warp_size),
        "Invalid warp size {}. Expected 1, 4, 8, 16, 32, 64, or 128.",
        warp_size);
}
LUISA_DSL_API void luisa_compute_validate_block_size(uint x, uint y, uint z) noexcept {
    auto size = make_uint3(x, y, z);
    LUISA_ASSERT(all(size >= 1u && size <= 1024u),
                 "Invalid block size ({}, {}, {}). "
                 "Block size must be in range [1, 1024].",
                 x, y, z);
    LUISA_ASSERT(x * y * z <= 1024u,
                 "Invalid block size ({}, {}, {}). "
                 "Threads per block must not exceed 1024.",
                 x, y, z);
}

LUISA_DSL_API void luisa_compute_validate_local_array_backward_types(const Type *x, const Type *grad) noexcept {
    LUISA_ASSERT(*x == *grad,
                 "Invalid backward type: {} vs {}.",
                 x->description(), grad->description());
}

LUISA_DSL_API void luisa_compute_check_matrix_size(uint idx, uint max_size) {
    if (idx >= max_size) [[unlikely]] {
        LUISA_ERROR("Matrix access index {} out of range [0, {}]", idx, max_size - 1);
    }
}

#define LUISA_EXPR(value) \
    extract_expression(std::forward<decltype(value)>(value))

#define LUISA_ACCESS(v, element_type, index)     \
    FunctionBuilder::current()->access(          \
        Type::of<element_type>(), LUISA_EXPR(v), \
        FunctionBuilder::current()->literal(Type::of<uint>(), uint(index)))

template<typename T, uint dim>
auto make_matrix_row(Var<Matrix<T, dim>> &mat, uint value) {
#define LUISA_MAKE_MAT(TYPE)                                                               \
    if constexpr (dim == 2)                                                                \
        return make_##TYPE##2(mat.cols[0][value], mat.cols[1][value]);                     \
    else if constexpr (dim == 3)                                                           \
        return make_##TYPE##3(mat.cols[0][value], mat.cols[1][value], mat.cols[2][value]); \
    else                                                                                   \
        return make_##TYPE##4(mat.cols[0][value], mat.cols[1][value], mat.cols[2][value], mat.cols[3][value]);
    if constexpr (std::is_same_v<T, double>) {
        LUISA_MAKE_MAT(double);
    } else {
        LUISA_MAKE_MAT(half);
    }
#undef LUISA_MAKE_MAT
}

template<typename T, uint dim, typename... Args>
auto make_vector(Args... args) {
#define LUISA_MAKE_MAT(TYPE)            \
    if constexpr (dim == 2)             \
        return make_##TYPE##2(args...); \
    else if constexpr (dim == 3)        \
        return make_##TYPE##3(args...); \
    else                                \
        return make_##TYPE##4(args...);

    if constexpr (std::is_same_v<T, double>) {
        LUISA_MAKE_MAT(double);
    } else {
        LUISA_MAKE_MAT(half);
    }
#undef LUISA_MAKE_MAT
}

template<typename T, uint dim, typename... Args>
auto make_matrix(Args... args) {
#define LUISA_MAKE_MAT(TYPE)              \
    if constexpr (dim == 2)               \
        return make_##TYPE##2x2(args...); \
    else if constexpr (dim == 3)          \
        return make_##TYPE##3x3(args...); \
    else                                  \
        return make_##TYPE##4x4(args...);

    if constexpr (std::is_same_v<T, double>) {
        LUISA_MAKE_MAT(double);
    } else {
        LUISA_MAKE_MAT(half);
    }
#undef LUISA_MAKE_MAT
}

// !!! TO MAXWELL: DEFINING GLOBAL FUNCTIONS WITH NAME STARTING WITH UNDERSCORE IS UNDEFINED BEHAVIOR !!!

template<typename T>
auto luisa_compute_transpose(Expr<Matrix<T, 2>> m) noexcept {
    return make_matrix<T, 2>(
        m.cols[0].x, m.cols[1].x,
        m.cols[0].y, m.cols[1].y);
}

template<typename T>
auto luisa_compute_transpose(Expr<Matrix<T, 3>> m) noexcept {
    return make_matrix<T, 3>(
        m.cols[0].x, m.cols[1].x, m.cols[2].x,
        m.cols[0].y, m.cols[1].y, m.cols[2].y,
        m.cols[0].z, m.cols[1].z, m.cols[2].z);
}

template<typename T>
auto luisa_compute_transpose(Expr<Matrix<T, 4>> m) noexcept {
    return make_matrix<T, 4>(
        m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x,
        m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y,
        m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z,
        m.cols[0].w, m.cols[1].w, m.cols[2].w, m.cols[3].w);
}

template<typename T>
auto luisa_compute_determinant(Expr<Matrix<T, 2>> m) noexcept {
    return m.cols[0][0] * m.cols[1][1] - m.cols[1][0] * m.cols[0][1];
}

template<typename T>
auto luisa_compute_determinant(Expr<Matrix<T, 3>> m) noexcept {// from GLM
    return m.cols[0].x * (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) -
           m.cols[1].x * (m.cols[0].y * m.cols[2].z - m.cols[2].y * m.cols[0].z) +
           m.cols[2].x * (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z);
}

template<typename T>
auto luisa_compute_determinant(Expr<Matrix<T, 4>> m) noexcept {// from GLM
    const auto coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const auto coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const auto coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const auto coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const auto coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const auto coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const auto coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const auto coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const auto coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const auto coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const auto coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const auto coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const auto coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const auto coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const auto coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const auto coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const auto coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const auto coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const auto fac0 = make_vector<T, 4>(coef00, coef00, coef02, coef03);
    const auto fac1 = make_vector<T, 4>(coef04, coef04, coef06, coef07);
    const auto fac2 = make_vector<T, 4>(coef08, coef08, coef10, coef11);
    const auto fac3 = make_vector<T, 4>(coef12, coef12, coef14, coef15);
    const auto fac4 = make_vector<T, 4>(coef16, coef16, coef18, coef19);
    const auto fac5 = make_vector<T, 4>(coef20, coef20, coef22, coef23);
    const auto Vec0 = make_vector<T, 4>(m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x);
    const auto Vec1 = make_vector<T, 4>(m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y);
    const auto Vec2 = make_vector<T, 4>(m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z);
    const auto Vec3 = make_vector<T, 4>(m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    const auto sign_a = make_vector<T, 4>(T{1.0}, -T{1.0}, T{1.0}, -T{1.0});
    const auto sign_b = make_vector<T, 4>(-T{1.0}, T{1.0}, -T{1.0}, T{1.0});
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m.cols[0] * make_vector<T, 4>(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

template<typename T>
auto luisa_compute_inverse(Expr<Matrix<T, 2>> m) noexcept {
    const auto one_over_determinant = T{1.0} / (m.cols[0][0] * m.cols[1][1] - m.cols[1][0] * m.cols[0][1]);
    return make_matrix<T, 2>(m.cols[1][1] * one_over_determinant,
                             -m.cols[0][1] * one_over_determinant,
                             -m.cols[1][0] * one_over_determinant,
                             +m.cols[0][0] * one_over_determinant);
}

template<typename T>
auto luisa_compute_inverse(Expr<Matrix<T, 3>> m) noexcept {// from GLM
    const auto one_over_determinant = T{1.0} / (m.cols[0].x * (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) - m.cols[1].x * (m.cols[0].y * m.cols[2].z - m.cols[2].y * m.cols[0].z) + m.cols[2].x * (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z));
    return make_matrix<T, 3>(
        (m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z) * one_over_determinant,
        (m.cols[2].y * m.cols[0].z - m.cols[0].y * m.cols[2].z) * one_over_determinant,
        (m.cols[0].y * m.cols[1].z - m.cols[1].y * m.cols[0].z) * one_over_determinant,
        (m.cols[2].x * m.cols[1].z - m.cols[1].x * m.cols[2].z) * one_over_determinant,
        (m.cols[0].x * m.cols[2].z - m.cols[2].x * m.cols[0].z) * one_over_determinant,
        (m.cols[1].x * m.cols[0].z - m.cols[0].x * m.cols[1].z) * one_over_determinant,
        (m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y) * one_over_determinant,
        (m.cols[2].x * m.cols[0].y - m.cols[0].x * m.cols[2].y) * one_over_determinant,
        (m.cols[0].x * m.cols[1].y - m.cols[1].x * m.cols[0].y) * one_over_determinant);
}

template<typename T>
auto luisa_compute_inverse(Expr<Matrix<T, 4>> m) noexcept {// from GLM
    const auto coef00 = m.cols[2].z * m.cols[3].w - m.cols[3].z * m.cols[2].w;
    const auto coef02 = m.cols[1].z * m.cols[3].w - m.cols[3].z * m.cols[1].w;
    const auto coef03 = m.cols[1].z * m.cols[2].w - m.cols[2].z * m.cols[1].w;
    const auto coef04 = m.cols[2].y * m.cols[3].w - m.cols[3].y * m.cols[2].w;
    const auto coef06 = m.cols[1].y * m.cols[3].w - m.cols[3].y * m.cols[1].w;
    const auto coef07 = m.cols[1].y * m.cols[2].w - m.cols[2].y * m.cols[1].w;
    const auto coef08 = m.cols[2].y * m.cols[3].z - m.cols[3].y * m.cols[2].z;
    const auto coef10 = m.cols[1].y * m.cols[3].z - m.cols[3].y * m.cols[1].z;
    const auto coef11 = m.cols[1].y * m.cols[2].z - m.cols[2].y * m.cols[1].z;
    const auto coef12 = m.cols[2].x * m.cols[3].w - m.cols[3].x * m.cols[2].w;
    const auto coef14 = m.cols[1].x * m.cols[3].w - m.cols[3].x * m.cols[1].w;
    const auto coef15 = m.cols[1].x * m.cols[2].w - m.cols[2].x * m.cols[1].w;
    const auto coef16 = m.cols[2].x * m.cols[3].z - m.cols[3].x * m.cols[2].z;
    const auto coef18 = m.cols[1].x * m.cols[3].z - m.cols[3].x * m.cols[1].z;
    const auto coef19 = m.cols[1].x * m.cols[2].z - m.cols[2].x * m.cols[1].z;
    const auto coef20 = m.cols[2].x * m.cols[3].y - m.cols[3].x * m.cols[2].y;
    const auto coef22 = m.cols[1].x * m.cols[3].y - m.cols[3].x * m.cols[1].y;
    const auto coef23 = m.cols[1].x * m.cols[2].y - m.cols[2].x * m.cols[1].y;
    const auto fac0 = make_vector<T, 4>(coef00, coef00, coef02, coef03);
    const auto fac1 = make_vector<T, 4>(coef04, coef04, coef06, coef07);
    const auto fac2 = make_vector<T, 4>(coef08, coef08, coef10, coef11);
    const auto fac3 = make_vector<T, 4>(coef12, coef12, coef14, coef15);
    const auto fac4 = make_vector<T, 4>(coef16, coef16, coef18, coef19);
    const auto fac5 = make_vector<T, 4>(coef20, coef20, coef22, coef23);
    const auto Vec0 = make_vector<T, 4>(m.cols[1].x, m.cols[0].x, m.cols[0].x, m.cols[0].x);
    const auto Vec1 = make_vector<T, 4>(m.cols[1].y, m.cols[0].y, m.cols[0].y, m.cols[0].y);
    const auto Vec2 = make_vector<T, 4>(m.cols[1].z, m.cols[0].z, m.cols[0].z, m.cols[0].z);
    const auto Vec3 = make_vector<T, 4>(m.cols[1].w, m.cols[0].w, m.cols[0].w, m.cols[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    const auto sign_a = make_vector<T, 4>(T{1.0}, -T{1.0}, T{1.0}, -T{1.0});
    const auto sign_b = make_vector<T, 4>(-T{1.0}, T{1.0}, -T{1.0}, T{1.0});
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m.cols[0] * make_vector<T, 4>(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = T{1.0} / dot1;
    return make_matrix<T, 4>(inv_0 * one_over_determinant,
                             inv_1 * one_over_determinant,
                             inv_2 * one_over_determinant,
                             inv_3 * one_over_determinant);
}

template<typename T, uint dim>
struct MulMatMatCallable {
    using MatType = Matrix<T, dim>;
    using VecType = Vector<T, dim>;
    Callable<MatType(MatType, MatType)> func;
    MulMatMatCallable()
        : func([](Var<MatType> a, Var<MatType> b) {
              Var<MatType> r;
              for (uint x = 0; x < dim; ++x)
                  for (uint y = 0; y < dim; ++y) {
                      r.cols[y][x] = dot(
                          make_matrix_row<T, dim>(a, x),
                          b.cols[y]);
                  }
              return r;
          }) {
    }
};

template<typename T, uint dim>
struct MulMatVecCallable {
    using MatType = Matrix<T, dim>;
    using VecType = Vector<T, dim>;
    Callable<VecType(MatType, VecType)> func;
    MulMatVecCallable()
        : func([](Var<MatType> a, Var<VecType> b) {
              Var<VecType> r;
              for (int y = 0; y < dim; ++y) {
                  r[y] = dot(
                      make_matrix_row<T, dim>(a, y),
                      // a.cols[y],
                      b);
              }
              return r;
          }) {}
};

#define LUISA_IMPL_MUL(TT, dim)                                                                                             \
    LUISA_DSL_API Var<TT##dim##x##dim> luisa_compute_mul_##TT##dim##x##dim(Expr<TT##dim##x##dim> a, Expr<TT##dim##x##dim> b) { \
        static detail::MulMatMatCallable<TT, dim> _func;                                                                    \
        return _func.func(a, b);                                                                                            \
    }                                                                                                                       \
    LUISA_DSL_API Var<TT##dim> luisa_compute_mul_##TT##dim##x##dim(Expr<TT##dim##x##dim> a, Expr<TT##dim> b) {                 \
        static detail::MulMatVecCallable<TT, dim> _func;                                                                    \
        return _func.func(a, b);                                                                                            \
    }

#define LUISA_IMPL_MUL_ALL(TT) \
    LUISA_IMPL_MUL(TT, 2)      \
    LUISA_IMPL_MUL(TT, 3)      \
    LUISA_IMPL_MUL(TT, 4)

LUISA_IMPL_MUL_ALL(half)
LUISA_IMPL_MUL_ALL(double)

}// namespace detail

#define LUISA_MATRIX_INTRIN(TYPE, DIM)                                                  \
    LUISA_DSL_API Var<TYPE##DIM##x##DIM> transpose(Expr<TYPE##DIM##x##DIM> mat) {          \
        static Callable<TYPE##DIM##x##DIM(TYPE##DIM##x##DIM)> _callable{[&](auto &&v) { \
            return detail::luisa_compute_transpose<TYPE>(v);                            \
        }};                                                                             \
        return _callable(mat);                                                          \
    }                                                                                   \
    LUISA_DSL_API Var<TYPE##DIM##x##DIM> inverse(Expr<TYPE##DIM##x##DIM> mat) {            \
        static Callable<TYPE##DIM##x##DIM(TYPE##DIM##x##DIM)> _callable{[&](auto &&v) { \
            return detail::luisa_compute_inverse<TYPE>(v);                              \
        }};                                                                             \
        return _callable(mat);                                                          \
    }                                                                                   \
    LUISA_DSL_API Var<TYPE> determinant(Expr<TYPE##DIM##x##DIM> mat) {                     \
        static Callable<TYPE(TYPE##DIM##x##DIM)> _callable{[&](auto &&v) {              \
            return detail::luisa_compute_determinant<TYPE>(v);                          \
        }};                                                                             \
        return _callable(mat);                                                          \
    }

LUISA_MATRIX_INTRIN(double, 2)
LUISA_MATRIX_INTRIN(double, 3)
LUISA_MATRIX_INTRIN(double, 4)
LUISA_MATRIX_INTRIN(half, 2)
LUISA_MATRIX_INTRIN(half, 3)
LUISA_MATRIX_INTRIN(half, 4)

#undef LUISA_ACCESS
#undef LUISA_EXPR
#undef LUISA_IMPL_MUL
#undef LUISA_MATRIX_INTRIN
#undef LUISA_IMPL_MUL_ALL

}// namespace luisa::compute
