#pragma once

#include <cstdint>
#include <cstddef>

#include <luisa/core/stl/format.h>
#include <luisa/dsl/soa.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>

template<typename T>
struct luisa_compute_extension {};

namespace luisa::compute::detail {

template<typename T>
struct c_array_to_std_array {
    using type = T;
};

template<typename T, size_t N>
struct c_array_to_std_array<T[N]> {
    using type = std::array<T, N>;
};

template<typename T>
using c_array_to_std_array_t = typename c_array_to_std_array<T>::type;

};// namespace luisa::compute::detail

#define LUISA_STRUCT_MAKE_MEMBER_TYPE(m)                    \
    using member_type_##m = detail::c_array_to_std_array_t< \
        std::remove_cvref_t<                                \
            decltype(std::declval<this_type>().m)>>;

#define LUISA_STRUCT_MAKE_MEMBER_INIT(m)          \
    m(detail::FunctionBuilder::current()->member( \
        Type::of<member_type_##m>(),              \
        this->_expression,                        \
        _member_index(#m)))

#define LUISA_STRUCT_MAKE_MEMBER_REF_DECL(m) \
    Var<member_type_##m> m;

#define LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL(m) \
    Expr<member_type_##m> m;

#define LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL(m) \
    AtomicRef<member_type_##m> m{                   \
        this->member<member_type_##m>(_member_index(#m))};

#define LUISA_DERIVE_FMT_MAP_STRUCT_FIELD(x) fmt::format(FMT_STRING(#x ": {}"), input.x)

#define LUISA_DERIVE_FMT(Struct, DisplayName, ...)                                    \
    template<>                                                                        \
    struct fmt::formatter<Struct> {                                                   \
        constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {    \
            return ctx.end();                                                         \
        }                                                                             \
        template<typename FormatContext>                                              \
        auto format(const Struct &input, FormatContext &ctx) -> decltype(ctx.out()) { \
            return fmt::format_to(ctx.out(), FMT_STRING(#DisplayName "{{ {} }}"),     \
                                  fmt::join(std::array{LUISA_MAP_LIST(                \
                                                LUISA_DERIVE_FMT_MAP_STRUCT_FIELD,    \
                                                __VA_ARGS__)},                        \
                                            ", "));                                   \
        }                                                                             \
    };

#define LUISA_DERIVE_DSL_STRUCT(S, ...)                                                       \
    namespace luisa::compute {                                                                \
    namespace detail {                                                                        \
    template<>                                                                                \
    class AtomicRef<S> : private AtomicRefBase {                                              \
    private:                                                                                  \
        using this_type = S;                                                                  \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                 \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept { \
            constexpr const std::string_view member_names[]{                                  \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                \
            return std::find(std::begin(member_names),                                        \
                             std::end(member_names),                                          \
                             name) -                                                          \
                   std::begin(member_names);                                                  \
        }                                                                                     \
                                                                                              \
    public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL, __VA_ARGS__)                      \
        explicit AtomicRef(const AtomicRefNode *node) noexcept                                \
            : AtomicRefBase{node} {}                                                          \
    };                                                                                        \
    }                                                                                         \
    template<>                                                                                \
    struct Expr<S> : public detail::ExprEnableBitwiseCast<Expr<S>> {                          \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const Expression *_expression;                                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                 \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept { \
            constexpr const std::string_view member_names[]{                                  \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                \
            return std::find(std::begin(member_names),                                        \
                             std::end(member_names),                                          \
                             name) -                                                          \
                   std::begin(member_names);                                                  \
        }                                                                                     \
                                                                                              \
    public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)                            \
        explicit Expr(const Expression *e) noexcept                                           \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Expr(Expr &&another) noexcept = default;                                              \
        Expr(const Expr &another) noexcept = default;                                         \
        Expr &operator=(Expr) noexcept = delete;                                              \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return Expr<M>{detail::FunctionBuilder::current()->member(                        \
                Type::of<M>(), this->expression(), i)};                                       \
        };                                                                                    \
    };                                                                                        \
    namespace detail {                                                                        \
    template<>                                                                                \
    struct Ref<S> : public detail::ExprEnableBitwiseCast<Ref<S>> {                            \
    private:                                                                                  \
        using this_type = S;                                                                  \
        const Expression *_expression;                                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                 \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept { \
            constexpr const std::string_view member_names[]{                                  \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                \
            return std::find(std::begin(member_names), std::end(member_names), name) -        \
                   std::begin(member_names);                                                  \
        }                                                                                     \
                                                                                              \
    public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_REF_DECL, __VA_ARGS__)                             \
        explicit Ref(const Expression *e) noexcept                                            \
            : _expression{e},                                                                 \
              LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {}                   \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }          \
        Ref(Ref &&another) noexcept = default;                                                \
        Ref(const Ref &another) noexcept = default;                                           \
        [[nodiscard]] operator Expr<S>() const noexcept {                                     \
            return Expr<S>{this->expression()};                                               \
        }                                                                                     \
        template<typename Rhs>                                                                \
        void operator=(Rhs &&rhs) & noexcept { dsl::assign(*this, std::forward<Rhs>(rhs)); }  \
        void operator=(Ref rhs) & noexcept { (*this) = Expr{rhs}; }                           \
        template<size_t i>                                                                    \
        [[nodiscard]] auto get() const noexcept {                                             \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S>>;                      \
            return Ref<M>{detail::FunctionBuilder::current()->member(                         \
                Type::of<M>(), this->expression(), i)};                                       \
        };                                                                                    \
        [[nodiscard]] auto operator->() noexcept {                                            \
            return reinterpret_cast<luisa_compute_extension<S> *>(this);                      \
        }                                                                                     \
        [[nodiscard]] auto operator->() const noexcept {                                      \
            return reinterpret_cast<const luisa_compute_extension<S> *>(this);                \
        }                                                                                     \
    };                                                                                        \
    }                                                                                         \
    }

#define LUISA_SOA_VIEW_MAKE_MEMBER_DECL(m) \
    SOAView<member_type_##m> m;

#define LUISA_SOA_VIEW_MAKE_MEMBER_SOA_SIZE_ACCUM(m) \
    (SOAView<member_type_##m>::compute_soa_size(soa_size)) +

#define LUISA_SOA_VIEW_MAKE_MEMBER_INIT(m)                                                      \
    m(buffer, soa_offset + _accumulate_soa_offset<member_type_##m>(soa_offset_accum, soa_size), \
      soa_size, elem_offset, elem_size)

#define LUISA_DERIVE_SOA_VIEW(S, ...)                                                               \
    namespace luisa::compute {                                                                      \
    template<>                                                                                      \
    class SOAView<S> : public detail::SOAViewBase<S> {                                              \
                                                                                                    \
    private:                                                                                        \
        using this_type = S;                                                                        \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                       \
                                                                                                    \
    public:                                                                                         \
        [[nodiscard]] static auto compute_soa_size(auto soa_size) noexcept {                        \
            return LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_SOA_SIZE_ACCUM, __VA_ARGS__)                \
            0u;                                                                                     \
        }                                                                                           \
                                                                                                    \
    public:                                                                                         \
        LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_DECL, __VA_ARGS__)                                     \
                                                                                                    \
    private:                                                                                        \
        template<typename T>                                                                        \
        [[nodiscard]] static auto _accumulate_soa_offset(size_t &accum, size_t soa_size) noexcept { \
            auto offset = accum;                                                                    \
            accum += SOAView<T>::compute_soa_size(soa_size);                                        \
            return offset;                                                                          \
        }                                                                                           \
                                                                                                    \
        SOAView(size_t soa_offset_accum,                                                            \
                BufferView<uint> buffer,                                                            \
                size_t soa_offset, size_t soa_size,                                                 \
                size_t elem_offset, size_t elem_size) noexcept                                      \
            : detail::SOAViewBase<S>{buffer, soa_offset, soa_size, elem_offset, elem_size},         \
              LUISA_MAP_LIST(LUISA_SOA_VIEW_MAKE_MEMBER_INIT, __VA_ARGS__) {}                       \
                                                                                                    \
    public:                                                                                         \
        SOAView(BufferView<uint> buffer,                                                            \
                size_t soa_offset, size_t soa_size,                                                 \
                size_t elem_offset, size_t elem_size) noexcept                                      \
            : SOAView{0u, buffer, soa_offset, soa_size, elem_offset, elem_size} {}                  \
                                                                                                    \
    public:                                                                                         \
        using detail::SOAViewBase<S>::operator->;                                                   \
    };                                                                                              \
    }

#define LUISA_SOA_EXPR_MAKE_MEMBER_DECL(m) \
    Expr<SOA<member_type_##m>> m;

#define LUISA_SOA_EXPR_MAKE_MEMBER_INIT(m) \
    m(buffer, soa_offset + _accumulate_soa_offset<member_type_##m>(soa_offset_accum, soa_size), soa_size, elem_offset)

#define LUISA_SOA_EXPR_MAKE_MEMBER_READ(m) \
    this->m.read(i)

#define LUISA_SOA_EXPR_MAKE_MEMBER_WRITE(m) \
    this->m.write(i, value.m);

#define LUISA_DERIVE_SOA_EXPR(S, ...)                                                                      \
    namespace luisa::compute {                                                                             \
    template<>                                                                                             \
    struct Expr<SOA<S>> : public detail::SOAExprBase {                                                     \
    private:                                                                                               \
        using this_type = S;                                                                               \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                              \
                                                                                                           \
    public:                                                                                                \
        LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_DECL, __VA_ARGS__)                                            \
                                                                                                           \
    private:                                                                                               \
        template<typename T>                                                                               \
        [[nodiscard]] static auto _accumulate_soa_offset(Var<uint> &accum, Expr<uint> soa_size) noexcept { \
            auto offset = accum;                                                                           \
            accum += SOA<T>::compute_soa_size(soa_size);                                                   \
            return offset;                                                                                 \
        }                                                                                                  \
        Expr(Var<uint> soa_offset_accum,                                                                   \
             Expr<Buffer<uint>> buffer,                                                                    \
             Expr<uint> soa_offset,                                                                        \
             Expr<uint> soa_size,                                                                          \
             Expr<uint> elem_offset) noexcept                                                              \
            : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset},                              \
              LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_INIT, __VA_ARGS__) {}                              \
                                                                                                           \
    public:                                                                                                \
        Expr(Expr<Buffer<uint>> buffer,                                                                    \
             Expr<uint> soa_offset,                                                                        \
             Expr<uint> soa_size,                                                                          \
             Expr<uint> elem_offset) noexcept                                                              \
            : Expr{def(0u), buffer, soa_offset, soa_size, elem_offset} {}                                  \
                                                                                                           \
        Expr(SOAView<S> soa) noexcept                                                                      \
            : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {}                \
                                                                                                           \
        Expr(const SOA<S> &soa) noexcept                                                                   \
            : Expr{soa.view()} {}                                                                          \
                                                                                                           \
        template<typename I>                                                                               \
        [[nodiscard]] auto read(I &&index) const noexcept {                                                \
            auto i = dsl::def(std::forward<I>(index));                                                     \
            return dsl::def<S>(LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_READ, __VA_ARGS__));              \
        }                                                                                                  \
                                                                                                           \
        template<typename I>                                                                               \
        [[nodiscard]] auto write(I &&index, Expr<S> value) const noexcept {                                \
            auto i = dsl::def(std::forward<I>(index));                                                     \
            LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_WRITE, __VA_ARGS__)                                       \
        }                                                                                                  \
                                                                                                           \
        [[nodiscard]] auto operator->() const noexcept { return this; }                                    \
    };                                                                                                     \
    }

#define LUISA_DERIVE_SOA(S, ...)          \
    LUISA_DERIVE_SOA_VIEW(S, __VA_ARGS__) \
    LUISA_DERIVE_SOA_EXPR(S, __VA_ARGS__)

#define LUISA_STRUCT(S, ...)                \
    LUISA_DERIVE_FMT(S, S, __VA_ARGS__)     \
    LUISA_STRUCT_REFLECT(S, __VA_ARGS__)    \
    template<>                              \
    struct luisa_compute_extension<S>;      \
    LUISA_DERIVE_DSL_STRUCT(S, __VA_ARGS__) \
    LUISA_DERIVE_SOA(S, __VA_ARGS__)        \
    template<>                              \
    struct luisa_compute_extension<S> final : luisa::compute::detail::Ref<S>

#define LUISA_CUSTOM_STRUCT_EXT(S)                                                           \
    template<>                                                                               \
    struct luisa_compute_extension<S>;                                                       \
    namespace luisa::compute {                                                               \
    template<>                                                                               \
    struct Expr<S> {                                                                         \
    private:                                                                                 \
        using this_type = S;                                                                 \
        const Expression *_expression;                                                       \
                                                                                             \
    public:                                                                                  \
        explicit Expr(const Expression *e) noexcept                                          \
            : _expression{e} {}                                                              \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }         \
        Expr(Expr &&another) noexcept = default;                                             \
        Expr(const Expr &another) noexcept = default;                                        \
        Expr &operator=(Expr) noexcept = delete;                                             \
    };                                                                                       \
    namespace detail {                                                                       \
    template<>                                                                               \
    struct Ref<S> {                                                                          \
    private:                                                                                 \
        using this_type = S;                                                                 \
        const Expression *_expression;                                                       \
                                                                                             \
    public:                                                                                  \
        explicit Ref(const Expression *e) noexcept                                           \
            : _expression{e} {}                                                              \
        [[nodiscard]] auto expression() const noexcept { return this->_expression; }         \
        Ref(Ref &&another) noexcept = default;                                               \
        Ref(const Ref &another) noexcept = default;                                          \
        [[nodiscard]] operator Expr<this_type>() const noexcept {                            \
            return Expr<this_type>{this->expression()};                                      \
        }                                                                                    \
        template<typename Rhs>                                                               \
        void operator=(Rhs &&rhs) & noexcept { dsl::assign(*this, std::forward<Rhs>(rhs)); } \
        void operator=(Ref rhs) & noexcept { (*this) = Expr{rhs}; }                          \
        [[nodiscard]] auto operator->() noexcept {                                           \
            return reinterpret_cast<luisa_compute_extension<this_type> *>(this);             \
        }                                                                                    \
        [[nodiscard]] auto operator->() const noexcept {                                     \
            return reinterpret_cast<const luisa_compute_extension<this_type> *>(this);       \
        }                                                                                    \
    };                                                                                       \
    }                                                                                        \
    }                                                                                        \
    template<>                                                                               \
    struct luisa_compute_extension<S> final : luisa::compute::detail::Ref<S>
