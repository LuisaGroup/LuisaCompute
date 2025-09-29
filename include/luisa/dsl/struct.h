#pragma once

#include <cstdint>
#include <cstddef>

#include <luisa/core/stl/format.h>
#include <luisa/dsl/soa.h>
#include <luisa/dsl/func.h>
#include <luisa/runtime/shader.h>

template<typename T>
struct luisa_compute_extension {};

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

#define LUISA_DERIVE_FMT(Struct, DisplayName, ...)                                          \
    template<>                                                                              \
    struct fmt::formatter<Struct> {                                                         \
        constexpr auto parse(format_parse_context &ctx) const -> decltype(ctx.begin()) {    \
            return ctx.end();                                                               \
        }                                                                                   \
        template<typename FormatContext>                                                    \
        auto format(const Struct &input, FormatContext &ctx) const -> decltype(ctx.out()) { \
            return fmt::format_to(ctx.out(), FMT_STRING(#DisplayName "{{ {} }}"),           \
                                  fmt::join(std::array{LUISA_MAP_LIST(                      \
                                                LUISA_DERIVE_FMT_MAP_STRUCT_FIELD,          \
                                                __VA_ARGS__)},                              \
                                            ", "));                                         \
        }                                                                                   \
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
    struct Ref<S>                                                                             \
        : public detail::ExprEnableBitwiseCast<Ref<S>>,                                       \
          public detail::RefEnableGetAddress<Ref<S>> {                                        \
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

#define LUISA_DERIVE_FMT_TEMPLATE(TEMPLATE, S, DisplayName, ...)                                         \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                         \
    struct fmt::formatter<LUISA_MACRO_EVAL(S())> {                                                       \
        constexpr auto parse(format_parse_context &ctx) const -> decltype(ctx.begin()) {                 \
            return ctx.end();                                                                            \
        }                                                                                                \
        template<typename FormatContext>                                                                 \
        auto format(const LUISA_MACRO_EVAL(S()) & input, FormatContext &ctx) const                       \
            -> decltype(ctx.out()) {                                                                     \
            return fmt::format_to(ctx.out(),                                                             \
                                  FMT_STRING(#DisplayName "{{ {} }}"),                                   \
                                  fmt::join(std::array{LUISA_MAP_LIST(LUISA_DERIVE_FMT_MAP_STRUCT_FIELD, \
                                                                      __VA_ARGS__)},                     \
                                            ", "));                                                      \
        }                                                                                                \
    };

#define LUISA_DERIVE_DSL_STRUCT_TEMPLATE(TEMPLATE, S, ...)                                                       \
    namespace luisa::compute {                                                                                   \
    namespace detail {                                                                                           \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                                 \
    class AtomicRef<LUISA_MACRO_EVAL(S())> : private AtomicRefBase {                                             \
    private:                                                                                                     \
        using this_type = LUISA_MACRO_EVAL(S());                                                                 \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, __VA_ARGS__)                                                    \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{                                                     \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                                   \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL, __VA_ARGS__)                                         \
        explicit AtomicRef(const AtomicRefNode *node) noexcept                                                   \
            : AtomicRefBase{node} {                                                                              \
        }                                                                                                        \
    };                                                                                                           \
    }                                                                                                            \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                                 \
    struct Expr<LUISA_MACRO_EVAL(S())>                                                                           \
        : public detail::ExprEnableBitwiseCast<Expr<LUISA_MACRO_EVAL(S())>> {                                    \
    private:                                                                                                     \
        using this_type = LUISA_MACRO_EVAL(S());                                                                 \
        const Expression *_expression;                                                                           \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, ##__VA_ARGS__)                                                  \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{                                                     \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                                   \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)                                               \
        explicit Expr(const Expression *e) noexcept                                                              \
            : _expression{e}, LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {                       \
        }                                                                                                        \
        [[nodiscard]] auto expression() const noexcept {                                                         \
            return this->_expression;                                                                            \
        }                                                                                                        \
        Expr(Expr &&another) noexcept = default;                                                                 \
        Expr(const Expr &another) noexcept = default;                                                            \
        Expr &operator=(Expr) noexcept = delete;                                                                 \
        template<size_t i>                                                                                       \
        [[nodiscard]] auto get() const noexcept {                                                                \
            using M =                                                                                            \
                std::tuple_element_t<i, struct_member_tuple_t<LUISA_MACRO_EVAL(S())>>;                           \
            return Expr<M>{detail::FunctionBuilder::current()->member(                                           \
                Type::of<M>(), this->expression(), i)};                                                          \
        };                                                                                                       \
    };                                                                                                           \
    namespace detail {                                                                                           \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                                 \
    struct Ref<LUISA_MACRO_EVAL(S())>                                                                            \
        : public detail::ExprEnableBitwiseCast<Ref<LUISA_MACRO_EVAL(S())>>,                                      \
          public detail::RefEnableGetAddress<Ref<LUISA_MACRO_EVAL(S())>> {                                       \
    private:                                                                                                     \
        using this_type = LUISA_MACRO_EVAL(S());                                                                 \
        const Expression *_expression;                                                                           \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, ##__VA_ARGS__)                                                  \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept {                    \
            constexpr const std::string_view member_names[]{                                                     \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                                   \
            return std::find(std::begin(member_names), std::end(member_names), name) - std::begin(member_names); \
        }                                                                                                        \
                                                                                                                 \
    public:                                                                                                      \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_REF_DECL, __VA_ARGS__)                                                \
        explicit Ref(const Expression *e) noexcept                                                               \
            : _expression{e}, LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__) {                       \
        }                                                                                                        \
        [[nodiscard]] auto expression() const noexcept {                                                         \
            return this->_expression;                                                                            \
        }                                                                                                        \
        Ref(Ref &&another) noexcept = default;                                                                   \
        Ref(const Ref &another) noexcept = default;                                                              \
        [[nodiscard]] operator Expr<LUISA_MACRO_EVAL(S())>() const noexcept {                                    \
            return Expr<LUISA_MACRO_EVAL(S())>{this->expression()};                                              \
        }                                                                                                        \
        template<typename Rhs>                                                                                   \
        void operator=(Rhs &&rhs) & noexcept {                                                                   \
            dsl::assign(*this, std::forward<Rhs>(rhs));                                                          \
        }                                                                                                        \
        void operator=(Ref rhs) & noexcept {                                                                     \
            (*this) = Expr{rhs};                                                                                 \
        }                                                                                                        \
        template<size_t i>                                                                                       \
        [[nodiscard]] auto get() const noexcept {                                                                \
            using M =                                                                                            \
                std::tuple_element_t<i, struct_member_tuple_t<LUISA_MACRO_EVAL(S())>>;                           \
            return Ref<M>{detail::FunctionBuilder::current()->member(                                            \
                Type::of<M>(), this->expression(), i)};                                                          \
        };                                                                                                       \
        [[nodiscard]] auto operator->() noexcept {                                                               \
            return reinterpret_cast<luisa_compute_extension<LUISA_MACRO_EVAL(S())> *>(this);                     \
        }                                                                                                        \
        [[nodiscard]] auto operator->() const noexcept {                                                         \
            return reinterpret_cast<const luisa_compute_extension<LUISA_MACRO_EVAL(S())> *>(this);               \
        }                                                                                                        \
    };                                                                                                           \
    }                                                                                                            \
    }

#define LUISA_DERIVE_SOA_VIEW_TEMPLATE(TEMPLATE, S, ...)                                                                                                                       \
    namespace luisa::compute {                                                                                                                                                 \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                                                                                               \
    class SOAView<LUISA_MACRO_EVAL(S())>                                                                                                                                       \
        : public detail::SOAViewBase<LUISA_MACRO_EVAL(S())> {                                                                                                                  \
                                                                                                                                                                               \
    private:                                                                                                                                                                   \
        using this_type = LUISA_MACRO_EVAL(S());                                                                                                                               \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, ##__VA_ARGS__)                                                                                                                \
                                                                                                                                                                               \
    public:                                                                                                                                                                    \
        [[nodiscard]] static auto compute_soa_size(auto soa_size) noexcept {                                                                                                   \
            return LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_SOA_SIZE_ACCUM, __VA_ARGS__)                                                                                           \
            0u;                                                                                                                                                                \
        }                                                                                                                                                                      \
                                                                                                                                                                               \
    public:                                                                                                                                                                    \
        LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_DECL, __VA_ARGS__)                                                                                                                \
                                                                                                                                                                               \
    private:                                                                                                                                                                   \
        template<typename T>                                                                                                                                                   \
        [[nodiscard]] static auto _accumulate_soa_offset(size_t &accum, size_t soa_size) noexcept {                                                                            \
            auto offset = accum;                                                                                                                                               \
            accum += SOAView<T>::compute_soa_size(soa_size);                                                                                                                   \
            return offset;                                                                                                                                                     \
        }                                                                                                                                                                      \
                                                                                                                                                                               \
        SOAView(size_t soa_offset_accum,                                                                                                                                       \
                BufferView<uint> buffer,                                                                                                                                       \
                size_t soa_offset,                                                                                                                                             \
                size_t soa_size,                                                                                                                                               \
                size_t elem_offset,                                                                                                                                            \
                size_t elem_size) noexcept                                                                                                                                     \
            : detail::SOAViewBase<LUISA_MACRO_EVAL(S())>{buffer, soa_offset, soa_size, elem_offset, elem_size}, LUISA_MAP_LIST(LUISA_SOA_VIEW_MAKE_MEMBER_INIT, __VA_ARGS__) { \
        }                                                                                                                                                                      \
                                                                                                                                                                               \
    public:                                                                                                                                                                    \
        SOAView(BufferView<uint> buffer,                                                                                                                                       \
                size_t soa_offset,                                                                                                                                             \
                size_t soa_size,                                                                                                                                               \
                size_t elem_offset,                                                                                                                                            \
                size_t elem_size) noexcept                                                                                                                                     \
            : SOAView{0u, buffer, soa_offset, soa_size, elem_offset, elem_size} {                                                                                              \
        }                                                                                                                                                                      \
                                                                                                                                                                               \
    public:                                                                                                                                                                    \
        using detail::SOAViewBase<LUISA_MACRO_EVAL(S())>::operator->;                                                                                                          \
    };                                                                                                                                                                         \
    }

#define LUISA_DERIVE_SOA_EXPR_TEMPLATE(TEMPLATE, S, ...)                                                                                     \
    namespace luisa::compute {                                                                                                               \
    LUISA_MACRO_EVAL(TEMPLATE())                                                                                                             \
    struct Expr<SOA<LUISA_MACRO_EVAL(S())>> : public detail::SOAExprBase {                                                                   \
    private:                                                                                                                                 \
        using this_type = LUISA_MACRO_EVAL(S());                                                                                             \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_TYPE, ##__VA_ARGS__)                                                                              \
                                                                                                                                             \
    public:                                                                                                                                  \
        LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_DECL, __VA_ARGS__)                                                                              \
                                                                                                                                             \
    private:                                                                                                                                 \
        template<typename T>                                                                                                                 \
        [[nodiscard]] static auto _accumulate_soa_offset(Var<uint> &accum,                                                                   \
                                                         Expr<uint> soa_size) noexcept {                                                     \
            auto offset = accum;                                                                                                             \
            accum += SOA<T>::compute_soa_size(soa_size);                                                                                     \
            return offset;                                                                                                                   \
        }                                                                                                                                    \
        Expr(Var<uint> soa_offset_accum,                                                                                                     \
             Expr<Buffer<uint>> buffer,                                                                                                      \
             Expr<uint> soa_offset,                                                                                                          \
             Expr<uint> soa_size,                                                                                                            \
             Expr<uint> elem_offset) noexcept                                                                                                \
            : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset}, LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_INIT, __VA_ARGS__) { \
        }                                                                                                                                    \
                                                                                                                                             \
    public:                                                                                                                                  \
        Expr(Expr<Buffer<uint>> buffer,                                                                                                      \
             Expr<uint> soa_offset,                                                                                                          \
             Expr<uint> soa_size,                                                                                                            \
             Expr<uint> elem_offset) noexcept                                                                                                \
            : Expr{def(0u), buffer, soa_offset, soa_size, elem_offset} {                                                                     \
        }                                                                                                                                    \
                                                                                                                                             \
        Expr(SOAView<LUISA_MACRO_EVAL(S())> soa) noexcept                                                                                    \
            : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()} {                                                   \
        }                                                                                                                                    \
                                                                                                                                             \
        Expr(const SOA<LUISA_MACRO_EVAL(S())> &soa) noexcept                                                                                 \
            : Expr{soa.view()} {                                                                                                             \
        }                                                                                                                                    \
                                                                                                                                             \
        template<typename I>                                                                                                                 \
        [[nodiscard]] auto read(I &&index) const noexcept {                                                                                  \
            auto i = dsl::def(std::forward<I>(index));                                                                                       \
            return dsl::def<LUISA_MACRO_EVAL(S())>(                                                                                          \
                LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_READ, __VA_ARGS__));                                                               \
        }                                                                                                                                    \
                                                                                                                                             \
        template<typename I>                                                                                                                 \
        [[nodiscard]] auto write(I &&index, Expr<LUISA_MACRO_EVAL(S())> value) const noexcept {                                              \
            auto i = dsl::def(std::forward<I>(index));                                                                                       \
            LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_WRITE, __VA_ARGS__)                                                                         \
        }                                                                                                                                    \
                                                                                                                                             \
        [[nodiscard]] auto operator->() const noexcept {                                                                                     \
            return this;                                                                                                                     \
        }                                                                                                                                    \
    };                                                                                                                                       \
    }

#define LUISA_DERIVE_SOA_TEMPLATE(TEMPLATE, S, ...)          \
    LUISA_DERIVE_SOA_VIEW_TEMPLATE(TEMPLATE, S, __VA_ARGS__) \
    LUISA_DERIVE_SOA_EXPR_TEMPLATE(TEMPLATE, S, __VA_ARGS__)

#define LUISA_TEMPLATE_STRUCT(TEMPLATE, S, ...)                 \
    LUISA_DERIVE_FMT_TEMPLATE(TEMPLATE, S, S, __VA_ARGS__)      \
    LUISA_STRUCT_REFLECT_TEMPLATE(TEMPLATE, S, __VA_ARGS__)     \
    LUISA_MACRO_EVAL(TEMPLATE())                                \
    struct luisa_compute_extension<LUISA_MACRO_EVAL(S())>;      \
    LUISA_DERIVE_DSL_STRUCT_TEMPLATE(TEMPLATE, S, __VA_ARGS__)  \
    LUISA_DERIVE_SOA_TEMPLATE(TEMPLATE, S, __VA_ARGS__)         \
    LUISA_MACRO_EVAL(TEMPLATE())                                \
    struct luisa_compute_extension<LUISA_MACRO_EVAL(S())> final \
        : luisa::compute::detail::Ref<LUISA_MACRO_EVAL(S())>

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
        explicit Ref(const Expression *e) noexcept : _expression{e} {}                       \
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

namespace luisa::compute::detail {

LC_DSL_API void luisa_compute_check_matrix_size(uint32_t idx, uint32_t max_size);

// !!! TO MAXWELL: DEFINING GLOBAL FUNCTIONS WITH NAME STARTING WITH UNDERSCORE IS UNDEFINED BEHAVIOR !!!
#define LUISA_DECL_MUL(TT, dim)                                                                                            \
    LC_DSL_API Var<TT##dim##x##dim> luisa_compute_mul_##TT##dim##x##dim(Expr<TT##dim##x##dim> a, Expr<TT##dim##x##dim> b); \
    LC_DSL_API Var<TT##dim> luisa_compute_mul_##TT##dim##x##dim(Expr<TT##dim##x##dim> a, Expr<TT##dim> b);

#define LUISA_DECL_MUL_ALL(TT) \
    LUISA_DECL_MUL(TT, 2)      \
    LUISA_DECL_MUL(TT, 3)      \
    LUISA_DECL_MUL(TT, 4)

LUISA_DECL_MUL_ALL(half)
LUISA_DECL_MUL_ALL(double)

#undef LUISA_DECL_MUL_ALL
#undef LUISA_DECL_MUL

}// namespace luisa::compute::detail

#define LUISA_MATRIX_FUNCTIONS(T, FuncName, max_size)                                                          \
    [[nodiscard]] auto operator*(luisa::compute::Expr<luisa::compute::Matrix<T, max_size>> b) const noexcept { \
        return luisa::compute::detail::luisa_compute_mul_##FuncName##max_size##x##max_size(*this, b);          \
    }                                                                                                          \
    [[nodiscard]] auto operator*(luisa::compute::Expr<luisa::compute::Vector<T, max_size>> b) const noexcept { \
        return luisa::compute::detail::luisa_compute_mul_##FuncName##max_size##x##max_size(*this, b);          \
    }

LUISA_STRUCT(luisa::double2x2, cols) {
    LUISA_MATRIX_FUNCTIONS(double, double, 2)
};

LUISA_STRUCT(luisa::double3x3, cols) {
    LUISA_MATRIX_FUNCTIONS(double, double, 3)
};

LUISA_STRUCT(luisa::double4x4, cols) {
    LUISA_MATRIX_FUNCTIONS(double, double, 4)
};

LUISA_STRUCT(luisa::half2x2, cols) {
    LUISA_MATRIX_FUNCTIONS(luisa::half, half, 2)
};

LUISA_STRUCT(luisa::half3x3, cols) {
    LUISA_MATRIX_FUNCTIONS(luisa::half, half, 3)
};

LUISA_STRUCT(luisa::half4x4, cols) {
    LUISA_MATRIX_FUNCTIONS(luisa::half, half, 4)
};

#define LUISA_EXPR(value) \
    detail::extract_expression(std::forward<decltype(value)>(value))

namespace luisa::compute {

#define LUISA_MAT_ACCESS(v, element_type, index, matrix_size)                      \
    detail::FunctionBuilder::current()->access(                                    \
        Type::of<element_type>(),                                                  \
        detail::FunctionBuilder::current()->member(                                \
            Type::array(Type::of<element_type>(), matrix_size), LUISA_EXPR(v), 0), \
        detail::FunctionBuilder::current()->literal(Type::of<uint>(), uint(index)))

/// Make double2x2 from 2 column vector double2
template<typename C0, typename C1>
    requires any_dsl_v<C0, C1> &&
             is_same_expr_v<C0, double2> &&
             is_same_expr_v<C1, double2>
[[nodiscard]] inline auto make_double2x2(
    C0 &&c0, C1 &&c1) noexcept {

    Var<double2x2> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double2, 0, 2),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double2, 1, 2),
        LUISA_EXPR(c1));
    return mat;
}

/// Make double2x2 [ [M00, M10], [M01, M11] ]
template<typename M00, typename M01, typename M10, typename M11>
    requires any_dsl_v<M00, M01, M10, M11> &&
             is_floating_point_expr_v<M00> &&
             is_floating_point_expr_v<M01> &&
             is_floating_point_expr_v<M10> &&
             is_floating_point_expr_v<M11>
[[nodiscard]] inline auto make_double2x2(
    M00 &&m00, M01 &&m01,
    M10 &&m10, M11 &&m11) noexcept {
    return make_double2x2(
        make_double2(m00, m01),
        make_double2(m10, m11));
}

/// Make double2x2 from matrix.
/// Submatrix will be taken if matrix is larger than 2x2.
template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_double2x2(M &&m) noexcept {
    return make_double2x2(
        make_double2(m[0]),
        make_double2(m[1]));
}

/// Make double3x3 from 3 column vector double3
template<typename C0, typename C1, typename C2>
    requires any_dsl_v<C0, C1, C2> &&
             is_same_expr_v<C0, double3> &&
             is_same_expr_v<C1, double3> &&
             is_same_expr_v<C2, double3>
[[nodiscard]] inline auto make_double3x3(C0 &&c0, C1 &&c1, C2 &&c2) noexcept {
    Var<double3x3> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double3, 0, 3),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double3, 1, 3),
        LUISA_EXPR(c1));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double3, 2, 3),
        LUISA_EXPR(c2));
    return mat;
}

/// Make double3x3 [ [M00, M10, M20], [M01, M11, M21], [M02, M12, M22] ]
template<typename M00, typename M01, typename M02,
         typename M10, typename M11, typename M12,
         typename M20, typename M21, typename M22>
    requires any_dsl_v<M00, M01, M02, M10, M11, M12, M20, M21, M22> &&
             is_same_expr_v<M00, double> && is_same_expr_v<M01, double> && is_same_expr_v<M02, double> &&
             is_same_expr_v<M10, double> && is_same_expr_v<M11, double> && is_same_expr_v<M12, double> &&
             is_same_expr_v<M20, double> && is_same_expr_v<M21, double> && is_same_expr_v<M22, double>
[[nodiscard]] inline auto make_double3x3(
    M00 &&m00, M01 &&m01, M02 &&m02,
    M10 &&m10, M11 &&m11, M12 &&m12,
    M20 &&m20, M21 &&m21, M22 &&m22) noexcept {
    return make_double3x3(
        make_double3(m00, m01, m02),
        make_double3(m10, m11, m12),
        make_double3(m20, m21, m22));
}

/// Make double3x3 from double2x2 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_double3x3(M &&m) noexcept {
    return make_double3x3(
        make_double3(m[0], 0.f),
        make_double3(m[1], 0.f),
        luisa::make_double3(0.f, 0.f, 1.f));
}

/// Make double3x3 from double3x3/double4x4
/// Submatrix will be taken if matrix is larger than 3x3.
template<typename M>
    requires is_dsl_v<M> && std::disjunction_v<is_matrix3_expr<M>, is_matrix4_expr<M>>
[[nodiscard]] inline auto make_double3x3(M &&m) noexcept {
    return make_double3x3(
        make_double3(m[0]),
        make_double3(m[1]),
        make_double3(m[2]));
}

/// Make double4x4 from 4 column vector double4
template<typename C0, typename C1, typename C2, typename C3>
    requires any_dsl_v<C0, C1, C2, C3> &&
             is_same_expr_v<C0, double4> &&
             is_same_expr_v<C1, double4> &&
             is_same_expr_v<C2, double4> &&
             is_same_expr_v<C3, double4>
[[nodiscard]] inline auto make_double4x4(
    C0 &&c0, C1 &&c1, C2 &&c2, C3 &&c3) noexcept {
    Var<double4x4> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double4, 0, 4),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double4, 1, 4),
        LUISA_EXPR(c1));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double4, 2, 4),
        LUISA_EXPR(c2));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, double4, 3, 4),
        LUISA_EXPR(c3));
    return mat;
}

/// Make double4x4 [ [M00, M10, M20, M30], [M01, M11, M21, M31], [M02, M12, M22, M32], [M03, M13, M23, M33] ]
template<
    typename M00, typename M01, typename M02, typename M03,
    typename M10, typename M11, typename M12, typename M13,
    typename M20, typename M21, typename M22, typename M23,
    typename M30, typename M31, typename M32, typename M33>
    requires any_dsl_v<
                 M00, M01, M02, M03,
                 M10, M11, M12, M13,
                 M20, M21, M22, M23,
                 M30, M31, M32, M33> &&
             is_same_expr_v<M00, double> && is_same_expr_v<M01, double> && is_same_expr_v<M02, double> && is_same_expr_v<M03, double> &&
             is_same_expr_v<M10, double> && is_same_expr_v<M11, double> && is_same_expr_v<M12, double> && is_same_expr_v<M13, double> &&
             is_same_expr_v<M20, double> && is_same_expr_v<M21, double> && is_same_expr_v<M22, double> && is_same_expr_v<M23, double> &&
             is_same_expr_v<M30, double> && is_same_expr_v<M31, double> && is_same_expr_v<M32, double> && is_same_expr_v<M33, double>
[[nodiscard]] inline auto make_double4x4(
    M00 &&m00, M01 &&m01, M02 &&m02, M03 &&m03,
    M10 &&m10, M11 &&m11, M12 &&m12, M13 &&m13,
    M20 &&m20, M21 &&m21, M22 &&m22, M23 &&m23,
    M30 &&m30, M31 &&m31, M32 &&m32, M33 &&m33) noexcept {
    return make_double4x4(
        make_double4(m00, m01, m02, m03),
        make_double4(m10, m11, m12, m13),
        make_double4(m20, m21, m22, m23),
        make_double4(m30, m31, m32, m33));
}

/// Make double4x4 from double2x2 [ [M, 0], [0, I] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_double4x4(M &&m) noexcept {
    return make_double4x4(
        make_double4(m[0], 0.f, 0.f),
        make_double4(m[1], 0.f, 0.f),
        luisa::make_double4(0.f, 0.f, 1.f, 0.f),
        luisa::make_double4(0.f, 0.f, 0.f, 1.f));
}

/// Make double4x4 from double3x3 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix3_expr_v<M>
[[nodiscard]] inline auto make_double4x4(M &&m) noexcept {
    return make_double4x4(
        make_double4(m[0], 0.f),
        make_double4(m[1], 0.f),
        make_double4(m[2], 0.f),
        luisa::make_double4(0.f, 0.f, 0.f, 1.f));
}

/// Make double4x4 from double4x4
template<typename M>
    requires is_dsl_v<M> && is_matrix4_expr_v<M>
[[nodiscard]] inline auto make_double4x4(M &&m) noexcept {
    return def(std::forward<M>(m));
}
/// Make half2x2 from 2 column vector half2
template<typename C0, typename C1>
    requires any_dsl_v<C0, C1> &&
             is_same_expr_v<C0, half2> &&
             is_same_expr_v<C1, half2>
[[nodiscard]] inline auto make_half2x2(
    C0 &&c0, C1 &&c1) noexcept {
    Var<half2x2> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half2, 0, 2),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half2, 1, 2),
        LUISA_EXPR(c1));
    return mat;
}

/// Make half2x2 [ [M00, M10], [M01, M11] ]
template<typename M00, typename M01, typename M10, typename M11>
    requires any_dsl_v<M00, M01, M10, M11> &&
             is_floating_point_expr_v<M00> &&
             is_floating_point_expr_v<M01> &&
             is_floating_point_expr_v<M10> &&
             is_floating_point_expr_v<M11>
[[nodiscard]] inline auto make_half2x2(
    M00 &&m00, M01 &&m01,
    M10 &&m10, M11 &&m11) noexcept {
    return make_half2x2(
        make_half2(m00, m01),
        make_half2(m10, m11));
}

/// Make half2x2 from matrix.
/// Submatrix will be taken if matrix is larger than 2x2.
template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_half2x2(M &&m) noexcept {
    return make_half2x2(
        make_half2(m[0]),
        make_half2(m[1]));
}

/// Make half3x3 from 3 column vector half3
template<typename C0, typename C1, typename C2>
    requires any_dsl_v<C0, C1, C2> &&
             is_same_expr_v<C0, half3> &&
             is_same_expr_v<C1, half3> &&
             is_same_expr_v<C2, half3>
[[nodiscard]] inline auto make_half3x3(C0 &&c0, C1 &&c1, C2 &&c2) noexcept {
    Var<half3x3> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half3, 0, 3),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half3, 1, 3),
        LUISA_EXPR(c1));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half3, 2, 3),
        LUISA_EXPR(c2));
    return mat;
}

/// Make half3x3 [ [M00, M10, M20], [M01, M11, M21], [M02, M12, M22] ]
template<typename M00, typename M01, typename M02,
         typename M10, typename M11, typename M12,
         typename M20, typename M21, typename M22>
    requires any_dsl_v<M00, M01, M02, M10, M11, M12, M20, M21, M22> &&
             is_same_expr_v<M00, half> && is_same_expr_v<M01, half> && is_same_expr_v<M02, half> &&
             is_same_expr_v<M10, half> && is_same_expr_v<M11, half> && is_same_expr_v<M12, half> &&
             is_same_expr_v<M20, half> && is_same_expr_v<M21, half> && is_same_expr_v<M22, half>
[[nodiscard]] inline auto make_half3x3(
    M00 &&m00, M01 &&m01, M02 &&m02,
    M10 &&m10, M11 &&m11, M12 &&m12,
    M20 &&m20, M21 &&m21, M22 &&m22) noexcept {
    return make_half3x3(
        make_half3(m00, m01, m02),
        make_half3(m10, m11, m12),
        make_half3(m20, m21, m22));
}

/// Make half3x3 from half2x2 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_half3x3(M &&m) noexcept {
    return make_half3x3(
        make_half3(m[0], 0._h),
        make_half3(m[1], 0._h),
        luisa::make_half3(0._h, 0._h, 1._h));
}

/// Make half3x3 from half3x3/half4x4
/// Submatrix will be taken if matrix is larger than 3x3.
template<typename M>
    requires is_dsl_v<M> && std::disjunction_v<is_matrix3_expr<M>, is_matrix4_expr<M>>
[[nodiscard]] inline auto make_half3x3(M &&m) noexcept {
    return make_half3x3(
        make_half3(m[0]),
        make_half3(m[1]),
        make_half3(m[2]));
}

/// Make half4x4 from 4 column vector half4
template<typename C0, typename C1, typename C2, typename C3>
    requires any_dsl_v<C0, C1, C2, C3> &&
             is_same_expr_v<C0, half4> &&
             is_same_expr_v<C1, half4> &&
             is_same_expr_v<C2, half4> &&
             is_same_expr_v<C3, half4>
[[nodiscard]] inline auto make_half4x4(
    C0 &&c0, C1 &&c1, C2 &&c2, C3 &&c3) noexcept {
    Var<half4x4> mat;
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half4, 0, 4),
        LUISA_EXPR(c0));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half4, 1, 4),
        LUISA_EXPR(c1));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half4, 2, 4),
        LUISA_EXPR(c2));
    detail::FunctionBuilder::current()->assign(
        LUISA_MAT_ACCESS(mat, half4, 3, 4),
        LUISA_EXPR(c3));
    return mat;
}

/// Make half4x4 [ [M00, M10, M20, M30], [M01, M11, M21, M31], [M02, M12, M22, M32], [M03, M13, M23, M33] ]
template<
    typename M00, typename M01, typename M02, typename M03,
    typename M10, typename M11, typename M12, typename M13,
    typename M20, typename M21, typename M22, typename M23,
    typename M30, typename M31, typename M32, typename M33>
    requires any_dsl_v<
                 M00, M01, M02, M03,
                 M10, M11, M12, M13,
                 M20, M21, M22, M23,
                 M30, M31, M32, M33> &&
             is_same_expr_v<M00, half> && is_same_expr_v<M01, half> && is_same_expr_v<M02, half> && is_same_expr_v<M03, half> &&
             is_same_expr_v<M10, half> && is_same_expr_v<M11, half> && is_same_expr_v<M12, half> && is_same_expr_v<M13, half> &&
             is_same_expr_v<M20, half> && is_same_expr_v<M21, half> && is_same_expr_v<M22, half> && is_same_expr_v<M23, half> &&
             is_same_expr_v<M30, half> && is_same_expr_v<M31, half> && is_same_expr_v<M32, half> && is_same_expr_v<M33, half>
[[nodiscard]] inline auto make_half4x4(
    M00 &&m00, M01 &&m01, M02 &&m02, M03 &&m03,
    M10 &&m10, M11 &&m11, M12 &&m12, M13 &&m13,
    M20 &&m20, M21 &&m21, M22 &&m22, M23 &&m23,
    M30 &&m30, M31 &&m31, M32 &&m32, M33 &&m33) noexcept {
    return make_half4x4(
        make_half4(m00, m01, m02, m03),
        make_half4(m10, m11, m12, m13),
        make_half4(m20, m21, m22, m23),
        make_half4(m30, m31, m32, m33));
}

/// Make half4x4 from half2x2 [ [M, 0], [0, I] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_half4x4(M &&m) noexcept {
    return make_half4x4(
        make_half4(m[0], 0._h, 0._h),
        make_half4(m[1], 0._h, 0._h),
        luisa::make_half4(0._h, 0._h, 1._h, 0._h),
        luisa::make_half4(0._h, 0._h, 0._h, 1._h));
}

/// Make half4x4 from half3x3 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix3_expr_v<M>
[[nodiscard]] inline auto make_half4x4(M &&m) noexcept {
    return make_half4x4(
        make_half4(m[0], 0._h),
        make_half4(m[1], 0._h),
        make_half4(m[2], 0._h),
        luisa::make_half4(0._h, 0._h, 0._h, 1._h));
}

/// Make half4x4 from half4x4
template<typename M>
    requires is_dsl_v<M> && is_matrix4_expr_v<M>
[[nodiscard]] inline auto make_half4x4(M &&m) noexcept {
    return def(std::forward<M>(m));
}
#define LUISA_MATRIX_INTRIN(TYPE, DIM)                                        \
    LC_DSL_API Var<TYPE##DIM##x##DIM> transpose(Expr<TYPE##DIM##x##DIM> mat); \
    LC_DSL_API Var<TYPE##DIM##x##DIM> inverse(Expr<TYPE##DIM##x##DIM> mat);   \
    LC_DSL_API Var<TYPE> determinant(Expr<TYPE##DIM##x##DIM> mat);

LUISA_MATRIX_INTRIN(double, 2)
LUISA_MATRIX_INTRIN(double, 3)
LUISA_MATRIX_INTRIN(double, 4)
LUISA_MATRIX_INTRIN(half, 2)
LUISA_MATRIX_INTRIN(half, 3)
LUISA_MATRIX_INTRIN(half, 4)

#undef LUISA_MAT_ACCESS
#undef LUISA_MATRIX_INTRIN
#undef LUISA_EXPR
}// namespace luisa::compute