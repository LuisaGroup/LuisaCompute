#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute {

template<typename T>
class Callable;

template<size_t dim, typename... T>
class Kernel;

class LUISA_AST_API CallableLibrary {
public:
    using CallableMap = luisa::unordered_map<luisa::string, luisa::shared_ptr<const detail::FunctionBuilder>>;

private:
    struct GraphWriteCtx;
    struct GraphReadCtx;
    [[nodiscard]] static bool _fg_collect(detail::FunctionBuilder const *fb, GraphWriteCtx &ctx) noexcept;
    [[nodiscard]] static bool _fg_write_func(detail::FunctionBuilder const &fb, GraphWriteCtx &ctx) noexcept;
    [[nodiscard]] static bool _fg_write_expr(Expression const *expr, detail::FunctionBuilder const *owner, GraphWriteCtx &ctx, uint32_t &out_id) noexcept;
    [[nodiscard]] static bool _fg_write_stmt(Statement const &stmt, detail::FunctionBuilder const *owner, GraphWriteCtx &ctx) noexcept;
    [[nodiscard]] static bool _fg_read_func(detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept;
    [[nodiscard]] static bool _fg_read_stmt_payload(Statement &stmt, Statement::Tag tag, detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept;
    [[nodiscard]] static Statement *_fg_read_stmt_ptr(detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept;
    [[nodiscard]] static bool _fg_read_stmt_into(Statement &stmt, detail::FunctionBuilder &fb, GraphReadCtx &ctx) noexcept;
    template<typename StmtT>
    [[nodiscard]] static luisa::unique_ptr<StmtT> _fg_alloc_stmt() noexcept;
    template<typename ExprT>
    [[nodiscard]] static luisa::unique_ptr<ExprT> _fg_alloc_expr() noexcept;
    struct DeserPackage {
        detail::FunctionBuilder *builder;
        luisa::unordered_map<uint64_t, luisa::shared_ptr<detail::FunctionBuilder>> callable_map;
    };
    CallableMap _callables;
    static void serialize_func_builder(detail::FunctionBuilder const &builder, luisa::vector<std::byte> &vec) noexcept;
    static void deserialize_func_builder(detail::FunctionBuilder &builder, std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept;
    template<typename T>
    static T deser_value(std::byte const *&ptr, DeserPackage &pack) noexcept;
    template<typename T>
    static void deser_ptr(T obj, std::byte const *&ptr, DeserPackage &pack) noexcept;

public:
    [[nodiscard]] bool empty() const { return _callables.empty(); }
    [[nodiscard]] auto const &callable_map() const { return _callables; }
    template<typename T>
    [[nodiscard]] Callable<T> get_callable(luisa::string_view name) const noexcept;
    template<size_t dim, typename... T>
    [[nodiscard]] Kernel<dim, T...> get_kernel(luisa::string_view name) const noexcept;
    [[nodiscard]] Function get_function(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::shared_ptr<const detail::FunctionBuilder> get_function_builder(luisa::string_view name) const noexcept;
    CallableLibrary() noexcept;
    void add_callable(luisa::string_view name, const luisa::shared_ptr<const detail::FunctionBuilder> &callable) noexcept;
    void load(luisa::span<const std::byte> binary) noexcept;
    [[nodiscard]] luisa::vector<std::byte> serialize() const noexcept;
    [[nodiscard]] static luisa::vector<std::byte> serialize_function_graph(
        luisa::span<const luisa::shared_ptr<const detail::FunctionBuilder>> roots) noexcept;
    [[nodiscard]] static bool deserialize_function_graph(
        luisa::span<const std::byte> binary,
        luisa::vector<luisa::shared_ptr<const detail::FunctionBuilder>> &out_roots,
        luisa::shared_ptr<const detail::FunctionBuilder::XIRCacheProvider> xir_provider = {},
        luisa::span<const size_t> root_xir_provider_indices = {}) noexcept;
    CallableLibrary(CallableLibrary const &) = delete;
    CallableLibrary(CallableLibrary &&) noexcept;
    ~CallableLibrary() noexcept;
};

}// namespace luisa::compute
