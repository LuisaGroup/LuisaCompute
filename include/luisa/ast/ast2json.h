#pragma once
//
// Created by Mike on 8/29/2023.
//

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/ast/function.h>

namespace luisa::compute {

constexpr uint32_t ast_json_schema_version = 1u;
constexpr luisa::string_view ast_json_indirect_dispatch_buffer_type_name =
    "LC_IndirectDispatchBuffer";
constexpr luisa::string_view ast_json_ray_query_all_type_name =
    "LC_RayQueryAll";
constexpr luisa::string_view ast_json_ray_query_any_type_name =
    "LC_RayQueryAny";

struct ASTJsonLimits {
    size_t max_document_bytes{64u * 1024u * 1024u};
    size_t max_parse_memory_bytes{256u * 1024u * 1024u};
    size_t max_functions{256u};
    size_t max_types{4096u};
    size_t max_nodes{1u << 20u};
    size_t max_depth{256u};
    size_t max_constant_bytes{16u * 1024u * 1024u};
    size_t max_string_bytes{1u * 1024u * 1024u};
};

class LUISA_AST_API ASTJsonBindingResolver {

public:
    virtual ~ASTJsonBindingResolver() noexcept = default;

    [[nodiscard]] virtual bool resolve_buffer(
        const Type *serialized_type, uint64_t serialized_handle,
        size_t serialized_offset,
        size_t serialized_size, Function::BufferBinding &binding,
        luisa::string &error) const noexcept;

    [[nodiscard]] virtual bool resolve_texture(
        uint64_t serialized_handle, uint32_t serialized_level,
        Function::TextureBinding &binding,
        luisa::string &error) const noexcept;

    [[nodiscard]] virtual bool resolve_bindless_array(
        uint64_t serialized_handle, Function::BindlessArrayBinding &binding,
        luisa::string &error) const noexcept;

    [[nodiscard]] virtual bool resolve_accel(
        uint64_t serialized_handle, Function::AccelBinding &binding,
        luisa::string &error) const noexcept;
};

struct ASTJsonEncodeResult {
    luisa::string json;
    luisa::string error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty();
    }
};

struct ASTJsonDecodeResult {
    luisa::shared_ptr<const detail::FunctionBuilder> function;
    luisa::string error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return function != nullptr && error.empty();
    }
};

[[nodiscard]] LUISA_AST_API luisa::string to_json(const Type *type) noexcept;
[[nodiscard]] LUISA_AST_API luisa::string to_json(Function function) noexcept;

[[nodiscard]] LUISA_AST_API ASTJsonEncodeResult try_to_json(
    Function function, const ASTJsonLimits &limits = {}) noexcept;

[[nodiscard]] LUISA_AST_API ASTJsonDecodeResult from_json(
    luisa::string_view json, const ASTJsonLimits &limits = {},
    const ASTJsonBindingResolver *binding_resolver = nullptr) noexcept;

}// namespace luisa::compute
