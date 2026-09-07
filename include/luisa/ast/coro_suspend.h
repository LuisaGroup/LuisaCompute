#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {

class Expression;

enum class CoroSuspendFallback : uint8_t {
    ignore,
    warn,
    reject,
};

enum class CoroSuspendBindingAccess : uint8_t {
    read,
    write,
    read_write,
};

enum class CoroSuspendBindingLifetime : uint8_t {
    boundary,
    queued,
    resumed,
};

using CoroSuspendAttributeValue =
    luisa::variant<bool, int64_t, uint64_t, double, luisa::string>;

struct CoroSuspendAttribute {
    luisa::string name;
    CoroSuspendAttributeValue value;
};

// A binding is frontend-neutral. `index` is resolved by the owner: an AST
// SuspendStmt maps it to an Expression/access path, a CoroSuspendInst maps it
// to an XIR operand, and CoroGraph maps it to a materialized frame field.
struct CoroSuspendBinding {
    luisa::string name;
    CoroSuspendBindingAccess access{CoroSuspendBindingAccess::read};
    CoroSuspendBindingLifetime lifetime{
        CoroSuspendBindingLifetime::boundary};
    uint32_t index{0u};
};

class CoroSuspendExtension;
using CoroSuspendExtensionPtr = luisa::unique_ptr<CoroSuspendExtension>;

// Source-side extensions use this interface while freezing plugin-owned
// objects into owner-indexed, data-backed extension objects.
class LUISA_AST_API CoroSuspendExtensionRecorder {
public:
    virtual ~CoroSuspendExtensionRecorder() noexcept = default;
    [[nodiscard]] virtual uint32_t bind(
        CoroSuspendBinding binding,
        const Expression *value) noexcept = 0;
};

class LUISA_AST_API CoroSuspendExtension {
public:
    CoroSuspendExtension() noexcept = default;
    CoroSuspendExtension(const CoroSuspendExtension &) = delete;
    CoroSuspendExtension(CoroSuspendExtension &&) noexcept = default;
    CoroSuspendExtension &operator=(const CoroSuspendExtension &) = delete;
    CoroSuspendExtension &operator=(CoroSuspendExtension &&) noexcept = default;
    virtual ~CoroSuspendExtension() noexcept = default;

    [[nodiscard]] virtual luisa::string_view schema() const noexcept = 0;
    [[nodiscard]] virtual uint32_t version() const noexcept = 0;
    [[nodiscard]] virtual bool is_annotation() const noexcept {
        return false;
    }
    [[nodiscard]] virtual CoroSuspendFallback fallback() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const CoroSuspendBinding>
    bindings() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const CoroSuspendAttribute>
    attributes() const noexcept = 0;

    [[nodiscard]] virtual CoroSuspendExtensionPtr clone() const noexcept = 0;
    [[nodiscard]] virtual CoroSuspendExtensionPtr freeze(
        CoroSuspendExtensionRecorder &recorder) && noexcept = 0;
};

class LUISA_AST_API CoroSuspendAnnotation : public CoroSuspendExtension {
public:
    [[nodiscard]] bool is_annotation() const noexcept final { return true; }
};

// Construct normalized, data-backed representations. Attribute order is
// canonicalized and the complete object is independent of the source plugin.
[[nodiscard]] LUISA_AST_API CoroSuspendExtensionPtr
make_coro_suspend_extension_data(
    luisa::string schema, uint32_t version,
    CoroSuspendFallback fallback,
    luisa::vector<CoroSuspendBinding> bindings,
    luisa::vector<CoroSuspendAttribute> attributes) noexcept;

[[nodiscard]] LUISA_AST_API CoroSuspendExtensionPtr
make_coro_suspend_annotation_data(
    luisa::string schema, uint32_t version,
    CoroSuspendFallback fallback,
    luisa::vector<CoroSuspendBinding> bindings,
    luisa::vector<CoroSuspendAttribute> attributes) noexcept;

}// namespace luisa::compute
