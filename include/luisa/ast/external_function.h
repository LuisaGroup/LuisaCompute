#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include <luisa/ast/usage.h>
#include <luisa/ast/type.h>

namespace luisa::compute {

class CallableLibrary;

class LC_AST_API ExternalFunction {
    friend class CallableLibrary;

private:
    luisa::string _name;
    const Type *_return_type;
    uint64_t _hash;
    luisa::vector<const Type *> _argument_types;
    luisa::vector<Usage> _argument_usages;

private:
    void _compute_hash() noexcept;

public:
    ExternalFunction(luisa::string name,
                     const Type *return_type,
                     luisa::vector<const Type *> argument_types,
                     luisa::vector<Usage> argument_usages) noexcept;

    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }
    [[nodiscard]] auto hash() const noexcept { return _hash; }
    [[nodiscard]] auto return_type() const noexcept { return _return_type; }
    [[nodiscard]] auto argument_types() const noexcept { return luisa::span{_argument_types}; }
    [[nodiscard]] auto argument_usages() const noexcept { return luisa::span{_argument_usages}; }
};

}// namespace luisa::compute

