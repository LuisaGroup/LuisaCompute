//
// Created by Mike on 5/29/2023.
//

#pragma once

#include <core/dll_export.h>
#include <core/stl/string.h>
#include <core/stl/vector.h>

#include <ast/usage.h>
#include <ast/type.h>

namespace luisa::compute {

class LC_AST_API ExternalFunction {

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
