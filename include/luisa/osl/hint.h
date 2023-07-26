#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::osl {

class LC_OSL_API Hint {

private:
    luisa::string _identifier;
    luisa::vector<luisa::string> _args;

public:
    Hint(luisa::string identifier,
         luisa::vector<luisa::string> args) noexcept
        : _identifier{std::move(identifier)},
          _args{std::move(args)} {}
    [[nodiscard]] auto identifier() const noexcept { return luisa::string_view{_identifier}; }
    [[nodiscard]] auto args() const noexcept { return luisa::span{_args}; }

    // for debugging
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::osl
