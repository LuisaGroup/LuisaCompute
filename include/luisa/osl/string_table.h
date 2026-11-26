#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::osl {

using StringToken = uint32_t;
inline constexpr StringToken empty_string_token = 0u;
inline constexpr StringToken invalid_string_token = ~0u;

class LUISA_OSL_API StringTable {

private:
    luisa::vector<luisa::string> _strings;
    luisa::unordered_map<luisa::string, StringToken> _map;

public:
    StringTable() noexcept;
    ~StringTable() noexcept = default;
    StringTable(const StringTable &) noexcept = default;
    StringTable(StringTable &&) noexcept = default;
    StringTable &operator=(const StringTable &) noexcept = default;
    StringTable &operator=(StringTable &&) noexcept = default;

    [[nodiscard]] StringToken intern(luisa::string_view s) noexcept;
    [[nodiscard]] luisa::optional<StringToken> find(luisa::string_view s) const noexcept;
    [[nodiscard]] luisa::string_view lookup(StringToken token) const noexcept;
    [[nodiscard]] size_t size() const noexcept { return _strings.size(); }
    [[nodiscard]] bool empty() const noexcept { return _strings.empty(); }
    [[nodiscard]] luisa::span<const luisa::string> strings() const noexcept { return _strings; }
    void clear() noexcept;
};

}// namespace luisa::compute::osl
