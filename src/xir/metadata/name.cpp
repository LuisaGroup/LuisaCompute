#include <luisa/core/logging.h>
#include <luisa/xir/metadata/name.h>

namespace luisa::compute::xir {

namespace detail {

inline void name_metadata_validate(luisa::string_view name) noexcept {
    if (!name.empty()) {
        static constexpr auto is_valid_name_head = [](char c) noexcept {
            return std::isalpha(c) || c == '_';
        };
        static constexpr auto is_valid_name_tail = [](char c) noexcept {
            return std::isalnum(c) || c == '_' || c == '.';
        };
        LUISA_ASSERT(is_valid_name_head(name.front()) &&
                         std::all_of(name.begin() + 1, name.end(), is_valid_name_tail),
                     "Invalid name '{}'.", name);
    }
}

}// namespace detail

NameMD::NameMD(luisa::string name) noexcept
    : _name{std::move(name)} {
    detail::name_metadata_validate(_name);
}

void NameMD::set_name(luisa::string_view name) noexcept {
    detail::name_metadata_validate(name);
    _name = name;
}

ManagedPtr<Metadata> NameMD::clone() const noexcept {
    return luisa::make_managed<NameMD>(name());
}

}// namespace luisa::compute::xir
