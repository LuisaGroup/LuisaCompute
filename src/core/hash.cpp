//
// Created by Mike on 3/14/2021.
//

#include <array>
#include <string_view>
#include <core/hash.h>

namespace luisa {

std::string_view hash_to_string(uint64_t hash) noexcept {
    static thread_local std::array<char, 16u> temp;
    fmt::format_to_n(temp.data(), temp.size(), "{:016X}", hash);
    return std::string_view{temp.data(), temp.size()};
}

}// namespace luisa
