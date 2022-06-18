//
// Created by Mike Smith on 2022/6/18.
//

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <core/stl.h>

namespace luisa {

using json = nlohmann::basic_json<
    std::map, luisa::vector, luisa::string,
    bool, std::int64_t, std::uint64_t, double,
    luisa::allocator, ::nlohmann::adl_serializer,
    luisa::vector<std::uint8_t>>;

}
