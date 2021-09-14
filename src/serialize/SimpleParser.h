#pragma once

#include <serialize/Common.h>

namespace toolhub::db {

struct ParsingException {
    luisa::string message;
    ParsingException() noexcept = default;
    explicit ParsingException(luisa::string msg)
        : message(std::move(msg)) {}
};

}// namespace toolhub::db
