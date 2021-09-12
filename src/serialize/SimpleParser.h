#pragma once

#include <serialize/Common.h>

namespace toolhub::db {

struct ParsingException {
    std::string message;
    ParsingException() noexcept = default;
    explicit ParsingException(std::string &&msg)
        : message(std::move(msg)) {}
};

}// namespace toolhub::db
