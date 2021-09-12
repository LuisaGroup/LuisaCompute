#pragma once

#include <serialize/Common.h>
#include <serialize/IJsonDatabase.h>
#include <serialize/IJsonObject.h>

namespace toolhub::db {

class IJsonDatabase;

class Database {
public:
    [[nodiscard]] virtual IJsonDatabase *CreateDatabase() const = 0;
    [[nodiscard]] virtual IJsonDatabase *CreateConcurrentDatabase() const = 0;
};

}// namespace toolhub::db

namespace luisa::serialize {
using DatabaseFactory = auto() -> const toolhub::db::Database *;
constexpr std::string_view database_factory_symbol = "Database_GetFactory";
}
