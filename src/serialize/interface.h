#pragma once

#include <serialize/Common.h>
#include <serialize/IJsonDatabase.h>
#include <serialize/IJsonObject.h>

namespace toolhub::db {

// Entry:
// toolhub::db::Database const* Database_GetFactory()

class IJsonDatabase;

class Database {
public:
    [[nodiscard]] virtual IJsonDatabase *CreateDatabase() const = 0;
    [[nodiscard]] virtual IJsonDatabase *CreateConcurrentDatabase() const = 0;
};

}// namespace toolhub::db

namespace luisa::serialize {
using toolhub::db::Database;
using toolhub::db::IJsonDatabase;
using DatabaseFactory = auto() -> const Database *;
constexpr std::string_view database_factory_symbol = "Database_GetFactory";
}
