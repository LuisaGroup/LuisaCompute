#pragma once

#include <serialize/Common.h>

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
