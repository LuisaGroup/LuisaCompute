#pragma once
#include <util/serde/Common.h>
// Entry:
// toolhub::db::Database const* Database_GetFactory()
namespace toolhub::db {
class IJsonDatabase;
class Database {
public:
	virtual IJsonDatabase* CreateDatabase() const = 0;
	virtual IJsonDatabase* CreateConcurrentDatabase() const = 0;
};
}// namespace toolhub::db