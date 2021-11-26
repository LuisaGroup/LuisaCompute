#pragma once
#include <vstl/Common.h>
#include <serde/IJsonObject.h>
namespace toolhub::db {
class Database {
public:
	virtual IJsonDatabase* CreateDatabase() const = 0;
	virtual IJsonDatabase* CreateDatabase(
		vstd::function<void*(size_t)> const& allocFunc) const = 0;
};
// Entry:
// Database const *Database_GetFactory();
#ifdef VENGINE_DATABASE_PROJECT
class Database_Impl final : public Database {
public:
	IJsonDatabase* CreateDatabase() const override;
	IJsonDatabase* CreateDatabase(
		vstd::function<void*(size_t)> const& allocFunc) const override;
};
#endif
}// namespace toolhub::db