#pragma once
#include <vstl/Common.h>
#include <serialize/INoSqlDatabase.h>
#include <serialize/IJsonObject.h>
// Entry:
// toolhub::db::Database const* Database_GetFactory()
namespace toolhub::db {
class Database {
public:
	virtual IJsonDatabase* CreateDatabase() const = 0;
	virtual INoSqlDatabase* CreateSimpleNoSql(vstd::string&& name) const = 0;
	virtual std::pair<
		vstd::unique_ptr<IJsonDict>,
		vstd::vector<INoSqlDatabase::KeyValue>>
	DBTableToDict(
		IJsonDatabase* db,
		INoSqlDatabase::Table const& table) const = 0;
	virtual std::pair<
		vstd::unique_ptr<IJsonArray>,
		vstd::vector<INoSqlDatabase::KeyValue>>
	DBTablesToArray(
		IJsonDatabase* db,
		vstd::IEnumerable<INoSqlDatabase::Table>* tables) const = 0;
};
}// namespace toolhub::db