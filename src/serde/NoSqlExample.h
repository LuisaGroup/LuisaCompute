#pragma once

#include <serde/DatabaseInclude.h>
#include <serde/INoSqlDatabase.h>
vstd::Guid testGuid(true);
template<size_t i, bool Delete>
void TestNoSql(toolhub::db::Database* dbBase, toolhub::db::CompareFlag flag) {
	using namespace toolhub::db;

	auto db = vstd::make_unique(dbBase->CreateSimpleNoSql(""));

	vstd::vector<INoSqlDatabase::KeyValue> kvs;
	auto BuildVec = [&](auto&& id, auto&& ver, auto&& name, auto&& ele) {
		kvs.clear();
		kvs.emplace_back("sid", id);
		kvs.emplace_back(testGuid, ver);
		kvs.emplace_back("sname", name);
		kvs.emplace_back(99999, ele);
	};
	auto Add = [&](auto&& id, auto&& ver, auto&& name, auto&& ele) {
		BuildVec(id, ver, name, ele);
		db->AddNode(vstd::get_rvalue_ptr(vstd::GetVectorIEnumerable_Move(kvs)));
		kvs.clear();
	};
	Add("sid1", 0, "sfuck1", true);
	Add("sid2", 1, "sfuck2", false);
	Add("sid3", 3, "[1,2,3,4,5]", nullptr);
	Add("sid4", 2, "sfuck3", "{66:\"fuck\"}");
	Add("sid5", 2, "sfuck3", false);
	Add("sid6", 3, "sfuck3", true);
	auto PrintValue = [&](INoSqlDatabase::ValueType const& v) {
		v.multi_visit(
			[](auto&& v) { std::cout << v << '\n'; },
			[](auto&& v) { std::cout << v << '\n'; },
			[](auto&& v) { std::cout << v << '\n'; },
			[](auto&& v) { std::cout << v.ToString() << '\n'; },
			[](auto&& v) { std::cout << (v ? "true" : "false") << '\n'; },
			[](auto&&) { std::cout << "null\n"; });
	};
	auto serde = vstd::make_unique(dbBase->CreateDatabase());
	auto PrintKV = [&](auto&& ite) {
		if (!ite)
			return;
		auto dict = dbBase->DBTableToDict(serde.get(), *ite);
		std::cout << dict.first->FormattedPrint() << '\n';
		std::cout << "illegal keys:\n";
		for (auto& i : dict.second) {
			i.first.visit(
				[](auto&& v) { std::cout << v << '\n'; });
		}
	};
	auto PrintAll = [&](auto&& ite) {
		for (auto&& vec : ite) {
			for (auto&& kv : vec) {
				std::cout << "key: ";
				kv.first.visit(
					[](auto&& v) { std::cout << v; });
				std::cout << "  value: ";
				PrintValue(kv.second);
			}
			std::cout << '\n';
		}
	};
	//Test Simple Add
	if constexpr (i == 0) {
		auto ite = db->GetAll();
		PrintAll(ite);
	}
	//Find One with kv
	else if constexpr (i == 1) {
		if constexpr (Delete) {
			auto kv = db->FindOneAndDelete({testGuid, 2, flag});
			PrintKV(kv);
		} else {
			auto kv = db->FindOne({testGuid, 2, flag});
			PrintKV(kv);
		}
	}
	//Find All with kv
	else if constexpr (i == 2) {
		if constexpr (Delete) {
			auto ite = db->FindAllAndDelete({testGuid, 2, flag});
			PrintAll(ite);
		} else {

			auto ite = db->FindAll({testGuid, 2, flag});
			PrintAll(ite);
		}
	}
	//Find One with kv vec
	else if constexpr (i == 3) {
		vstd::vector<INoSqlDatabase::CompareKey> kvs;
		kvs.emplace_back(testGuid, 3, flag);
		kvs.emplace_back(99999, nullptr, flag);
		if constexpr (Delete) {
			auto kv = db->FindOneAndDelete_And(vstd::get_rvalue_ptr(vstd::GetSpanIEnumerable_Move(std::span<INoSqlDatabase::CompareKey>(kvs))));
			PrintKV(kv);
		} else {

			auto kv = db->FindOne_And(vstd::get_rvalue_ptr(vstd::GetSpanIEnumerable_Move(std::span<INoSqlDatabase::CompareKey>(kvs))));
			PrintKV(kv);
		}
	}
	//Find All with kv vec
	else if constexpr (i == 4) {
		vstd::vector<INoSqlDatabase::CompareKey> kvs;
		kvs.emplace_back(testGuid, 3, flag);
		kvs.emplace_back(99999, nullptr, flag);
		if constexpr (Delete) {
			auto ite = db->FindAllAndDelete_And(vstd::get_rvalue_ptr(vstd::GetSpanIEnumerable_Move(std::span<INoSqlDatabase::CompareKey>(kvs))));
			PrintAll(ite);
		} else {
			auto ite = db->FindAll_And(vstd::get_rvalue_ptr(vstd::GetSpanIEnumerable_Move(std::span<INoSqlDatabase::CompareKey>(kvs))));
			PrintAll(ite);
		}
	}
	if constexpr (i != 0) {
		std::cout << "\n####################### After Deleted:\n";
		auto ite = db->GetAll();
		PrintAll(ite);
	}
	std::cout << "\n######################################################################\n";
}

template<bool b>
void TestDB(toolhub::db::Database* dbBase, toolhub::db::CompareFlag flag) {
	std::cout << "Get All\n\n";
	TestNoSql<0, b>(dbBase, flag);
	std::cout << "Find one GUID:2\n\n";
	TestNoSql<1, b>(dbBase, flag);
	std::cout << "Find all GUID:2\n\n";
	TestNoSql<2, b>(dbBase, flag);
	std::cout << "Find one GUID:3, 99999:null\n\n";

	TestNoSql<3, b>(dbBase, flag);
	std::cout << "Find all GUID:3, 99999:null\n\n";
	TestNoSql<4, b>(dbBase, flag);
}
void RunTest() {
	using namespace toolhub::db;

	DllFactoryLoader<Database> dll("VEngine_Database.dll", "Database_GetFactory");
	auto dbBase = dll();
	TestDB<true>(dbBase, CompareFlag::Equal);
}